"""
gpu_bot.py -- Telegram bot for monitoring GPUs on a local headless box.

Outbound-only: works fine behind a VPN/NAT, no inbound port needed.
Stdlib only, Python 3.6+.

Setup
-----
  mkdir -p ~/.config/gpu-telegram
  cat > ~/.config/gpu-telegram/conf.json <<'EOF'
  {"token": "123456:AA...", "chat_id": 987654321,
   "disks": ["/", "/home", "/data"], "disk_threshold": 90}
  EOF
  chmod 600 ~/.config/gpu-telegram/conf.json

Run
---
  tmux new -s gpubot && python3 gpu_bot.py
  # or as a user service, see notes at the bottom

Commands
--------
  /q                 GPUs + processes right now
  /disk              disk usage
  /watch [seconds]   notify on process start/exit and GPUs freeing up
  /stop              stop notifying
  /interval <sec>    change poll interval
  /status            bot state
  /help              command list
"""

import html
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

CONF_PATH = os.path.expanduser("~/.config/gpu_bot/conf.json")
STATE_PATH = os.path.expanduser("~/.cache/gpu_bot/state.json")

MIN_INTERVAL = 15
DEFAULT_INTERVAL = 300
LONG_POLL = 50

GPU_Q = "index,uuid,utilization.gpu,memory.used,memory.total,temperature.gpu"
APP_Q = "pid,gpu_uuid,used_gpu_memory"

# ---------------------------------------------------------------- config/state

with open(CONF_PATH) as fh:
    _conf = json.load(fh)
TOKEN = _conf["token"]
CHAT_ID = str(_conf["chat_id"])
DISKS = _conf.get("disks", ["/"])
DISK_THRESHOLD = _conf.get("disk_threshold", 90)

state = {"watching": False, "interval": DEFAULT_INTERVAL, "disk_alerted": {}}


def load_state():
    try:
        with open(STATE_PATH) as fh:
            state.update(json.load(fh))
    except (OSError, ValueError):
        pass


def save_state():
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(state, fh)
    os.replace(tmp, STATE_PATH)


# ------------------------------------------------------------------ telegram
COMMANDS = [
    {"command": "q", "description": "show the queue now"},
    {"command": "disk", "description": "disk usage"},
    {"command": "watch", "description": "notify on changes — /watch [sec]"},
    {"command": "stop", "description": "stop notifying"},
    {"command": "interval", "description": "set poll interval — /interval <sec>"},
    {"command": "status", "description": "bot state"},
    {"command": "help", "description": "command list"},
]


def register_commands():
    api("setMyCommands", {"commands": json.dumps(COMMANDS)}, timeout=20)


def api(method, params=None, timeout=70):
    url = "https://api.telegram.org/bot%s/%s" % (TOKEN, method)
    data = urllib.parse.urlencode(params or {}).encode()
    req = urllib.request.Request(url, data=data)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.load(resp)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        print("[warn] telegram %s failed: %s" % (method, exc), file=sys.stderr)
        return None


def send(text):
    api("sendMessage", {"chat_id": CHAT_ID, "parse_mode": "HTML", "text": text})


def pre(text):
    return "<pre>%s</pre>" % html.escape(text)


# --------------------------------------------------------------------- shell


def run(cmd, timeout=30, allowed_rc=(0,)):
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        print("[warn] %s failed: %s" % (cmd[0], exc), file=sys.stderr)
        return None
    if proc.returncode not in allowed_rc:
        print(
            "[warn] %s rc=%d: %s" % (cmd[0], proc.returncode, proc.stderr.strip()),
            file=sys.stderr,
        )
        return None
    return proc.stdout


def fmt_secs(n):
    """Seconds -> H:MM:SS, days folded into hours (same convention as squeue bot)."""
    try:
        n = int(n)
    except (TypeError, ValueError):
        return "?"
    return "%d:%02d:%02d" % (n // 3600, (n % 3600) // 60, n % 60)


def short_cmd(args, width=12):
    """Turn a full command line into something readable in 12 chars."""
    parts = args.split()
    if not parts:
        return "?"
    exe = os.path.basename(parts[0])
    if exe.startswith("python") or exe in ("torchrun", "accelerate", "deepspeed"):
        for p in parts[1:]:
            if p.endswith(".py"):
                return os.path.basename(p)[:width]
            if not p.startswith("-"):
                return os.path.basename(p)[:width]
    return exe[:width]


def cmd_line(args):
    """/path/to/python train.py --config x -> train.py --config x"""
    parts = args.split()
    if not parts:
        return "?"
    exe = os.path.basename(parts[0])
    rest = parts[1:]
    if exe.startswith("python") or exe in ("torchrun", "accelerate", "deepspeed", "uv"):
        for i, p in enumerate(rest):  # prefer the script itself
            if p.endswith(".py"):
                return " ".join([os.path.basename(p)] + rest[i + 1 :])
        for i, p in enumerate(rest):  # else first non-flag token
            if not p.startswith("-"):
                return " ".join([os.path.basename(p)] + rest[i + 1 :])
        return exe
    return " ".join([exe] + rest)


def trunc(s, width):
    return s if len(s) <= width else s[: width - 1] + "…"


def local_ip():
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
        finally:
            sock.close()
    except OSError:
        return "?"


# ----------------------------------------------------------------------- gpus


def gpu_snapshot():
    """Return (gpus, procs) or (None, None) if nvidia-smi is unreachable."""
    out = run(["nvidia-smi", "--query-gpu=" + GPU_Q, "--format=csv,noheader,nounits"])
    if out is None:
        return None, None

    gpus, by_uuid = {}, {}
    for line in out.strip().splitlines():
        f = [p.strip() for p in line.split(",")]
        if len(f) < 6:
            continue
        idx = f[0]
        gpus[idx] = {
            "util": f[2],
            "mem_used": f[3],
            "mem_total": f[4],
            "temp": f[5],
            "pids": [],
        }
        by_uuid[f[1]] = idx

    out = run(
        ["nvidia-smi", "--query-compute-apps=" + APP_Q, "--format=csv,noheader,nounits"]
    )
    if out is None:
        return gpus, {}

    raw = {}
    for line in out.strip().splitlines():
        f = [p.strip() for p in line.split(",")]
        if len(f) < 3:
            continue
        pid, idx = f[0], by_uuid.get(f[1], "?")
        raw.setdefault(pid, {"gpus": [], "mem": 0})
        raw[pid]["gpus"].append(idx)
        try:
            raw[pid]["mem"] += int(f[2])
        except ValueError:
            pass
        if idx in gpus:
            gpus[idx]["pids"].append(pid)

    if not raw:
        return gpus, {}

    # One ps call for owner, elapsed seconds and command line.
    # rc=1 simply means every pid exited between the two calls.
    out = run(
        ["ps", "-o", "pid=,user=,etimes=,args=", "-p", ",".join(sorted(raw))],
        allowed_rc=(0, 1),
    )
    procs = {}
    for line in (out or "").strip().splitlines():
        f = line.split(None, 3)
        if len(f) < 4:
            continue
        pid = f[0]
        if pid not in raw:
            continue
        procs[pid] = {
            "user": f[1],
            "secs": f[2],
            "cmd": cmd_line(f[3]),  # "cmd": short_cmd(f[3]),
            "gpus": sorted(raw[pid]["gpus"]),
            "mem": raw[pid]["mem"],
        }
    return gpus, procs


def format_gpus(gpus, procs):
    rows = ["%3s %5s %9s %5s" % ("GPU", "UTIL", "MEMORY", "TEMP")]
    for idx in sorted(gpus):
        g = gpus[idx]
        try:
            mem = "{:.1f}/{:.0f}G".format(
                int(g["mem_used"]) / 1024.0,
                int(g["mem_total"]) / 1024.0,
            )
        except ValueError:
            mem = "?"
        rows.append("%3s %4s%% %9s %4sC" % (idx, g["util"], mem, g["temp"]))

    if procs:
        rows.append("")
        rows.append("%3s %7s %-8s %8s" % ("GPU", "PID", "USER", "TIME"))
        for pid, p in sorted(procs.items(), key=lambda kv: kv[1]["gpus"]):
            rows.append(
                "%3s %7s %-8s %8s"
                % (",".join(p["gpus"]), pid, p["user"][:8], fmt_secs(p["secs"]))
            )
            rows.append("  > " + trunc(p["cmd"], 44))
    else:
        rows.append("")
        rows.append("No compute processes — both GPUs free.")
    return pre("\n".join(rows))


# ----------------------------------------------------------------------- disk


def disk_usage():
    out = []
    for path in DISKS:
        try:
            st = os.statvfs(path)
        except OSError:
            continue
        total = st.f_blocks * st.f_frsize
        free = st.f_bavail * st.f_frsize
        if not total:
            continue
        out.append(
            {
                "path": path,
                "pct": int(round(100.0 * (total - free) / total)),
                "free_gb": free / (1024.0**3),
                "total_gb": total / (1024.0**3),
            }
        )
    return out


def format_disks(usage):
    return "\n".join(
        [
            "💾 <code>%s</code> (%.0fGB) at %d%% — %.0fGB free"  # noqa: UP031
            % (html.escape(d["path"]), d["total_gb"], d["pct"], d["free_gb"])
            for d in usage
        ]
    )


def check_disks():
    """Alert on crossing the threshold, with hysteresis so it fires once."""
    lines = []
    alerted = state.get("disk_alerted", {})
    for d in disk_usage():
        was = alerted.get(d["path"], False)
        if d["pct"] >= DISK_THRESHOLD and not was:
            lines.append(
                f"💾 <code>{html.escape(d['path'])}</code> ({d['total_gb']:.0f}GB) at {d['pct']}%% — {d['free_gb']:.0f}GB free"
            )
            alerted[d["path"]] = True
        elif d["pct"] < DISK_THRESHOLD - 5 and was:
            lines.append(
                f"💾 <code>{html.escape(d['path'])}</code> ({d['total_gb']:.0f}GB) back down to {d['pct']}%% — {d['free_gb']:.0f}GB free"
            )
            alerted[d["path"]] = False
    if lines:
        state["disk_alerted"] = alerted
        save_state()
    return lines


# ------------------------------------------------------------- change detect


def check_all(prev_procs, prev_busy):
    """Notify on process start/exit and GPUs freeing up."""
    gpus, procs = gpu_snapshot()
    if gpus is None:
        return prev_procs, prev_busy
    if prev_procs is None:
        return procs, {i for i in gpus if gpus[i]["pids"]}

    lines = []

    for pid in sorted(set(procs) - set(prev_procs)):
        p = procs[pid]
        lines.append(
            f"🔥 GPU#{p['gpus'][0]} — {html.escape(p['user'])} started <code>{pid}</code>\n"
            + pre(f"{html.escape(p['cmd'].replace(' --', '\n--'))}")
        )

    for pid in sorted(set(prev_procs) - set(procs)):
        p = prev_procs[pid]
        lines.append(
            f"✅ GPU#{p['gpus'][0]} — <code>{pid}</code> exited after {fmt_secs(p['secs'])}\n"
            + pre(f"{html.escape(p['cmd'].replace(' --', '\n--'))}")
        )

    busy = {i for i in gpus if gpus[i]["pids"]}
    # freed = prev_busy - busy
    # if freed:
    #    lines.append("✅ GPU {} now free".format(",".join(sorted(freed))))

    lines.extend(check_disks())

    if lines:
        send("\n".join(lines))
    return procs, busy


# ------------------------------------------------------------------ commands

HELP = (
    "<b>Commands</b>\n"
    "/q — GPUs and processes now\n"
    "/disk — disk usage\n"
    "/watch [sec] — notify on changes\n"
    "/stop — stop notifying\n"
    "/interval &lt;sec&gt; — set poll interval\n"
    "/status — bot state\n"
    "/help — this message"
)


def handle(text):
    parts = text.strip().split()
    if not parts:
        return False
    cmd = parts[0].split("@")[0].lower()
    arg = parts[1] if len(parts) > 1 else None

    if cmd == "/q":
        gpus, procs = gpu_snapshot()
        send("nvidia-smi unavailable." if gpus is None else format_gpus(gpus, procs))
        return False

    if cmd == "/disk":
        send(format_disks(disk_usage()))
        return False

    if cmd == "/watch":
        if arg:
            try:
                state["interval"] = max(MIN_INTERVAL, int(arg))
            except ValueError:
                send("Interval must be a number of seconds.")
                return False
        state["watching"] = True
        save_state()
        send(f"Watching, polling every {state['interval']}s.")
        return True

    if cmd == "/stop":
        state["watching"] = False
        save_state()
        send("Stopped. Use /q for on-demand checks.")
        return False

    if cmd == "/interval":
        try:
            state["interval"] = max(MIN_INTERVAL, int(arg))
        except (TypeError, ValueError):
            send(f"Usage: /interval 300  (minimum {MIN_INTERVAL}s)")
            return False
        save_state()
        send(f"Interval set to {state['interval']}s.")
        return state["watching"]

    if cmd == "/status":
        send(
            f"🤖 <code>{html.escape(os.uname().nodename)}</code>"
            + f"\n🛜 Host: <code>{html.escape(local_ip())}</code>"
            + (
                f"\n👀 watching: {state['watching']} (⏳ {state['interval']}s)"
                if state["watching"]
                else "\n💤 sleeping"
            )
        )

        return False

    send(HELP)
    return False


# ---------------------------------------------------------------------- main


def main():
    load_state()

    offset = 0
    stale = api("getUpdates", {"offset": -1, "timeout": 0}, timeout=20)
    if stale and stale.get("result"):
        offset = stale["result"][-1]["update_id"] + 1

    gpus, procs = gpu_snapshot()
    busy = {i for i in gpus if gpus[i]["pids"]} if gpus else set()

    send(
        f"🤖 Bot online on <code>{html.escape(os.uname().nodename)}</code>\n🛜 <code>{html.escape(local_ip())}</code>\n"
        + (
            f"👀 watching: {state['watching']} (⏳ {state['interval']}s)"
            if state["watching"]
            else "💤 sleeping"
        )
        + f"\n\n{HELP}"
    )
    send("nvidia-smi unavailable." if gpus is None else format_gpus(gpus, procs))

    next_poll = time.time() + state["interval"]

    while True:
        if state["watching"]:
            wait = int(max(1, min(LONG_POLL, next_poll - time.time())))
        else:
            wait = LONG_POLL

        resp = api(
            "getUpdates",
            {"offset": offset, "timeout": wait, "allowed_updates": '["message"]'},
            timeout=wait + 20,
        )

        if resp is None:
            time.sleep(5)
        else:
            for upd in resp.get("result", []):
                offset = upd["update_id"] + 1
                msg = upd.get("message") or {}
                if str(msg.get("chat", {}).get("id")) != CHAT_ID:
                    continue
                text = msg.get("text")
                if not text:
                    continue
                if handle(text):
                    next_poll = 0

        if state["watching"] and time.time() >= next_poll:
            procs, busy = check_all(procs, busy)
            next_poll = time.time() + state["interval"]


if __name__ == "__main__":
    register_commands()
    try:
        main()
    except KeyboardInterrupt:
        pass
