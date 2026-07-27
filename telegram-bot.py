import html
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

CONF_PATH = os.path.expanduser("~/.config/squeue-telegram/conf.json")
STATE_PATH = os.path.expanduser("~/.cache/squeue-telegram/state.json")

MIN_INTERVAL = 30  # don't hammer slurmctld
DEFAULT_INTERVAL = 300
LONG_POLL = 50  # seconds to block on getUpdates
SQUEUE_FMT = "%i|%j|%T|%t|%M|%L|%D|%R"

# ---------------------------------------------------------------- config/state

with open(CONF_PATH) as fh:
    _conf = json.load(fh)
TOKEN = _conf["token"]
CHAT_ID = str(_conf["chat_id"])

state = {"watching": False, "interval": DEFAULT_INTERVAL}


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


# --------------------------------------------------------------------- slurm


def run(cmd, timeout=60):
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        print("[warn] %s failed: %s" % (cmd[0], exc), file=sys.stderr)
        return None
    if proc.returncode != 0:
        print(
            "[warn] %s rc=%d: %s" % (cmd[0], proc.returncode, proc.stderr.strip()),
            file=sys.stderr,
        )
        return None
    return proc.stdout


def snapshot():
    """Return {job_id: {...}} or None if squeue could not be reached."""
    out = run(["squeue", "--me", "-h", "-o", SQUEUE_FMT])
    if out is None:
        return None
    jobs = {}
    for line in out.strip().splitlines():
        f = [p.strip() for p in line.split("|")]
        if len(f) < 8:
            print("[warn] unparsed squeue line: %r" % line, file=sys.stderr)
            continue
        jobs[f[0]] = {
            "name": f[1],
            "state": f[2],
            "st": f[3],
            "elapsed": f[4],
            "left": f[5],
            "nodes": f[6],
            "reason": f[7],
        }
    return jobs


def sacct(job_id, tries=3, delay=5):
    """Final accounting for a job that left the queue. sacct lags a little."""
    for attempt in range(tries):
        out = run(
            [
                "sacct",
                "-j",
                job_id,
                "-X",
                "-n",
                "-P",
                "-o",
                "JobID,JobName,State,ExitCode,Elapsed,MaxRSS",
            ]
        )
        if out and out.strip():
            f = out.strip().splitlines()[0].split("|")
            return {"state": f[2], "exit": f[3], "elapsed": f[4]}
        if attempt < tries - 1:
            time.sleep(delay)
    return None


def short_time(t):
    """SLURM elapsed ([DD-]HH:MM:SS or MM:SS) -> H:MM:SS, days folded into hours."""
    days = 0
    try:
        if "-" in t:
            d, _, t = t.partition("-")
            days = int(d)
        parts = [int(p) for p in t.split(":")]
    except ValueError:
        return t
    if len(parts) == 3:
        hours, minutes, seconds = parts
    elif len(parts) == 2:
        hours, minutes, seconds = 0, parts[0], parts[1]
    else:
        return t
    return "%d:%02d:%02d" % (days * 24 + hours, minutes, seconds)


def format_queue(jobs):
    if not jobs:
        return "Queue empty — nothing pending or running."
    rows = ["%-9s %-8s %-2s %8s %s" % ("JOBID", "NAME", "ST", "TIME", "WHY/NODE")]
    for jid, j in sorted(jobs.items()):
        where = j["reason"] if j["state"] == "RUNNING" else j["reason"].strip("()")
        rows.append(
            "%-9s %-8s %-2s %8s %s"
            % (jid, j["name"][:8], j["st"], short_time(j["elapsed"]), where)
        )
    return pre("\n".join(rows))


# ------------------------------------------------------------- change detect


def signature(jobs):
    """Compare on state only, not elapsed time -- otherwise every poll 'changes'."""
    return {jid: j["state"] for jid, j in jobs.items()}


def check_queue(prev):
    """Poll the queue, notify on real changes, return the new snapshot."""
    cur = snapshot()
    if cur is None:
        return prev  # slurmctld hiccup, keep old state
    if prev is None:
        return cur  # first run, just establish a baseline

    old, new = signature(prev), signature(cur)
    if old == new:
        return cur

    lines = []
    for jid in sorted(set(new) - set(old)):
        j = cur[jid]
        lines.append(
            "🆕 %s <b>%s</b> submitted (%s)" % (jid, html.escape(j["name"]), j["state"])
        )

    for jid in sorted(set(new) & set(old)):
        if old[jid] != new[jid]:
            j = cur[jid]
            extra = " on %s" % j["reason"] if j["state"] == "RUNNING" else ""
            lines.append(
                "▶️ %s <b>%s</b> %s → %s%s"
                % (jid, html.escape(j["name"]), old[jid], new[jid], html.escape(extra))
            )

    for jid in sorted(set(old) - set(new)):
        name = html.escape(prev[jid]["name"])
        acc = sacct(jid)
        if acc is None:
            lines.append(
                "⏹ %s <b>%s</b> left the queue (no sacct record yet)" % (jid, name)
            )
        else:
            ok = acc["state"].startswith("COMPLETED")
            lines.append(
                "%s %s <b>%s</b> %s — exit %s, ran %s"
                % (
                    "✅" if ok else "❌",
                    jid,
                    name,
                    html.escape(acc["state"]),
                    acc["exit"],
                    acc["elapsed"],
                )
            )

    if lines:
        send("\n".join(lines))
    return cur


# ------------------------------------------------------------------ commands

HELP = (
    "<b>Commands</b>\n"
    "/q — show the queue now\n"
    "/watch [sec] — notify me on changes\n"
    "/stop — stop notifying\n"
    "/interval &lt;sec&gt; — set poll interval\n"
    "/status — bot state\n"
    "/help — this message"
)


def handle(text, prev):
    """Returns (force_immediate_poll, snapshot)."""
    parts = text.strip().split()
    if not parts:
        return False, prev
    cmd = parts[0].split("@")[0].lower()
    arg = parts[1] if len(parts) > 1 else None

    if cmd == "/q":
        jobs = snapshot()
        send("Could not reach slurmctld." if jobs is None else format_queue(jobs))
        return False, jobs if jobs is not None else prev

    if cmd == "/watch":
        if arg:
            try:
                state["interval"] = max(MIN_INTERVAL, int(arg))
            except ValueError:
                send("Interval must be a number of seconds.")
                return False, prev
        state["watching"] = True
        save_state()
        send("Watching, polling every %ds." % state["interval"])
        return True, prev

    if cmd == "/stop":
        state["watching"] = False
        save_state()
        send("Stopped. Use /q for on-demand checks.")
        return False, prev

    if cmd == "/interval":
        try:
            state["interval"] = max(MIN_INTERVAL, int(arg))
        except (TypeError, ValueError):
            send("Usage: /interval 300  (minimum %ds)" % MIN_INTERVAL)
            return False, prev
        save_state()
        send("Interval set to %ds." % state["interval"])
        return state["watching"], prev

    if cmd == "/status":
        send(
            "👀 Watching: %s\nInterval: %ds\nHost: <code>%s</code>"
            % (state["watching"], state["interval"], html.escape(os.uname().nodename))
        )
        return False, prev

    send(HELP)
    return False, prev


# ---------------------------------------------------------------------- main


def main():
    load_state()

    # Drop any commands queued while the bot was down.
    offset = 0
    stale = api("getUpdates", {"offset": -1, "timeout": 0}, timeout=20)
    if stale and stale.get("result"):
        offset = stale["result"][-1]["update_id"] + 1

    send(
        "Bot online on <code>%s</code>\n👀 Watching: %s (%ds)\n\n%s"
        % (html.escape(os.uname().nodename), state["watching"], state["interval"], HELP)
    )

    prev = snapshot()
    send("Could not reach slurmctld." if prev is None else format_queue(prev))
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
            time.sleep(5)  # network blip; back off and retry
        else:
            for upd in resp.get("result", []):
                offset = upd["update_id"] + 1
                msg = upd.get("message") or {}
                if str(msg.get("chat", {}).get("id")) != CHAT_ID:
                    continue  # ignore everyone else
                text = msg.get("text")
                if not text:
                    continue
                force, prev = handle(text, prev)
                if force:
                    next_poll = 0

        if state["watching"] and time.time() >= next_poll:
            prev = check_queue(prev)
            next_poll = time.time() + state["interval"]


if __name__ == "__main__":
    print("Starting squeue-telegram bot on %s" % os.uname().nodename, file=sys.stderr)
    register_commands()
    try:
        main()
    except KeyboardInterrupt:
        pass
