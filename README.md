# squeue-telegram

Telegram bots (stdlib only, no pip install) that monitors:
- `squeue --me` on a SLURM cluster (runs on a login node)
- `nvidia-smi` on linux servers

The bots notify you of job changes. 

## Create the Telegram bot

1. Message [@BotFather](https://t.me/BotFather), send `/newbot`, and follow the prompts. It gives you a **token**.
2. Start a chat with your new bot and send it a message, the content is not important.
3. Get your **chat_id** by visiting `https://api.telegram.org/bot<TOKEN>/getUpdates` and reading `message.chat.id`.

## Store credentials

Store the **token** and the **chat_id** somewhere *safe*.

```
mkdir -p ~/.config/squeue-telegram
cat > ~/.config/squeue-telegram/conf.json <<'EOF'
{"token": "123456:AA...", "chat_id": 987654321}
EOF
chmod 600 ~/.config/squeue-telegram/conf.json
```

If you change where they are stored, don't forget to change the paths in the python script.

## Run
The script lives on the login nodes; to avoid killing it when closing the terminal, open a new tmux session and start it from there.

```
tmux new -s tgbot
python3 telegram-bot.py
```

Detach with `Ctrl-b, d`.

## Commands

- `/q` — show the queue right now
- `/disk` — show the disk usage (only linux servers) 
- `/watch [seconds]` — turn on change notifications (default 300s)
- `/stop` — turn off notifications
- `/interval <sec>` — change the polling interval
- `/status` — show bot state
- `/help` — command list
