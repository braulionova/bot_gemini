#!/bin/bash
# Gemini Flash Trading Bot — startup script with auto-restart
cd /root/bot_gemini

while true; do
    echo "[$(date)] Starting Gemini Trading Bot..."
    python3 main.py
    EXIT_CODE=$?
    echo "[$(date)] Bot exited with code $EXIT_CODE"

    if [ $EXIT_CODE -eq 0 ]; then
        echo "Clean exit, not restarting."
        break
    fi

    echo "Restarting in 10 seconds..."
    sleep 10
done
