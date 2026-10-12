#!/usr/bin/env python3
# set-realtime-prompt.py - writes realtime-call-prompt.txt (beside this script) into Open WebUI's Realtime call
# prompt (Admin -> Settings -> Audio -> Voice calls -> Prompt Template). Pasting the text into that field does
# the same; this is for scripted setups. Open WebUI's default prompt left our voice model (Qwen3.5-9B) answering
# factual questions itself, inventing facts, and saying "let me check" without handing off. Ours hands off
# everything but small talk: 14/14 turns right against 6/14, same model and tool. Results are read word for
# word rather than rephrased (README.md).
#
# Idempotent. Runs INSIDE the Open WebUI container (SQLite database); back up webui.db first, restart after:
#
#   docker cp set-realtime-prompt.py open-webui:/tmp/ && docker cp realtime-call-prompt.txt open-webui:/tmp/
#   docker exec open-webui python3 /tmp/set-realtime-prompt.py && docker restart open-webui
import json
import os
import sqlite3
import time

KEY = "audio.realtime.prompt_template"
prompt = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "realtime-call-prompt.txt")).read().strip()
db = sqlite3.connect("/app/backend/data/webui.db")
cur = db.execute("UPDATE config SET value = ?, updated_at = ? WHERE key = ?", (json.dumps(prompt), int(time.time()), KEY))
if cur.rowcount == 0:
    db.execute("INSERT INTO config (key, value, updated_at) VALUES (?, ?, ?)", (KEY, json.dumps(prompt), int(time.time())))
db.commit()
print("%s: %d chars" % (KEY, len(prompt)))
