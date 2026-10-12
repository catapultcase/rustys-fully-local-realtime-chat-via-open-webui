#!/usr/bin/env bash
# voice-model.sh - which model is the Realtime voice bridge's voice-side model (README.md).
#   voice-model.sh           show the current choice
#   voice-model.sh local     the small model on this box, thinking off (the default)
#   voice-model.sh remote    a bigger model on another box, thinking low
# Writes /opt/llm/s2s/voice-model.env (read by run.sh) and restarts the bridge, which ends a call.
# Fill in the two profiles. Ours: local = Qwen3.5-9B on a 3080 Ti; remote = Qwen3.8-27B on a 5090.
set -euo pipefail
ENV=/opt/llm/s2s/voice-model.env
case "${1:-}" in
  "") cat "$ENV" 2>/dev/null || echo "VOICE_MODEL=local (default, no $ENV)"; exit 0 ;;
  local)  url=http://127.0.0.1:8080/v1;        model="<local-model-id>";  think=off ;;
  remote) url="http://<remote-box-ip>:8080/v1";  model="<remote-model-id>"; think=low ;;
  *) echo "usage: $0 [local|remote]" >&2; exit 2 ;;
esac
if [ "$1" != local ]; then
  # Load the model now, so a remote box that is off fails here, not on a call.
  curl -sf -m 300 "$url/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":1}" >/dev/null \
    || { echo "$url does not answer - start it and retry; nothing changed" >&2; exit 1; }
fi
printf 'VOICE_MODEL=%s\nVOICE_LLM_URL=%s\nVOICE_LLM_MODEL=%s\nVOICE_LLM_THINK=%s\n' "$1" "$url" "$model" "$think" > "$ENV"
pid=$(systemctl show s2s -p MainPID --value)
[ "$pid" != 0 ] && kill -9 "$pid"   # Restart=on-failure brings it back with the new env (no sudo needed)
for _ in $(seq 1 40); do sleep 3; ss -ltn | grep -q ':8765 ' && break; done
ss -ltn | grep -q ':8765 ' && echo "voice model: $1 ($model, thinking $think) - bridge up" || { echo "bridge did not come back; see /opt/llm/logs/s2s.log" >&2; exit 1; }
