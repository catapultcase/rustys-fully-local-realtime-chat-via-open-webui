#!/usr/bin/env bash
# Realtime voice bridge for Open WebUI (README.md). Paths are ours (/opt/llm); change to suit.
# STT is llama-swap's parakeet-tdt-0.6b-v3 (parakeet-server.py), shared with Open WebUI's Standard mode, not a copy in here.
# The voice-side model is chosen by voice-model.sh (local | remote); unset = the local small model, thinking off.
[ -f /opt/llm/s2s/voice-model.env ] && . /opt/llm/s2s/voice-model.env
VOICE_LLM_URL=${VOICE_LLM_URL:-http://127.0.0.1:8080/v1}
VOICE_LLM_MODEL=${VOICE_LLM_MODEL:-qwen3.5-9b}
THINK=()
# Thinking on at an effort: the bridge sends reasoning_effort instead of enable_thinking=false.
[ "${VOICE_LLM_THINK:-off}" != off ] && THINK=(--no_responses_api_disable_thinking --responses_api_reasoning_effort "$VOICE_LLM_THINK")
exec /opt/llm/s2s/venv/bin/speech-to-speech serve \
  --host 0.0.0.0 --port 8765 \
  --stt openai \
  --openai_stt_base_url http://127.0.0.1:8080/v1 \
  --openai_stt_api_key none \
  --openai_stt_model parakeet-tdt-0.6b-v3 \
  --openai_stt_response_format json \
  --smart_turn_model_path /opt/llm/models/voice/smart-turn-v3/smart-turn-v3.2-cpu.onnx \
  --llm_backend chat-completions \
  --model_name "$VOICE_LLM_MODEL" \
  --responses_api_base_url "$VOICE_LLM_URL" \
  "${THINK[@]}" \
  --responses_api_api_key none \
  --responses_api_stream \
  --tts openai \
  --openai_tts_base_url http://127.0.0.1:8080/v1 \
  --openai_tts_api_key none \
  --openai_tts_model kokoro-82m \
  --openai_tts_voice am_onyx \
  --openai_tts_response_format wav \
  --openai_tts_sample_rate 24000 \
  --enable_live_transcription
