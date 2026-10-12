#!/usr/bin/env python3
"""Kokoro-82M TTS server - a llama-swap-managed model.

llama-swap launches this like any llama-server (`--port ${PORT}`), health-checks
`/health`, and proxies requests to it - which is what puts `kokoro-82m` on the HUD
as an ordinary model row. Speech-to-text is its own llama-swap model (parakeet-server.py).

Endpoints:
  GET  /health            -> 200 {"status":"ok","tts_model":"Kokoro-82M", ...}
  POST /tts               -> {"text": "...", "voice": "am_onyx"} -> WAV bytes
  POST /v1/audio/speech   -> OpenAI shape: {"input": "...", "voice": "..."} -> WAV
                             (model field accepted and ignored - this IS the model)

Environment: its own Python 3.12 venv - install torch (a CUDA build) FIRST, then kokoro==0.9.4,
or pip pulls a CPU torch. Weights load from a local hexgrad/Kokoro-82M download (KOKORO_MODEL),
never hf.co at runtime, so it serves with the internet down.

⚠️ With two or more GPUs, set CUDA_DEVICE_ORDER=PCI_BUS_ID and CUDA_VISIBLE_DEVICES in llama-swap's
environment: torch defaults to FASTEST_FIRST ordering, which may not match nvidia-smi.
"""

import argparse
import io
import json
import os
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL_DIR = os.environ.get("KOKORO_MODEL", "/opt/llm/models/voice/Kokoro-82M")
DEFAULT_VOICE = os.environ.get("KOKORO_VOICE", "am_onyx")   # any voice file in voices/
# 1.2x Kokoro's own pace, our taste; KOKORO_SPEED=1.0 for stock. A client's OpenAI `speed` multiplies it.
SPEED = float(os.environ.get("KOKORO_SPEED", "1.2"))

PIPELINE = None
READY = threading.Event()
LOCK = threading.Lock()          # kokoro inference is not proven thread-safe


def load():
    global PIPELINE
    # ⛔ repo_id cannot be a filesystem path - huggingface_hub validates its shape and
    # raises before kokoro looks at the disk (hit 2026-08-19). Local weights go in
    # through KModel(config=, model=) instead, and voices load per-call by FILE PATH.
    # It must serve with the internet down, so nothing here may touch hf.co.
    import torch
    from kokoro import KModel, KPipeline
    model = KModel(config=os.path.join(MODEL_DIR, "config.json"),
                   model=os.path.join(MODEL_DIR, "kokoro-v1_0.pth"))
    if torch.cuda.is_available():
        model = model.to("cuda")
    # lang 'a' = American English.
    PIPELINE = KPipeline(lang_code="a", model=model.eval())
    # Warm it so the first real request does not pay torch's lazy init. A warm-up failure
    # must NOT kill this thread and leave READY unset (that is 503s forever with no clue on
    # the log): report it and mark ready anyway - the pipeline object is real, and a failed
    # request then surfaces its own error instead of a silent wall.
    try:
        wav = synth("ready", DEFAULT_VOICE)
        print("kokoro loaded, warmup wav %d bytes" % len(wav), flush=True)
    except Exception:
        import traceback
        traceback.print_exc()
        print("kokoro loaded, WARMUP FAILED - first request will surface the real error", flush=True)
    READY.set()


def synth(text, voice, speed=1.0):
    import numpy as np
    import soundfile as sf
    chunks = []
    # voice by FILE, not by name: a bare name sends KPipeline to the hub.
    vpath = os.path.join(MODEL_DIR, "voices", voice + ".pt")
    with LOCK:
        for _, _, audio in PIPELINE(text, voice=vpath, speed=SPEED * speed):
            chunks.append(audio)
    if not chunks:
        raise RuntimeError("kokoro produced no audio (empty text? text=%r)" % text[:80])
    audio = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
    audio = np.asarray(audio, dtype=np.float32)   # Kokoro's own level (about -26 dBFS)
    buf = io.BytesIO()
    sf.write(buf, audio, 24000, format="WAV")
    return buf.getvalue()


CITATION = re.compile(r"\s*(\[\^?[1-9]\d*(?:[,\s-]+\d+)*\])+")   # [1], [1][3], [1, 2], [^2]
MD_LINK = re.compile(r"\[([^\]]+)\]\((?:https?://|/)[^)]*\)")   # [text](url) -> text
# A model that tries a tool call with no tools offered writes it as text: our voice model (Qwen3.5-9B)
# spoke "<tool_call> <function=generate_chat_completion> ..." (2026-10-11). Markup is never speech.
TOOL_CALL = re.compile(r"<tool_call>.*?(?:</tool_call>|$)", re.S)
TAG = re.compile(r"</?[A-Za-z_][\w:=.-]*[^>]{0,200}>")
URL = re.compile(r"<?https?://\S+?>?(?=[\s)\],.;:!?]*(?:\s|$))")


def speakable(text):
    """What a reader would say: web-search citation markers made Open WebUI read "one" at the end
    of every sentence (2026-10-03); a link is read as its text, a bare URL not at all."""
    text = TOOL_CALL.sub("", text)
    text = TAG.sub("", text)
    text = MD_LINK.sub(r"\1", text)
    text = CITATION.sub("", text)
    text = URL.sub("", text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code, body, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.split("?")[0] == "/health":
            if not READY.is_set():
                # 503 while loading: llama-swap holds requests until 200.
                return self._send(503, json.dumps({"status": "loading"}).encode())
            return self._send(200, json.dumps({
                "status": "ok", "tts_model": "Kokoro-82M",
                "default_voice": DEFAULT_VOICE}).encode())
        self._send(404, b'{"error":"not found"}')

    def do_POST(self):
        path = self.path.split("?")[0]
        if path not in ("/tts", "/v1/audio/speech"):
            return self._send(404, b'{"error":"not found"}')
        if not READY.is_set():
            return self._send(503, b'{"error":"loading"}')
        try:
            n = int(self.headers.get("Content-Length") or 0)
            req = json.loads(self.rfile.read(n) or b"{}")
            # /tts takes "text" (an older client's shape); the OpenAI route
            # takes "input". Accept either on both - the distinction buys nothing.
            raw = req.get("text") or req.get("input") or ""
            if not raw.strip():
                return self._send(400, b'{"error":"no text"}')
            text = speakable(raw) or "."   # all citations or links: a pause, never a 400
            wav = synth(text, req.get("voice") or DEFAULT_VOICE, float(req.get("speed") or 1.0))
            return self._send(200, wav, "audio/wav")
        except Exception as e:
            return self._send(500, json.dumps({"error": str(e)}).encode())

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    args = ap.parse_args()
    threading.Thread(target=load, daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print("kokoro server on 127.0.0.1:%d (model loading in background)" % args.port,
          flush=True)
    srv.serve_forever()
