#!/usr/bin/env python3
"""Parakeet TDT 0.6B v3 STT server - a llama-swap-managed model (README.md).

One Parakeet on the GPU for everything that needs speech-to-text: Open WebUI's Standard mode, any other
client on the OpenAI path, and the Realtime voice bridge. llama-swap launches it (`--port ${PORT}`), health-checks
`/health`, and proxies to it.

Endpoints:
  GET  /health                    -> 200 {"status":"ok", ...}, 503 while loading
  POST /v1/audio/transcriptions   -> OpenAI shape: multipart `file` (any format ffmpeg reads),
                                     `response_format` json (default) | text | verbose_json;
                                     `model` and `language` accepted and ignored

Environment: the voice bridge's venv (/opt/llm/s2s/venv: nano-parakeet, torch). The weights are the
nvidia/parakeet-tdt-0.6b-v3 .nemo downloaded beforehand (PARAKEET_NEMO), never hf.co at runtime - this box must serve with the internet down.
"""

import argparse
import json
import os
import subprocess
import threading
from email.parser import BytesParser
from email.policy import HTTP
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

NEMO = os.environ.get("PARAKEET_NEMO",
                      "/opt/llm/models/voice/parakeet-tdt-0.6b-v3/parakeet-tdt-0.6b-v3.nemo")
NAME = "parakeet-tdt-0.6b-v3"

MODEL = None
READY = threading.Event()
LOCK = threading.Lock()          # one CUDA decode graph; requests take turns


def load():
    # nano_parakeet.from_pretrained, minus hf_hub_download: the same steps on a local .nemo.
    global MODEL
    import sentencepiece as spm
    import torch
    from nano_parakeet import ParakeetTDT
    from nano_parakeet._loader import get_bundled_tokenizer_proto, load_nemo_state_dict, remap_state_dict
    model = ParakeetTDT()
    missing, _ = model.load_state_dict(remap_state_dict(load_nemo_state_dict(NEMO, map_location="cpu")),
                                       strict=False)
    if missing:
        raise RuntimeError("missing keys in %s: %s" % (NEMO, missing[:10]))
    model = model.to("cuda").eval()
    if torch.cuda.is_bf16_supported():
        model.encoder.to(torch.bfloat16)
        model.decoder.to(torch.bfloat16)
        model.joint.to(torch.bfloat16)
    sp = spm.SentencePieceProcessor()
    sp.LoadFromSerializedProto(get_bundled_tokenizer_proto())
    model.sp = sp
    model.warmup()
    MODEL = model
    READY.set()
    print("parakeet loaded from %s" % NEMO, flush=True)


def pcm16k(data):
    """Any container ffmpeg reads -> 16 kHz mono float32: Open WebUI sends webm, an agent's voice note WAV or Opus."""
    import numpy as np
    out = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", "pipe:0",
                          "-ar", "16000", "-ac", "1", "-f", "f32le", "pipe:1"],
                         input=data, capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.float32).copy()


def form(headers, body):
    msg = BytesParser(policy=HTTP).parsebytes(
        b"Content-Type: " + headers.get("Content-Type", "").encode() + b"\r\n\r\n" + body)
    fields = {}
    for part in msg.iter_parts():
        name = part.get_param("name", header="content-disposition")
        fields[name] = part.get_payload(decode=True)
    return fields


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
            return self._send(200, json.dumps({"status": "ok", "stt_model": NAME}).encode())
        self._send(404, b'{"error":"not found"}')

    def do_POST(self):
        if self.path.split("?")[0] != "/v1/audio/transcriptions":
            return self._send(404, b'{"error":"not found"}')
        if not READY.is_set():
            return self._send(503, b'{"error":"loading"}')
        try:
            n = int(self.headers.get("Content-Length") or 0)
            fields = form(self.headers, self.rfile.read(n))
            if not fields.get("file"):
                return self._send(400, b'{"error":"no file"}')
            audio = pcm16k(fields["file"])
            fmt = (fields.get("response_format") or b"json").decode()
            with LOCK:
                text = MODEL.transcribe(audio).strip() if audio.size else ""
            if fmt == "text":
                return self._send(200, text.encode(), "text/plain; charset=utf-8")
            body = {"text": text}
            if fmt == "verbose_json":
                body.update(language="en", duration=round(audio.size / 16000, 3), segments=[])
            return self._send(200, json.dumps(body).encode())
        except subprocess.CalledProcessError as e:
            return self._send(400, json.dumps({"error": "ffmpeg: " + e.stderr.decode(errors="replace")[-300:]}).encode())
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
    print("parakeet server on 127.0.0.1:%d (model loading in background)" % args.port, flush=True)
    srv.serve_forever()
