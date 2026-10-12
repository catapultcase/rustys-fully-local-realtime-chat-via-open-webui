# Rusty's Fully Local Realtime Chat via Open WebUI

Our setup for Open WebUI 0.12's **Realtime** call mode with no cloud anywhere. Small talk is answered
at once, you can talk over it or mute it, and real questions go to your chat model with its tools,
knowledge and history, and the answer is read back word for word. In daily use since 2026-10-11.

Open WebUI's Realtime mode expects an **OpenAI Realtime API** server. Almost no local inference server
speaks that protocol, so a bridge does it:
[`huggingface/speech-to-speech`](https://github.com/huggingface/speech-to-speech) run as a Realtime
server, with every model behind it served locally.

```
browser ⇄ Open WebUI ⇄ (websocket, OpenAI Realtime protocol) ⇄ speech-to-speech bridge
                                                                 ├─ ears:  Parakeet TDT v3   (llama-swap)
                                                                 ├─ turns: Silero VAD + Smart Turn v3.2 (CPU, in the bridge)
                                                                 ├─ voice model: Qwen3.5-9B  (llama-swap)
                                                                 └─ mouth: Kokoro-82M        (llama-swap)
Open WebUI ──(the hand-off: generate_chat_completion)──▶ your chat's selected model, unchanged
```

| Part | Model | Why |
|---|---|---|
| Speech-to-text | [Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3), [`servers/parakeet-server.py`](servers/parakeet-server.py) | a 3 s question in ~0.08 s. Silence and room noise come back empty (Whisper turned them into "Thank you.", and the chat model answered) |
| Turn end | Silero VAD + [Smart Turn v3.2](https://huggingface.co/pipecat-ai/smart-turn-v3) | tells a pause from being done; "um" gets a longer grace |
| Voice model | [Qwen3.5-9B](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF) UD-Q4_K_XL, thinking off | answers small talk, hands everything else off; ~0.5 s median reply |
| Text-to-speech | [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M), [`servers/kokoro-server.py`](servers/kokoro-server.py) | first audio in ~0.1 s |
| Chat model | whatever the chat has selected | unchanged: its prompt, tools, knowledge and filters all apply |

## Hardware

One always-on x86 Linux box with an NVIDIA GPU. Ours: Ubuntu Server 26.04, an RTX 3080 Ti (12 GB),
of which the three models use **9.1 GB** (Kokoro 1.0, Parakeet 1.8, Qwen3.5-9B 6.2 with its KV cache at
q8_0). Kokoro grows to ~1.3 GB with use, so leave headroom. The chat model runs elsewhere (ours is on a
pair of DGX Sparks); anything Open WebUI can talk to works.

The models are served by [llama-swap](https://github.com/mostlygeek/llama-swap) on port 8080, and the
bridge reaches all three at `127.0.0.1:8080/v1`. Any OpenAI-compatible server works if you change the
URLs in [`bridge/run.sh`](bridge/run.sh).

## Install

Paths are ours (`/opt/llm`); change them to suit. You need `uv`, Python 3.12 and `ffmpeg`.

1. **Download the models** (the servers load local files and never reach hf.co at runtime):
   - `nvidia/parakeet-tdt-0.6b-v3`: the `.nemo` file
   - `pipecat-ai/smart-turn-v3`: `smart-turn-v3.2-cpu.onnx`
   - `unsloth/Qwen3.5-9B-GGUF`: `Qwen3.5-9B-UD-Q4_K_XL.gguf`
   - `hexgrad/Kokoro-82M`: the whole repo (`config.json`, `kokoro-v1_0.pth`, `voices/`)

2. **The bridge venv** (`/opt/llm/s2s/venv`, Python 3.12). It also runs Parakeet:

   ```bash
   uv venv --python 3.12 /opt/llm/s2s/venv
   uv pip install --python /opt/llm/s2s/venv/bin/python \
     'speech-to-speech @ git+https://github.com/huggingface/speech-to-speech.git@14f9e33d5aa7a1d65c94cfb7b9c0f9c630310689' \
     nano-parakeet
   /opt/llm/s2s/venv/bin/python bridge/patch-bridge.py
   ```

   ⛔ **Use that commit, not PyPI's `speech-to-speech` 1.0.0.** 1.0.0 rejects Open WebUI's
   `chat_context_N` items ("ID must start with 'sys_'") and `conversation.item.delete`. The patch script
   checks its anchors and refuses to run if upstream has changed. A reinstall of the same commit keeps the
   patched files, so `uv pip uninstall speech-to-speech` first, then reinstall and patch again.

3. **The Kokoro venv** (`/opt/llm/kokoro/venv312`, Python 3.12, since kokoro needs < 3.13). Install a
   CUDA build of `torch` **first**, then `kokoro==0.9.4 soundfile`. If you do it the other way round,
   pip pulls a CPU torch.

4. **llama-swap**: copy the servers to `/opt/llm/` and merge [`llama-swap/voice.yaml`](llama-swap/voice.yaml)
   into your config. All three models go in one no-swap group and are preloaded at startup.

5. **The bridge service**: copy [`bridge/run.sh`](bridge/run.sh) and [`bridge/voice-model.sh`](bridge/voice-model.sh)
   to `/opt/llm/s2s/`. Copy [`bridge/s2s.service`](bridge/s2s.service) to `/etc/systemd/system/` and set
   `<user>`, then run `sudo systemctl enable --now s2s`. The bridge listens on `:8765` and logs to
   `/opt/llm/logs/s2s.log`.

6. **Open WebUI** ≥ 0.12, Admin → Settings → Audio → Voice calls:

   | Field | Value |
   |---|---|
   | Call mode | Realtime |
   | OpenAI API Base URL | `http://<bridge-box>:8765/v1` |
   | API Key | `none` (anything; the bridge checks no key) |
   | Voice Model | any text (the bridge chooses its own) |
   | Voice | **a Kokoro voice**, e.g. `am_onyx`. Open WebUI passes it through, and the default `marin` is not a Kokoro voice |
   | Input Transcription Model | `parakeet-tdt-0.6b-v3` |
   | Prompt Template | the contents of [`open-webui/realtime-call-prompt.txt`](open-webui/realtime-call-prompt.txt) (or run [`set-realtime-prompt.py`](open-webui/set-realtime-prompt.py)) |

   Give your users the **Allow Call** permission. The microphone needs HTTPS or localhost. For a plain
   `http://` LAN address, add it to `chrome://flags/#unsafely-treat-insecure-origin-as-secure`.

7. **Test it**: say hello (answered locally), ask "who directed the original Halloween?" (handed off,
   then read back), talk over the answer, mute and unmute, and ask "is it still running?" during a
   hand-off.

## What we changed from the defaults

### The call prompt

[`realtime-call-prompt.txt`](open-webui/realtime-call-prompt.txt) is Open WebUI's built-in prompt with
three changes:

- The voice model answers **only small talk**. Every factual question, check, correction or task is
  handed off, and when it's unsure it hands off.
- It never says "let me check" unless the hand-off happens in the same reply.
- Results are read **word for word**. Open WebUI's default rephrases them, and the spoken version
  drifting from the chat text was confusing.

With the default prompt, the 9B answered facts itself and made them up. It also once said "let me check"
and never handed off, which left the call silent. Our 14-turn hand-off check (5 small talk, 9 that need
the chat model) went from **6/14 to 14/14** with the same model.

### Bridge patches ([`bridge/patch-bridge.py`](bridge/patch-bridge.py))

Open WebUI ends the call on **any** error event from the provider ("Voice provider rejected a request").

| Patch | Before | After |
|---|---|---|
| Mute | `input_audio_buffer.clear` had no handler, so every Mute ended the call | acknowledged with `input_audio_buffer.cleared`, as OpenAI's API does |
| Follow-up during a hand-off | the bridge refused a reply while a tool result was pending, so asking "is it running?" ended the call | answered on a per-response copy of the chat where the pending call reads "still working" |
| Speaking a hand-off's answer | the voice model re-read the call and rewrote the answer, taking 1.2-1.6 s, and the result could drift | a completed answer goes straight to TTS: speech starts **~0.2 s** after the text lands, exactly as written. A failed result still goes to the model to explain |
| History label | the chat model sometimes copied Open WebUI's `[Historical voice assistant transcript; not current task status]` into its answer, and it was read aloud | the line is dropped before speaking; it stays in the chat text |

### Speech cleanup ([`servers/kokoro-server.py`](servers/kokoro-server.py))

Before speaking, Kokoro's server strips `<tool_call>` markup and tags, citation markers like `[1]` (web
search answers were read with "one" after every sentence), and URLs. A Markdown link is read as its text.
Speech is 1.2× Kokoro's pace; set `KOKORO_SPEED=1.0` for stock.

## A bigger voice model

[`bridge/voice-model.sh`](bridge/voice-model.sh) switches the voice model between two profiles: `local`
(the 9B on this box) and `remote` (a bigger model on another box). Fill in the profiles, then run
`voice-model.sh remote`. Switching restarts the bridge, which ends a call in progress.

Same 14-turn check:

| Voice model | Thinking | Right (two runs) | Median reply |
|---|---|---|---|
| Qwen3.8-27B Q4 + MTP, RTX 5090 | low | **14/14, 14/14** | 0.92 s |
| Qwen3.8-27B, RTX 5090 | off | 12/14, 13/14 | 0.41-0.51 s |
| Qwen3.5-9B, RTX 3080 Ti | **off** | 14/14, 13/14 | 0.51 s |
| Qwen3.5-9B, RTX 3080 Ti | on, capped at 256 tokens | 13/14, 13/14 | 0.92-1.23 s, and spilled `</think>` and half a tool call into the speech |

The count matters less than where they fail. When the 27B answers a fact itself, it's right. When the
9B does, it invents. Qwen3.5 has no effort levels, so the 9B stays thinking off. The 27B doesn't fit
beside the speech models on 12 GB. On a 24-32 GB card it would.

## Limits

- **Speech starts only when the chat model's answer is complete.** Open WebUI sends the hand-off result
  to the voice side only once it's finished, so a long answer means silence while the text writes itself
  into the chat. We've asked upstream to stream partial results so speech can start on the first sentence.
- **One call at a time**; the bridge refuses a second.
- Parakeet's weak spot is unusual names.
- Tested only on x86 Linux with NVIDIA.

## Credits and licenses

The scripts here are MIT ([LICENSE](LICENSE)). `patch-bridge.py` modifies an installed copy of
[speech-to-speech](https://github.com/huggingface/speech-to-speech) (Apache-2.0). The models carry their
own licenses; check each model card.
