#!/usr/bin/env python3
"""Patch the installed speech-to-speech for Open WebUI's Realtime call. Four fixes.

First: a follow-up during a hand-off is answered, not refused.

Open WebUI's Realtime call asks for a response (response.create) after every finished user turn. When the
user speaks while generate_chat_completion is still running, the bridge refused it: an error event
(function_call_output_pending, or the LLM's "Cannot generate a response while function call outputs are
pending"), and Open WebUI ends the call on ANY provider error ("Voice provider rejected a request").
Seen 2026-10-11: "Is it running?" asked during a hand-off ended the call.

The patch generates anyway, on the per-response copy of the chat, with each pending call given a placeholder
result saying the chat model is still working. The voice model can then say "still on it". The real
conversation is untouched, so the real result completes the call as before.

Second fix in the same run: Mute. Open WebUI sends input_audio_buffer.clear on Mute; the bridge has no
handler for it ("Unknown client event type"), answered with an error, and every Mute ended the call
(2026-10-11). The router now acknowledges it with input_audio_buffer.cleared, as OpenAI's API does;
muting already stops the audio, so there is nothing else to clear.

Third: a hand-off's answer is spoken directly. Open WebUI asks for a response to speak each finished
result (metadata call_id); unpatched, the voice model (Qwen3.5-9B) re-read the whole call and rewrote the
answer before Kokoro started, ~1.2-1.6 s after the text was already in the chat (2026-10-11), and even
with the word-for-word prompt it could drift. A completed result's answer now goes through the bridge's
normal output path (sentence chunks, history, timings) without the LLM call. A failed result still goes
to the model, which explains the failure as Open WebUI asks.

Fourth: Open WebUI's "[Historical voice assistant transcript...]" label lines are stripped from a hand-off's
answer before it is spoken; the chat model copied them from its history and Kokoro read them aloud.

Idempotent; re-run after every reinstall of speech-to-speech (README.md):
  /opt/llm/s2s/venv/bin/python /opt/llm/s2s/patch-bridge.py
"""

import pathlib
import re
import sys

import speech_to_speech

ROOT = pathlib.Path(speech_to_speech.__file__).parent
MARK = "HOMELAB PATCH followup-during-handoff"
DIRECT_MARK = "HOMELAB PATCH direct-answer"

CHAT_METHOD = f'''
    def fill_pending_tool_outputs(self) -> None:
        """{MARK}: give every pending call a placeholder result, on a per-response copy only."""
        with self._lock:
            call_ids = list(self._pending_tool_calls)
        for call_id in call_ids:
            self.append_tool_output(
                call_id,
                RealtimeConversationItemFunctionCallOutput(
                    type="function_call_output",
                    call_id=call_id,
                    output=json.dumps({{
                        "status": "in_progress",
                        "note": "The chat model is still working on this request. Its answer will arrive "
                        "separately. Do not answer it yourself and do not call the tool for it again.",
                    }}),
                ),
            )

    def has_pending_tool_calls(self) -> bool:'''

GUARD = re.compile(
    r"(?P<head>\n(?P<indent>[ ]+)if not (?:is_out_of_band\(response\)|out_of_band) and active_chat\.has_pending_tool_calls\(\):\n)"
    r"[ ]+yield EndOfResponse\((?:.|\n)*?\n[ ]+\)\n[ ]+return\n"
)

REFUSAL = re.compile(
    r"(?P<head>\n(?P<indent>[ ]+)if candidate_chat\.has_pending_tool_calls\(\):\n)"
    r"[ ]+return self\.make_error\(\n[ ]+message=\"Cannot create a response while function call outputs are pending\.\",\n"
    r"[ ]+_type=\"function_call_output_pending\",\n[ ]+\)\n"
)


def patch(path, fn, mark=MARK):
    text = path.read_text()
    if mark in text:
        print("already patched:", path.relative_to(ROOT))
        return
    new = fn(text)
    if new == text:
        sys.exit("pattern not found in %s - speech-to-speech changed; re-derive the patch" % path)
    path.write_text(new)
    print("patched:", path.relative_to(ROOT))


def chat(text):
    return text.replace("\n    def has_pending_tool_calls(self) -> bool:", CHAT_METHOD, 1)


def guards(text, expect):
    new, n = GUARD.subn(
        lambda m: m["head"] + m["indent"] + "    active_chat.fill_pending_tool_outputs()  # " + MARK + "\n", text)
    if n != expect:
        sys.exit("expected %d guard(s), found %d" % (expect, n))
    return new


def refusal(text):
    new, n = REFUSAL.subn(lambda m: m["head"] + m["indent"] + "    pass  # " + MARK + "\n", text)
    if n != 1:
        sys.exit("expected 1 refusal, found %d" % n)
    return new


CLEAR = f'''
    if raw.get("type") == "input_audio_buffer.clear":  # {MARK}: Open WebUI's Mute
        from openai.types.realtime.input_audio_buffer_cleared_event import InputAudioBufferClearedEvent

        await send_correlated(
            [InputAudioBufferClearedEvent(event_id=service._next_event_id(), type="input_audio_buffer.cleared")]
        )
        return

    event = service.parse_client_event(raw)'''


def clear(text):
    return text.replace("\n    event = service.parse_client_event(raw)", CLEAR, 1)


DIRECT_HELPER = f'''

def _homelab_direct_answer(response, chat):
    """{DIRECT_MARK}: the answer of a completed generate_chat_completion result this response should speak, or None."""
    import json as _json

    from openai.types.realtime.conversation_item import RealtimeConversationItemFunctionCallOutput

    metadata = getattr(response, "metadata", None) or {{}}
    call_id = metadata.get("call_id") if isinstance(metadata, dict) else None
    if not call_id or getattr(response, "instructions", None):
        return None
    for item in reversed(chat.copy().buffer):
        if isinstance(item, RealtimeConversationItemFunctionCallOutput) and item.call_id == call_id:
            try:
                result = _json.loads(item.output)
            except (TypeError, ValueError):
                return None
            answer = result.get("answer") if isinstance(result, dict) else None
            if isinstance(answer, str):
                # Open WebUI labels earlier spoken replies in the chat model's history with this line, and
                # the chat model copied it into its own answer, which was then read aloud (2026-10-11).
                answer = "\\n".join(
                    line for line in answer.splitlines()
                    if not line.strip().startswith("[Historical voice assistant transcript")
                ).strip()
            if result.get("status") == "completed" and isinstance(answer, str) and answer:
                return answer
            return None
    return None


class BaseOpenAICompatibleHandler'''

DIRECT_CALL = f'''        direct = _homelab_direct_answer(response, original_chat)  # {DIRECT_MARK}: speak the answer as written
        if direct is not None:
            from openai.types.realtime.realtime_conversation_item_assistant_message import Content as _Content

            def _direct_events(_api_response):
                yield TextDelta(text=direct)
                yield AssistantMessage(content=[_Content(type="output_text", text=direct)])
                yield Usage(input_tokens=0, output_tokens=0)
                yield ProviderResponseEnd()

            yield from self._generate(
                active_chat,
                original_chat,
                turn,
                optional_kwargs,
                serialize_fn=lambda _chat: ["direct"],
                request_fn=lambda _api_input, _kwargs: None,
                event_iterator_fn=_direct_events,
            )
            return
        yield from self._generate(active_chat, original_chat, turn, optional_kwargs)
'''


def direct(text):
    call = "        yield from self._generate(active_chat, original_chat, turn, optional_kwargs)\n"
    cls = "\n\nclass BaseOpenAICompatibleHandler"
    if text.count(call) != 1 or text.count(cls) != 1:
        sys.exit("direct-answer anchors not found once - speech-to-speech changed; re-derive the patch")
    return text.replace(cls, DIRECT_HELPER, 1).replace(call, DIRECT_CALL, 1)


patch(ROOT / "LLM/chat.py", chat)
patch(ROOT / "LLM/base_openai_compatible_language_model.py", lambda t: guards(t, 2))
patch(ROOT / "LLM/language_model.py", lambda t: guards(t, 1))
patch(ROOT / "api/openai_realtime/handlers/response.py", refusal)
patch(ROOT / "api/openai_realtime/websocket_router.py", clear)
patch(ROOT / "LLM/base_openai_compatible_language_model.py", direct, DIRECT_MARK)


# Fourth: streamed results (RustyRealTime, github.com/catapultcase/RustyRealTime). With Open WebUI's opt-in
# AUDIO_REALTIME_STREAM_RESULTS, the final text of a hand-off's answer arrives sentence by sentence as
# openwebui.function_call_output.delta {call_id, delta} while the chat model writes it. Each batch is spoken as an
# out-of-band response as soon as the bridge is free. The final function_call_output still carries the whole
# answer: the response that speaks it skips everything up to the end of what was already spoken.
STREAM_MARK = "HOMELAB PATCH stream-results"

STREAM_ROUTER = f'''
_HOMELAB_STREAMS: dict = {{}}  # {STREAM_MARK}: session_id -> call_id -> {{"pending", "spoken", "flushing"}}


def _homelab_stream_delta(session_id, raw):
    call_id, delta = raw.get("call_id"), raw.get("delta")
    if not isinstance(call_id, str) or not isinstance(delta, str) or not delta:
        return None
    stream = _HOMELAB_STREAMS.setdefault(session_id, {{}}).setdefault(
        call_id, {{"pending": "", "spoken": "", "flushing": False}}
    )
    stream["pending"] += delta
    return stream


def _homelab_stream_final(session_id, raw):
    """The response that speaks a streamed call's final result learns what was already spoken."""
    response = raw.get("response")
    metadata = response.get("metadata") if isinstance(response, dict) else None
    call_id = metadata.get("call_id") if isinstance(metadata, dict) else None
    stream = (_HOMELAB_STREAMS.get(session_id) or {{}}).pop(call_id, None) if call_id else None
    if stream and stream["spoken"]:
        metadata["openwebui_spoken"] = stream["spoken"]
    if stream:
        stream["pending"] = ""  # the final answer covers it


async def _homelab_stream_flush(unit, session_id, call_id, stream, send_correlated):
    """Speak pending sentences once no response is running; never sends an error (Open WebUI hangs up on one)."""
    service = unit.service
    try:
        for _ in range(1200):
            if not stream["pending"] or (_HOMELAB_STREAMS.get(session_id) or {{}}).get(call_id) is not stream:
                return
            st = service._state(session_id)
            if st.in_response or st.response_pending:
                await asyncio.sleep(0.05)
                continue
            text, stream["pending"] = stream["pending"], ""
            event = service.parse_client_event(
                {{
                    "type": "response.create",
                    "response": {{
                        "conversation": "none",
                        "metadata": {{"call_id": call_id, "openwebui_stream_text": text}},
                    }},
                }}
            )
            result = service.handle_response_create(session_id, event) if event is not None else None
            if result is None or result.type == "error":
                stream["pending"] = text + stream["pending"]
                await asyncio.sleep(0.05)
                continue
            stream["spoken"] += text
            unit.cancel_scope.new_response()
            response_key = service._state(session_id).current_response_key
            await send_correlated([result])
            service.response.mark_response_created_sent(session_id, response_key)
    finally:
        stream["flushing"] = False


async def _dispatch_client_event('''

STREAM_DISPATCH = f'''    if raw.get("type") == "openwebui.function_call_output.delta":  # {STREAM_MARK}
        stream = _homelab_stream_delta(session_id, raw)
        if stream is not None and not stream["flushing"]:
            stream["flushing"] = True
            asyncio.create_task(_homelab_stream_flush(unit, session_id, raw["call_id"], stream, send_correlated))
        return
    if raw.get("type") == "response.create":  # {STREAM_MARK}
        _homelab_stream_final(session_id, raw)

    event = service.parse_client_event(raw)'''


def stream_router(text):
    head = "\n\nasync def _dispatch_client_event("
    anchor = "\n    event = service.parse_client_event(raw)"
    if text.count(head) != 1 or text.count(anchor) != 1 or "import asyncio" not in text:
        sys.exit("stream-results router anchors not found once - speech-to-speech changed; re-derive the patch")
    return text.replace(head, "\n" + STREAM_ROUTER, 1).replace(anchor, "\n" + STREAM_DISPATCH, 1)


STREAM_HELPER_OLD = '''    metadata = getattr(response, "metadata", None) or {}
    call_id = metadata.get("call_id") if isinstance(metadata, dict) else None
    if not call_id or getattr(response, "instructions", None):
        return None
'''
STREAM_HELPER_NEW = f'''    metadata = getattr(response, "metadata", None) or {{}}
    call_id = metadata.get("call_id") if isinstance(metadata, dict) else None
    if not call_id or getattr(response, "instructions", None):
        return None
    streamed = metadata.get("openwebui_stream_text")  # {STREAM_MARK}: sentences of a still-streaming answer
    if isinstance(streamed, str) and streamed:
        return streamed
    spoken = metadata.get("openwebui_spoken")  # {STREAM_MARK}: already said while it streamed
'''
STREAM_ANSWER_OLD = '''            if result.get("status") == "completed" and isinstance(answer, str) and answer:
                return answer
'''
STREAM_ANSWER_NEW = f'''            if result.get("status") == "completed" and isinstance(answer, str) and answer:
                if isinstance(spoken, str) and spoken:  # {STREAM_MARK}: speak only what follows it
                    at = answer.find(spoken)
                    if at >= 0:
                        return answer[at + len(spoken):].strip()
                return answer
'''
STREAM_EVENTS_OLD = '''            def _direct_events(_api_response):
                yield TextDelta(text=direct)
'''
STREAM_EVENTS_NEW = f'''            def _direct_events(_api_response):
                if direct:  # {STREAM_MARK}: a streamed answer can be fully spoken already
                    yield TextDelta(text=direct)
'''


def stream_helper(text):
    for old in (STREAM_HELPER_OLD, STREAM_ANSWER_OLD, STREAM_EVENTS_OLD):
        if text.count(old) != 1:
            sys.exit("stream-results helper anchors not found once - run the direct-answer patch first")
    return (
        text.replace(STREAM_HELPER_OLD, STREAM_HELPER_NEW, 1)
        .replace(STREAM_ANSWER_OLD, STREAM_ANSWER_NEW, 1)
        .replace(STREAM_EVENTS_OLD, STREAM_EVENTS_NEW, 1)
    )


patch(ROOT / "api/openai_realtime/websocket_router.py", stream_router, STREAM_MARK)
patch(ROOT / "LLM/base_openai_compatible_language_model.py", stream_helper, STREAM_MARK)


# Fifth: streamed text drops Open WebUI's history label too. Open WebUI shows the chat model earlier spoken replies
# under "[Historical voice assistant transcript; not current task status]", and the model can open its own answer
# with it; the direct-answer path already drops that line, the streamed path now does the same (2026-10-13).
STREAM_LABEL_MARK = "HOMELAB PATCH stream-label"
STREAM_LABEL_OLD = '''    if isinstance(streamed, str) and streamed:
        return streamed
'''
STREAM_LABEL_NEW = f'''    if isinstance(streamed, str) and streamed:  # {STREAM_LABEL_MARK}: same history-label rule as a finished answer
        streamed = "\\n".join(
            line for line in streamed.splitlines(keepends=False)
            if not line.strip().startswith("[Historical voice assistant transcript")
        )
        return streamed if streamed.strip() else ""
'''
STREAM_SPOKEN_OLD = '''    spoken = metadata.get("openwebui_spoken")  # {STREAM_MARK}: already said while it streamed
'''
STREAM_SPOKEN_NEW = f'''    spoken = metadata.get("openwebui_spoken")  # {STREAM_MARK}: already said while it streamed
    if isinstance(spoken, str):  # {STREAM_LABEL_MARK}: compared with the answer after the same label rule
        spoken = "\\n".join(
            line for line in spoken.splitlines() if not line.strip().startswith("[Historical voice assistant transcript")
        ).strip()
'''


def stream_label(text):
    for old in (STREAM_LABEL_OLD, STREAM_SPOKEN_OLD.format(STREAM_MARK=STREAM_MARK)):
        if text.count(old) != 1:
            sys.exit("stream-label anchors not found once - run the stream-results patch first")
    return text.replace(STREAM_LABEL_OLD, STREAM_LABEL_NEW, 1).replace(
        STREAM_SPOKEN_OLD.format(STREAM_MARK=STREAM_MARK), STREAM_SPOKEN_NEW, 1
    )


patch(ROOT / "LLM/base_openai_compatible_language_model.py", stream_label, STREAM_LABEL_MARK)


# Sixth: a streamed answer keeps its whole text in the voice model's history. The streamed sentences are spoken in
# out-of-band responses, which stay out of the conversation, and the final result's response then speaks only the
# rest, often nothing. Its assistant message was that rest, so the voice model's own history showed an empty reply
# after the chat model's answer, and it later answered "check online" itself (2026-10-13). The final response now
# records the whole answer, as it did before streaming, and speaks only what was not said yet.
STREAM_HISTORY_MARK = "HOMELAB PATCH stream-history"
STREAM_HISTORY_ANSWER_OLD = '''                    if at >= 0:
                        return answer[at + len(spoken):].strip()
'''
STREAM_HISTORY_ANSWER_NEW = f'''                    if at >= 0:
                        metadata["openwebui_full_answer"] = answer  # {STREAM_HISTORY_MARK}
                        return answer[at + len(spoken):].strip()
'''
STREAM_HISTORY_EVENTS_OLD = '''                yield AssistantMessage(content=[_Content(type="output_text", text=direct)])
'''
STREAM_HISTORY_EVENTS_NEW = f'''                full = (getattr(response, "metadata", None) or {{}}).get("openwebui_full_answer") or direct  # {STREAM_HISTORY_MARK}
                yield AssistantMessage(content=[_Content(type="output_text", text=full)])
'''


def stream_history(text):
    for old in (STREAM_HISTORY_ANSWER_OLD, STREAM_HISTORY_EVENTS_OLD):
        if text.count(old) != 1:
            sys.exit("stream-history anchors not found once - run the stream-results patch first")
    return text.replace(STREAM_HISTORY_ANSWER_OLD, STREAM_HISTORY_ANSWER_NEW, 1).replace(
        STREAM_HISTORY_EVENTS_OLD, STREAM_HISTORY_EVENTS_NEW, 1
    )


patch(ROOT / "LLM/base_openai_compatible_language_model.py", stream_history, STREAM_HISTORY_MARK)
