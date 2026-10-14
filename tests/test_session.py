"""RealtimeSession against a scripted fake of the service."""

from __future__ import annotations

import asyncio
import base64

import aiohttp
import pytest
from livekit.agents import llm

from livekit import rtc
from livekit.plugins import polyai

from .conftest import FAST_RETRY
from .fake_server import CALL_ID, FakeDialogRSN1, echo_items, item_added, response_events

RealtimeSession = polyai.realtime.RealtimeSession


async def collect(gen: llm.GenerationCreatedEvent) -> tuple[str, list[llm.FunctionCall]]:
    text = ""
    async for msg in gen.message_stream:
        async for delta in msg.text_stream:
            text += delta
    return text, [call async for call in gen.function_stream]


def test_missing_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DIALOG_API_KEY", raising=False)
    with pytest.raises(ValueError, match="DIALOG_API_KEY"):
        polyai.realtime.RealtimeModel()


def test_capabilities() -> None:
    caps = polyai.realtime.RealtimeModel(api_key="k").capabilities
    assert not caps.audio_output
    assert caps.turn_detection and caps.user_transcription
    assert not caps.per_response_tool_choice and not caps.auto_tool_reply_generation


async def test_connects_with_api_key_header(
    session: RealtimeSession, server: FakeDialogRSN1
) -> None:
    assert server.headers[0]["X-API-KEY"] == "test-key"
    [update] = server.of_type("session.update")
    assert update["session"]["output_modalities"] == ["text"]
    assert "output" not in update["session"]["audio"]


async def test_generate_reply_matches_response_without_metadata(
    session: RealtimeSession, server: FakeDialogRSN1
) -> None:
    fut = session.generate_reply()
    [create] = await server.wait_for("response.create")
    assert "response" not in create  # per-response options are ignored by the service
    await server.send(*response_events(1, text="Hello there"))

    gen = await asyncio.wait_for(fut, 2)
    assert gen.user_initiated and gen.response_id == "resp_1"
    text, calls = await collect(gen)
    assert text.strip() == "Hello there" and calls == []
    assert session.chat_ctx.items[-1].text_content == "Hello there"  # type: ignore[union-attr]


async def test_rejected_reply_fails_fast_and_frees_its_place(
    session: RealtimeSession, server: FakeDialogRSN1
) -> None:
    first = session.generate_reply()
    [create] = await server.wait_for("response.create")
    await server.send(
        {
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "code": "conversation_already_has_active_response",
                "message": "busy",
                "event_id": create["event_id"],
            },
        }
    )
    with pytest.raises(llm.RealtimeError) as exc:
        await asyncio.wait_for(first, 2)
    assert exc.value.code == "conversation_already_has_active_response"

    second = session.generate_reply()
    await server.wait_for("response.create", 2)
    await server.send(*response_events(2, text="Second"))
    assert (await asyncio.wait_for(second, 2)).response_id == "resp_2"


async def test_replies_are_matched_in_order(
    session: RealtimeSession, server: FakeDialogRSN1
) -> None:
    first, second = session.generate_reply(), session.generate_reply()
    await server.wait_for("response.create", 2)
    await server.send(*response_events(1, text="One"), *response_events(2, text="Two"))
    assert (await asyncio.wait_for(first, 2)).response_id == "resp_1"
    assert (await asyncio.wait_for(second, 2)).response_id == "resp_2"


async def test_server_initiated_reply(session: RealtimeSession, server: FakeDialogRSN1) -> None:
    created: list[llm.GenerationCreatedEvent] = []
    session.on("generation_created", created.append)
    await server.send(*response_events(1, text="Unprompted"))
    await asyncio.sleep(0.1)
    assert len(created) == 1 and not created[0].user_initiated


async def test_cancelled_reply_is_discarded(
    session: RealtimeSession, server: FakeDialogRSN1
) -> None:
    created: list[llm.GenerationCreatedEvent] = []
    session.on("generation_created", created.append)
    fut = session.generate_reply()
    await server.wait_for("response.create")
    fut.cancel()
    await server.wait_for("response.cancel")
    # the response arrives after the caller gave up on it
    await server.send(*response_events(1, text="Too late"))
    await server.wait_for("response.cancel", 2)
    assert created == []


async def test_tool_call_keeps_the_service_call_id(
    session: RealtimeSession, server: FakeDialogRSN1
) -> None:
    server.auto["conversation.item.create"] = echo_items
    fut = session.generate_reply()
    await server.wait_for("response.create")
    call = {"call_id": CALL_ID, "name": "look_up_reservation", "arguments": '{"code": "GM1"}'}
    await server.send(*response_events(1, call=call))
    _, [fn_call] = await collect(await asyncio.wait_for(fut, 2))
    assert fn_call.call_id == CALL_ID and fn_call.arguments == '{"code": "GM1"}'

    # LiveKit adds the tool result to the chat context and syncs it
    chat_ctx = session.chat_ctx.copy()
    chat_ctx.items.append(llm.FunctionCallOutput(call_id=CALL_ID, output="{}", is_error=False))
    await session.update_chat_ctx(chat_ctx)

    [create] = server.of_type("conversation.item.create")
    assert create["item"]["type"] == "function_call_output"
    assert create["item"]["call_id"] == CALL_ID
    assert "previous_item_id" not in create  # the service appends regardless


async def test_update_chat_ctx_waits_for_the_service(
    session: RealtimeSession, server: FakeDialogRSN1
) -> None:
    server.auto["conversation.item.create"] = echo_items
    server.auto["conversation.item.delete"] = echo_items
    chat_ctx = llm.ChatContext.empty()
    chat_ctx.add_message(role="user", content="Hi", id="m1")
    chat_ctx.add_message(role="assistant", content="Hello", id="m2")
    await session.update_chat_ctx(chat_ctx)
    assert [i.id for i in session.chat_ctx.items] == ["m1", "m2"]

    chat_ctx.items.pop(0)
    await session.update_chat_ctx(chat_ctx)
    assert [i.id for i in session.chat_ctx.items] == ["m2"]


async def test_update_chat_ctx_surfaces_rejections(
    session: RealtimeSession, server: FakeDialogRSN1
) -> None:
    def reject(event: dict) -> list[dict]:
        return [
            {
                "type": "error",
                "error": {
                    "code": "invalid_request_error",
                    "message": "no",
                    "event_id": event["event_id"],
                },
            }
        ]

    server.auto["conversation.item.create"] = reject
    chat_ctx = llm.ChatContext.empty()
    chat_ctx.add_message(role="user", content="Hi")
    # a rejected item is logged, not raised, and doesn't wait out the timeout
    await asyncio.wait_for(session.update_chat_ctx(chat_ctx), 2)
    assert session.chat_ctx.items == []


async def test_audio_turn_transcript(session: RealtimeSession, server: FakeDialogRSN1) -> None:
    started: list[llm.InputSpeechStartedEvent] = []
    transcripts: list[llm.InputTranscriptionCompleted] = []
    session.on("input_speech_started", started.append)
    session.on("input_audio_transcription_completed", transcripts.append)

    for _ in range(5):  # 500 ms at 48 kHz; the resampler holds some back
        session.push_audio(rtc.AudioFrame(b"\0\0" * 4800, 48000, 1, 4800))
    appends = await server.wait_for("input_audio_buffer.append")
    # 100 ms of 24 kHz PCM16 per event
    assert len(base64.b64decode(appends[0]["audio"])) == 4800

    user_item = {
        "id": "u1",
        "type": "message",
        "role": "user",
        "content": [{"type": "input_audio"}],
    }
    await server.send(
        {"type": "input_audio_buffer.speech_started", "item_id": "u1", "audio_start_ms": 0},
        {"type": "input_audio_buffer.speech_stopped", "item_id": "u1", "audio_end_ms": 900},
        item_added(user_item),
        {
            "type": "conversation.item.input_audio_transcription.delta",
            "item_id": "u1",
            "delta": "Hi",
        },
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "u1",
            "transcript": "Hi there",
        },
    )
    await asyncio.sleep(0.1)
    assert len(started) == 1
    assert [(t.transcript, t.is_final) for t in transcripts] == [("Hi", False), ("Hi there", True)]
    assert transcripts[-1].turn_started_at is not None
    assert session.chat_ctx.items[0].text_content == "Hi there"  # type: ignore[union-attr]


async def test_truncate_rewrites_the_reply_to_what_was_heard(
    session: RealtimeSession, server: FakeDialogRSN1
) -> None:
    fut = session.generate_reply()
    await server.wait_for("response.create")
    await server.send(*response_events(1, text="I can tell you about your booking"))
    await collect(await asyncio.wait_for(fut, 2))

    session.truncate(
        message_id="item_msg_1", modalities=["text"], audio_end_ms=900, audio_transcript="I can"
    )
    [delete] = await server.wait_for("conversation.item.delete")
    [create] = await server.wait_for("conversation.item.create")
    assert delete["item_id"] == "item_msg_1"
    assert create["item"]["id"] == "item_msg_1"
    assert create["item"]["content"] == [{"type": "output_text", "text": "I can"}]


async def test_rejected_session_update_is_reported(
    session: RealtimeSession, server: FakeDialogRSN1
) -> None:
    errors: list[llm.RealtimeModelError] = []
    session.on("error", errors.append)
    await server.send(
        {
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "code": "unknown_parameter",
                "message": "Unknown parameter.",
                "param": "session.foo",
                "event_id": "session_update_x",
            },
        }
    )
    await asyncio.sleep(0.1)
    assert len(errors) == 1 and errors[0].recoverable
    assert "session.foo" in str(errors[0].error)


async def test_reconnect_rebuilds_the_conversation(
    session: RealtimeSession, server: FakeDialogRSN1
) -> None:
    server.auto["conversation.item.create"] = echo_items
    reconnected: list[llm.RealtimeSessionReconnectedEvent] = []
    session.on("session_reconnected", reconnected.append)

    await session.update_instructions("Be brief.")
    chat_ctx = llm.ChatContext.empty()
    chat_ctx.add_message(role="user", content="Hi", id="m1")
    await session.update_chat_ctx(chat_ctx)
    await server.ws.close(code=1011)  # the service's close code for a server error

    await server.wait_connections(2)
    await asyncio.sleep(0.2)
    assert len(reconnected) == 1
    replayed_update = server.of_type("session.update")[-1]["session"]
    assert replayed_update["instructions"] == "Be brief."
    assert server.of_type("conversation.item.create")[-1]["item"]["id"] == "m1"
    assert [i.id for i in session.chat_ctx.items] == ["m1"]


async def test_bad_key_does_not_retry(server: FakeDialogRSN1) -> None:
    server.handshake_status = 401
    errors: list[llm.RealtimeModelError] = []
    async with aiohttp.ClientSession() as http:
        model = polyai.realtime.RealtimeModel(
            api_key="bad", base_url=server.url, http_session=http, conn_options=FAST_RETRY
        )
        sess = model.session()
        sess.on("error", errors.append)
        await asyncio.sleep(0.3)
        await sess.aclose()
    assert len(server.headers) == 1
    assert len(errors) == 1 and not errors[0].recoverable
