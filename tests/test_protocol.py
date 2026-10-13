from __future__ import annotations

from typing import Any

import pytest
from livekit.agents import llm
from livekit.agents.llm import function_tool

from livekit.plugins.polyai.realtime import TurnDetection, protocol

# every key, at any depth, the service accepts in a session.update from this plugin
ACCEPTED_SESSION_KEYS = {
    "type",
    "model",
    "instructions",
    "tools",
    "output_modalities",
    "max_output_tokens",
    "audio",
    "poly_response_language",
}


@function_tool
async def look_up_reservation(confirmation_code: str, last_name: str = "") -> str:
    """Find a reservation.

    Args:
        confirmation_code: Confirmation code, for example GM40912.
        last_name: The guest's last name.
    """
    return "{}"


@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        ("https://api.studio.poly.ai/v1", "wss://api.studio.poly.ai/v1/realtime"),
        ("https://api.studio.poly.ai/v1/", "wss://api.studio.poly.ai/v1/realtime"),
        ("http://127.0.0.1:8080/v1", "ws://127.0.0.1:8080/v1/realtime"),
        ("wss://example.test/v1/realtime", "wss://example.test/v1/realtime"),
        ("https://example.test/v1?session_id=abc", "wss://example.test/v1/realtime?session_id=abc"),
    ],
)
def test_realtime_url(base_url: str, expected: str) -> None:
    assert protocol.realtime_url(base_url) == expected


def test_initial_session_sends_only_accepted_fields() -> None:
    session = protocol.session_config(
        instructions="Be brief.",
        tools=protocol.to_wire_tools([look_up_reservation]),
        turn_detection=TurnDetection(create_response=False),
        max_output_tokens=512,
        response_language="es-ES",
        initial=True,
    )
    assert set(session) <= ACCEPTED_SESSION_KEYS
    assert session["model"] == "dialog-rsn-1"
    assert session["output_modalities"] == ["text"]
    assert session["audio"] == {
        "input": {
            "format": {"type": "audio/pcm", "rate": 24000},
            "turn_detection": {"type": "server_vad", "create_response": False},
        }
    }
    # the service rejects the whole update for any of these
    assert "output" not in session["audio"]
    assert "voice" not in session and "tool_choice" not in session


def test_later_updates_carry_only_what_changed() -> None:
    assert protocol.session_config(instructions="New.") == {
        "type": "realtime",
        "instructions": "New.",
    }


def test_function_tool_schema() -> None:
    [tool] = protocol.to_wire_tools([look_up_reservation])
    assert set(tool) == {"type", "name", "description", "parameters"}
    assert tool["type"] == "function"
    assert tool["name"] == "look_up_reservation"
    assert tool["parameters"]["required"] == ["confirmation_code"]


def test_raw_function_tool_keeps_only_known_keys() -> None:
    raw = llm.function_tool(
        lambda raw_arguments: "",
        raw_schema={
            "name": "ping",
            "description": "Ping.",
            "parameters": {"type": "object", "properties": {}},
            "strict": True,
            "meta": {"x": 1},
        },
    )
    assert protocol.to_wire_tools([raw]) == [
        {
            "type": "function",
            "name": "ping",
            "description": "Ping.",
            "parameters": {"type": "object", "properties": {}},
        }
    ]


def test_unsupported_tools_are_skipped() -> None:
    class NotAFunction(llm.ProviderTool):
        pass

    assert protocol.to_wire_tools([NotAFunction(id="provider")]) == []


def test_function_call_items_keep_the_full_call_id() -> None:
    call_id = "call_" + "a" * 32
    call = llm.FunctionCall(id="item_1", call_id=call_id, name="f", arguments="{}")
    output = llm.FunctionCallOutput(id="item_2", call_id=call_id, output="ok", is_error=False)
    assert protocol.to_wire_item(call)["call_id"] == call_id  # type: ignore[index]
    assert protocol.to_wire_item(output)["call_id"] == call_id  # type: ignore[index]


@pytest.mark.parametrize(
    ("role", "wire_role", "part_type"),
    [
        ("user", "user", "input_text"),
        ("assistant", "assistant", "output_text"),
        ("system", "system", "input_text"),
        ("developer", "system", "input_text"),
    ],
)
def test_message_items(role: Any, wire_role: str, part_type: str) -> None:
    msg = llm.ChatMessage(id="m1", role=role, content=["Hello"])
    assert protocol.to_wire_item(msg) == {
        "id": "m1",
        "type": "message",
        "role": wire_role,
        "content": [{"type": part_type, "text": "Hello"}],
    }


def test_message_without_text_is_not_sent() -> None:
    msg = llm.ChatMessage(id="m1", role="user", content=[llm.ImageContent(image="https://x/y.png")])
    assert protocol.to_wire_item(msg) is None


def test_from_wire_items() -> None:
    user_audio = protocol.from_wire_item(
        {"id": "u1", "type": "message", "role": "user", "content": [{"type": "input_audio"}]}
    )
    assert isinstance(user_audio, llm.ChatMessage) and user_audio.content == []

    reply = protocol.from_wire_item(
        {
            "id": "a1",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Hi"}],
        }
    )
    assert isinstance(reply, llm.ChatMessage) and reply.content == ["Hi"]

    call = protocol.from_wire_item(
        {"id": "c1", "type": "function_call", "call_id": "call_1", "name": "f", "arguments": "{}"}
    )
    assert isinstance(call, llm.FunctionCall) and call.call_id == "call_1"

    with pytest.raises(ValueError):
        protocol.from_wire_item({"id": "x", "type": "mcp_call"})


def test_describe_error_names_the_field() -> None:
    error = {
        "type": "invalid_request_error",
        "code": "unknown_parameter",
        "message": "Unknown parameter.",
        "param": "session.voice",
    }
    assert protocol.describe_error(error) == (
        "[unknown_parameter] Unknown parameter. (param: session.voice)"
    )
