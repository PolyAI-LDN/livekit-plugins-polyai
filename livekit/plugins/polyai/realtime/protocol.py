"""Dialog-RSN-1 wire format: what the session sends, and how server items map to LiveKit's.

Everything here is a pure function of its arguments, so it can be tested without a server.
Dialog-RSN-1 speaks the OpenAI Realtime GA protocol with a narrower session schema. It rejects
a whole ``session.update`` if any field is outside that schema, so this module only ever builds
the fields the service accepts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import urlparse, urlunparse

from livekit.agents import llm

from ..log import logger

DEFAULT_BASE_URL = "https://api.us.poly.ai/v1"
MODEL = "dialog-rsn-1"

SAMPLE_RATE = 24000
"""Input audio rate sent to the service. It is the service's default, so the session never
has to declare it."""
NUM_CHANNELS = 1

# The service closes the socket with these codes when the request can never succeed, so
# reconnecting would only repeat the failure.
FATAL_CLOSE_CODES = frozenset({1008, 4401, 4403})

# Error codes that mean the key or account is unusable. Anything else is worth a retry.
FATAL_ERROR_CODES = frozenset({"invalid_api_key", "policy_violation"})


@dataclass
class TurnDetection:
    """Dialog-RSN-1's own turn detection.

    The model decides when the caller has finished, in the context of the conversation, so
    there are no silence thresholds to tune. Describe how patient it should be in the
    instructions instead.

    Args:
        create_response: Reply as soon as the model judges the turn over. Set it to False to
            only commit the caller's turn and send ``generate_reply`` yourself.
    """

    create_response: bool = True


def realtime_url(base_url: str) -> str:
    """Turn an HTTP(S) or WS(S) base URL into the realtime WebSocket endpoint."""
    parsed = urlparse(base_url)
    scheme = {"http": "ws", "https": "wss"}.get(parsed.scheme, parsed.scheme)
    path = parsed.path.rstrip("/")
    if not path.endswith("/realtime"):
        path = f"{path}/realtime"
    return urlunparse((scheme, parsed.netloc, path, "", parsed.query, ""))


def session_config(
    *,
    instructions: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    turn_detection: TurnDetection | None = None,
    max_output_tokens: int | Literal["inf"] | None = None,
    response_language: str | None = None,
    initial: bool = False,
) -> dict[str, Any]:
    """Build the ``session`` object of a ``session.update``.

    ``initial`` adds the fields that only need sending once per connection. The service
    deep-merges updates, so a later update can carry only what changed.
    """
    session: dict[str, Any] = {"type": "realtime"}
    if initial:
        session["model"] = MODEL
        session["output_modalities"] = ["text"]
        td = turn_detection or TurnDetection()
        session["audio"] = {
            "input": {
                "format": {"type": "audio/pcm", "rate": SAMPLE_RATE},
                "turn_detection": {"type": "server_vad", "create_response": td.create_response},
            }
        }
        if max_output_tokens is not None:
            session["max_output_tokens"] = max_output_tokens
        if response_language is not None:
            session["poly_response_language"] = response_language
    if instructions is not None:
        session["instructions"] = instructions
    if tools is not None:
        session["tools"] = tools
    return session


def to_wire_tools(tools: list[llm.Tool]) -> list[dict[str, Any]]:
    """Convert LiveKit tools to Dialog-RSN-1 function tools.

    The service accepts function tools only, and rejects the whole update if it sees any other
    kind, so other tools are dropped here with a warning.
    """
    wire_tools: list[dict[str, Any]] = []
    for tool in tools:
        if isinstance(tool, llm.FunctionTool):
            schema = llm.utils.build_legacy_openai_schema(tool, internally_tagged=True)
        elif isinstance(tool, llm.RawFunctionTool):
            schema = dict(tool.info.raw_schema)
        else:
            logger.warning(
                "Dialog-RSN-1 supports function tools only, skipping this tool",
                extra={"tool": type(tool).__name__},
            )
            continue

        # Copy only the fields the service knows. An unknown key would fail the whole update.
        wire_tool: dict[str, Any] = {"type": "function", "name": schema["name"]}
        if schema.get("description"):
            wire_tool["description"] = schema["description"]
        wire_tool["parameters"] = schema.get("parameters") or {"type": "object", "properties": {}}
        wire_tools.append(wire_tool)
    return wire_tools


def to_wire_item(item: llm.ChatItem) -> dict[str, Any] | None:
    """Convert a LiveKit chat item to a ``conversation.item.create`` item.

    Returns None for items the service can't hold, such as a message with only images.
    """
    if item.type == "function_call":
        return {
            "id": item.id,
            "type": "function_call",
            "call_id": item.call_id,
            "name": item.name,
            "arguments": item.arguments,
        }

    if item.type == "function_call_output":
        return {
            "id": item.id,
            "type": "function_call_output",
            "call_id": item.call_id,
            "output": item.output,
        }

    if item.type != "message":
        return None

    texts: list[str] = []
    for content in item.content:
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, llm.AudioContent) and content.transcript:
            # the service only reads the transcript of an audio part
            texts.append(content.transcript)
        else:
            logger.warning(
                "Dialog-RSN-1 takes text content only, skipping a content part",
                extra={"content": type(content).__name__},
            )
    if not texts:
        return None

    role = "system" if item.role in ("system", "developer") else item.role
    part_type = "output_text" if role == "assistant" else "input_text"
    return {
        "id": item.id,
        "type": "message",
        "role": role,
        "content": [{"type": part_type, "text": text} for text in texts],
    }


def from_wire_item(item: dict[str, Any]) -> llm.ChatItem:
    """Convert a server item from ``conversation.item.added`` to a LiveKit chat item."""
    item_type = item.get("type")
    item_id = item["id"]

    if item_type == "function_call":
        return llm.FunctionCall(
            id=item_id,
            call_id=item["call_id"],
            name=item["name"],
            arguments=item.get("arguments") or "",
        )

    if item_type == "function_call_output":
        return llm.FunctionCallOutput(
            id=item_id,
            call_id=item["call_id"],
            output=item.get("output") or "",
            is_error=False,
        )

    if item_type == "message":
        texts: list[llm.ChatContent] = []
        for part in item.get("content") or []:
            text = part.get("text") or part.get("transcript")
            if text:
                texts.append(text)
        return llm.ChatMessage(id=item_id, role=item["role"], content=texts)

    raise ValueError(f"unsupported item type: {item_type}")


def is_fatal_error(error: dict[str, Any] | None) -> bool:
    return error is not None and error.get("code") in FATAL_ERROR_CODES


def describe_error(error: dict[str, Any]) -> str:
    """One line for a server error, naming the rejected field when there is one."""
    parts = [f"[{error.get('code') or error.get('type') or 'error'}]"]
    if error.get("message"):
        parts.append(str(error["message"]))
    if error.get("param"):
        parts.append(f"(param: {error['param']})")
    return " ".join(parts)
