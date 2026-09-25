"""A scripted stand-in for Dialog-RSN-1, for tests that don't need the real service.

It records every client event and only sends what a test tells it to, so each test controls
exactly what the session sees and in what order.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any

from aiohttp import web

CALL_ID = "call_" + "c" * 32  # the service's call ids are 37 characters


class FakeDialogRSN1:
    def __init__(self) -> None:
        self.received: list[dict[str, Any]] = []
        self.headers: list[dict[str, str]] = []
        self.connections: list[web.WebSocketResponse] = []
        self.handshake_status: int | None = None
        # auto-replies: client event type -> function returning server events to send back
        self.auto: dict[str, Callable[[dict[str, Any]], list[dict[str, Any]]]] = {}
        self._runner: web.AppRunner | None = None
        self._event = asyncio.Event()
        self.url = ""

    async def start(self) -> None:
        app = web.Application()
        app.router.add_get("/v1/realtime", self._handle)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
        self.url = f"http://127.0.0.1:{port}/v1"

    async def stop(self) -> None:
        for ws in self.connections:
            await ws.close()
        if self._runner:
            await self._runner.cleanup()

    async def _handle(self, request: web.Request) -> web.StreamResponse:
        self.headers.append(dict(request.headers))
        if self.handshake_status is not None:
            return web.Response(status=self.handshake_status)
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self.connections.append(ws)
        await ws.send_json({"type": "session.created", "session": {"type": "realtime"}})
        async for msg in ws:
            event = json.loads(msg.data)
            self.received.append(event)
            self._event.set()
            if reply := self.auto.get(event["type"]):
                for out in reply(event):
                    await ws.send_json(out)
        return ws

    @property
    def ws(self) -> web.WebSocketResponse:
        return self.connections[-1]

    async def send(self, *events: dict[str, Any]) -> None:
        for event in events:
            await self.ws.send_json(event)

    def of_type(self, event_type: str) -> list[dict[str, Any]]:
        return [e for e in self.received if e["type"] == event_type]

    async def wait_for(
        self, event_type: str, count: int = 1, timeout: float = 3.0
    ) -> list[dict[str, Any]]:
        async def _wait() -> list[dict[str, Any]]:
            while len(self.of_type(event_type)) < count:
                self._event.clear()
                await self._event.wait()
            return self.of_type(event_type)

        return await asyncio.wait_for(_wait(), timeout)

    async def wait_connections(self, count: int, timeout: float = 5.0) -> None:
        async def _wait() -> None:
            while len(self.connections) < count:
                await asyncio.sleep(0.01)

        await asyncio.wait_for(_wait(), timeout)


def item_added(item: dict[str, Any], previous_item_id: str | None = None) -> dict[str, Any]:
    return {"type": "conversation.item.added", "previous_item_id": previous_item_id, "item": item}


def echo_items(event: dict[str, Any]) -> list[dict[str, Any]]:
    """What the service sends back for conversation.item.create and .delete."""
    if event["type"] == "conversation.item.create":
        item = event["item"]
        # the service echoes previous_item_id verbatim and appends the item regardless
        return [
            item_added(item, event.get("previous_item_id")),
            {"type": "conversation.item.done", "item": item},
        ]
    return [{"type": "conversation.item.deleted", "item_id": event["item_id"]}]


def response_events(
    n: int,
    *,
    text: str | None = None,
    call: dict[str, Any] | None = None,
    status: str = "completed",
) -> list[dict[str, Any]]:
    """A complete response as the service streams it: created, items, deltas, done."""
    rid = f"resp_{n}"
    events: list[dict[str, Any]] = [
        {"type": "response.created", "response": {"id": rid, "status": "in_progress"}}
    ]
    output: list[dict[str, Any]] = []
    if text is not None:
        item_id = f"item_msg_{n}"
        item = {"id": item_id, "type": "message", "role": "assistant", "content": []}
        events += [
            {"type": "response.output_item.added", "response_id": rid, "item": item},
            item_added(item),
        ]
        for word in text.split(" "):
            events.append(
                {"type": "response.output_text.delta", "item_id": item_id, "delta": word + " "}
            )
        done_item = {**item, "content": [{"type": "output_text", "text": text}]}
        events += [
            {"type": "response.output_item.done", "item": done_item},
            {"type": "conversation.item.done", "item": done_item},
        ]
        output.append(done_item)
    if call is not None:
        item = {"id": f"item_call_{n}", "type": "function_call", **call}
        events += [
            {"type": "response.output_item.added", "item": {**item, "arguments": ""}},
            item_added({**item, "arguments": ""}),
            {"type": "response.output_item.done", "item": item},
            {"type": "conversation.item.done", "item": item},
        ]
        output.append(item)
    events.append(
        {
            "type": "response.done",
            "response": {
                "id": rid,
                "status": status,
                "output": output,
                "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            },
        }
    )
    return events
