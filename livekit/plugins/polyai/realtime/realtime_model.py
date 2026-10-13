"""Dialog-RSN-1 as a LiveKit Agents realtime model.

Dialog-RSN-1 hears the caller's audio, decides when their turn is over, and streams the reply
as text. In an ``AgentSession`` it takes the place of the STT, the turn detector and the LLM.
A TTS plugin speaks the reply.

The service speaks the OpenAI Realtime protocol, with three differences this session is built
around:

- The session schema is narrow, and an update with any field outside it is rejected whole.
  The session only ever sends fields the service accepts (see ``protocol.session_config``).
- ``response.created`` carries no metadata, so each ``generate_reply`` is matched to the
  response that answers it by order. The service answers every ``response.create`` as soon as
  it arrives, with ``response.created`` or an error naming it.
- The service appends every item it's given to the end of the conversation, whatever
  ``previous_item_id`` says, so the local mirror of the conversation does the same.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import copy
import json
import os
import time
import weakref
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Literal

import aiohttp
from livekit.agents import APIConnectionError, APIError, APIStatusError, llm, utils
from livekit.agents.metrics import RealtimeModelMetrics
from livekit.agents.metrics.base import Metadata
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGiven,
    NotGivenOr,
)
from livekit.agents.utils import is_given
from livekit.agents.voice.generation import remove_instructions

from livekit import rtc

from ..log import logger
from ..version import __version__
from . import protocol
from .protocol import NUM_CHANNELS, SAMPLE_RATE, TurnDetection

_debug = bool(int(os.getenv("LK_POLYAI_DEBUG", "0") or 0))

GENERATE_REPLY_TIMEOUT = 10.0
UPDATE_CHAT_CTX_TIMEOUT = 5.0


@dataclass
class _RealtimeOptions:
    api_key: str
    base_url: str
    turn_detection: TurnDetection
    max_output_tokens: int | Literal["inf"] | None
    response_language: str | None
    conn_options: APIConnectOptions


@dataclass
class _MessageGeneration:
    message_id: str
    text_ch: utils.aio.Chan[str]
    audio_ch: utils.aio.Chan[rtc.AudioFrame]
    modalities: asyncio.Future[list[Literal["text", "audio"]]]
    text: str = ""


@dataclass
class _ResponseGeneration:
    response_id: str
    message_ch: utils.aio.Chan[llm.MessageGeneration]
    function_ch: utils.aio.Chan[llm.FunctionCall]
    created_at: float
    messages: dict[str, _MessageGeneration] = field(default_factory=dict)
    first_token_at: float | None = None

    def close(self) -> None:
        for msg in self.messages.values():
            msg.text_ch.close()
            msg.audio_ch.close()
            if not msg.modalities.done():
                msg.modalities.set_result(["text"])
        self.function_ch.close()
        self.message_ch.close()


class _DiscardedGeneration:
    """A response nobody is waiting for any more. Its events are skipped until it's done."""

    def __init__(self, response_id: str) -> None:
        self.response_id = response_id


class RealtimeModel(llm.RealtimeModel):
    """Dialog-RSN-1: speech in, streamed text out.

    Pair it with a TTS plugin in an ``AgentSession``, and leave out ``stt`` and turn
    handling. Dialog-RSN-1 does both.

    Args:
        api_key: A Dialog-RSN-1 workspace API key. Defaults to ``DIALOGUE_API_KEY``.
        base_url: The API base URL. Defaults to ``DIALOGUE_BASE_URL``, then
            ``https://api.studio.poly.ai/v1``.
        turn_detection: Turn detection options. By default the model replies as soon as it
            judges the caller's turn over.
        max_output_tokens: Cap on tokens per reply, reasoning included. A low cap can cut
            replies short. Defaults to the service's own limit.
        response_language: Language to reply in, for example ``"es-ES"``.
        http_session: An aiohttp session to reuse. Defaults to LiveKit's shared session.
        conn_options: Connection timeout and retry policy.

    Example:
        ```python
        from livekit.agents import AgentSession
        from livekit.plugins import elevenlabs, polyai

        session = AgentSession(
            llm=polyai.realtime.RealtimeModel(),
            tts=elevenlabs.TTS(),
        )
        ```
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        turn_detection: NotGivenOr[TurnDetection] = NOT_GIVEN,
        max_output_tokens: NotGivenOr[int | Literal["inf"]] = NOT_GIVEN,
        response_language: NotGivenOr[str] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        super().__init__(
            capabilities=llm.RealtimeCapabilities(
                # an interrupted reply is rewritten to what the caller heard
                message_truncation=True,
                turn_detection=True,
                user_transcription=True,
                auto_tool_reply_generation=False,
                audio_output=False,
                manual_function_calls=True,
                mutable_chat_context=True,
                mutable_instructions=True,
                mutable_tools=True,
                # response.create options are ignored by the service
                per_response_tool_choice=False,
            )
        )

        api_key = api_key or os.environ.get("DIALOGUE_API_KEY")
        if not api_key:
            raise ValueError(
                "Dialog-RSN-1 needs an API key. Pass api_key, or set DIALOGUE_API_KEY to a "
                "workspace API key from Agent Studio."
            )

        self._opts = _RealtimeOptions(
            api_key=api_key,
            base_url=base_url
            if is_given(base_url)
            else os.environ.get("DIALOGUE_BASE_URL") or protocol.DEFAULT_BASE_URL,
            turn_detection=turn_detection if is_given(turn_detection) else TurnDetection(),
            max_output_tokens=None
            if isinstance(max_output_tokens, NotGiven)
            else max_output_tokens,
            response_language=response_language if is_given(response_language) else None,
            conn_options=conn_options,
        )
        self._capabilities.turn_detection = self._opts.turn_detection.create_response
        self._http_session = http_session
        self._http_session_owned = False
        self._sessions = weakref.WeakSet[RealtimeSession]()

    @property
    def model(self) -> str:
        return protocol.MODEL

    @property
    def provider(self) -> str:
        return "polyai"

    def update_options(
        self,
        *,
        turn_detection: NotGivenOr[TurnDetection] = NOT_GIVEN,
        max_output_tokens: NotGivenOr[int | Literal["inf"]] = NOT_GIVEN,
        response_language: NotGivenOr[str | None] = NOT_GIVEN,
    ) -> None:
        """Change options on this model and every open session."""
        if is_given(turn_detection):
            self._opts.turn_detection = turn_detection
            self._capabilities.turn_detection = turn_detection.create_response
        if not isinstance(max_output_tokens, NotGiven):
            self._opts.max_output_tokens = max_output_tokens
        if is_given(response_language):
            self._opts.response_language = response_language

        for sess in self._sessions:
            sess._update_model_options(
                turn_detection=turn_detection,
                max_output_tokens=max_output_tokens,
                response_language=response_language,
            )

    def session(self, *, turn_detection_disabled: bool = False) -> RealtimeSession:
        # Turn detection is Dialog-RSN-1's own and can't be switched off, so
        # can_disable_turn_detection is False and turn_detection_disabled is never set.
        sess = RealtimeSession(self)
        self._sessions.add(sess)
        return sess

    def _ensure_http_session(self) -> aiohttp.ClientSession:
        if not self._http_session:
            try:
                self._http_session = utils.http_context.http_session()
            except RuntimeError:
                self._http_session = aiohttp.ClientSession()
                self._http_session_owned = True
        return self._http_session

    async def aclose(self) -> None:
        if self._http_session_owned and self._http_session:
            await self._http_session.close()


class RealtimeSession(
    llm.RealtimeSession[Literal["polyai_server_event_received", "polyai_client_event_queued"]]
):
    """One conversation with Dialog-RSN-1 over a WebSocket.

    Besides LiveKit's realtime events, it emits two for debugging:

    - ``polyai_server_event_received``: each raw event from the service, as a dict
    - ``polyai_client_event_queued``: each raw event about to be sent, as a dict
    """

    def __init__(self, realtime_model: RealtimeModel) -> None:
        super().__init__(realtime_model)
        self._realtime_model: RealtimeModel = realtime_model
        self._opts = copy.copy(realtime_model._opts)

        self._instructions: str | None = None
        self._tools = llm.ToolContext.empty()
        self._remote_chat_ctx = llm.remote_chat_context.RemoteChatContext()

        self._msg_ch = utils.aio.Chan[dict[str, Any]]()
        self._closing = False

        # response.create event_ids sent and not yet answered, oldest first
        self._pending_creates: deque[str] = deque()
        self._response_futures: dict[str, asyncio.Future[llm.GenerationCreatedEvent]] = {}
        # generate_reply calls abandoned before their response arrived
        self._discarded_event_ids: set[str] = set()
        self._current_generation: _ResponseGeneration | _DiscardedGeneration | None = None

        # update_chat_ctx waits on these; the service answers with item.added / item.deleted
        self._item_create_futures: dict[str, asyncio.Future[None]] = {}
        self._item_delete_futures: dict[str, asyncio.Future[None]] = {}
        # the same futures by event_id, so an error naming the event settles the right one
        self._chat_ctx_event_futures: dict[str, asyncio.Future[None]] = {}
        self._update_chat_ctx_lock = asyncio.Lock()
        self._update_tools_lock = asyncio.Lock()

        self._speech_started_at: dict[str, float] = {}
        self._transcript_deltas: dict[str, str] = {}

        self._input_resampler: rtc.AudioResampler | None = None
        self._input_resampler_rate = 0
        # 100 ms chunks
        self._bstream = utils.audio.AudioByteStream(
            SAMPLE_RATE, NUM_CHANNELS, samples_per_channel=SAMPLE_RATE // 10
        )
        self._pushed_duration = 0.0
        self._warned: set[str] = set()

        self._main_atask = asyncio.create_task(self._main_task(), name="polyai.RealtimeSession")
        self.send_event(self._session_update_event(initial=True))

    # region LiveKit RealtimeSession interface

    @property
    def chat_ctx(self) -> llm.ChatContext:
        return self._remote_chat_ctx.to_chat_ctx()

    @property
    def tools(self) -> llm.ToolContext:
        return self._tools.copy()

    async def update_instructions(self, instructions: str) -> None:
        self._instructions = instructions
        self.send_event(self._session_update_event(instructions=instructions))

    async def update_tools(self, tools: list[llm.Tool]) -> None:
        async with self._update_tools_lock:
            wire_tools = protocol.to_wire_tools(tools)
            self.send_event(self._session_update_event(tools=wire_tools))
            sent = {t["name"] for t in wire_tools}
            self._tools = llm.ToolContext(
                [
                    t
                    for t in tools
                    if isinstance(t, (llm.FunctionTool, llm.RawFunctionTool))
                    and t.info.name in sent
                ]
            )

    def update_options(self, *, tool_choice: NotGivenOr[llm.ToolChoice | None] = NOT_GIVEN) -> None:
        # The service only accepts tool_choice "auto", its default. LiveKit also asks for
        # "none" on its own after some tool calls; that one is safe to ignore.
        if is_given(tool_choice) and tool_choice not in (None, "auto", "none"):
            self._warn_once(
                "tool_choice",
                f"Dialog-RSN-1 supports tool_choice='auto' only, ignoring {tool_choice!r}",
            )

    async def update_chat_ctx(self, chat_ctx: llm.ChatContext) -> None:
        async with self._update_chat_ctx_lock:
            chat_ctx = chat_ctx.copy(exclude_handoff=True, exclude_config_update=True)
            # the agent's instructions go in session.update, not in the conversation
            remove_instructions(chat_ctx)

            events = self._chat_ctx_update_events(chat_ctx)
            futs: list[asyncio.Future[None]] = []
            for ev in events:
                fut: asyncio.Future[None] = asyncio.Future()
                if ev["type"] == "conversation.item.delete":
                    self._item_delete_futures[ev["item_id"]] = fut
                else:
                    self._item_create_futures[ev["item"]["id"]] = fut
                self._chat_ctx_event_futures[ev["event_id"]] = fut
                futs.append(fut)
                self.send_event(ev)

            if not futs:
                return
            try:
                results = await asyncio.wait_for(
                    asyncio.gather(*futs, return_exceptions=True), UPDATE_CHAT_CTX_TIMEOUT
                )
            except asyncio.TimeoutError:
                raise llm.RealtimeError("update_chat_ctx timed out") from None
            finally:
                for ev in events:
                    self._chat_ctx_event_futures.pop(ev["event_id"], None)
                    if ev["type"] == "conversation.item.delete":
                        self._item_delete_futures.pop(ev["item_id"], None)
                    else:
                        self._item_create_futures.pop(ev["item"]["id"], None)

            if rejected := [str(r) for r in results if isinstance(r, BaseException)]:
                # one rejected item isn't worth failing the turn over
                logger.warning(
                    "Dialog-RSN-1 rejected part of a chat context update",
                    extra={"errors": rejected},
                )

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        for f in self._resample(frame):
            for chunk in self._bstream.write(f.data.tobytes()):
                self.send_event(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(chunk.data).decode("ascii"),
                    }
                )
                self._pushed_duration += chunk.duration

    def push_video(self, frame: rtc.VideoFrame) -> None:
        self._warn_once("video", "Dialog-RSN-1 takes audio only, dropping video frames")

    def generate_reply(
        self,
        *,
        instructions: NotGivenOr[str] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice] = NOT_GIVEN,
        tools: NotGivenOr[list[llm.Tool]] = NOT_GIVEN,
    ) -> asyncio.Future[llm.GenerationCreatedEvent]:
        if is_given(instructions) and instructions:
            self._warn_once(
                "reply_instructions",
                "Dialog-RSN-1 ignores per-reply instructions. Put them in the agent's "
                "instructions instead.",
            )

        event_id = utils.shortuuid("response_create_")
        fut: asyncio.Future[llm.GenerationCreatedEvent] = asyncio.Future()
        self._response_futures[event_id] = fut
        self.send_event({"type": "response.create", "event_id": event_id})

        def _on_timeout() -> None:
            if fut.done():
                return
            # The service answers every response.create at once, so this one never will be.
            # Left queued, it would claim the next response.
            with contextlib.suppress(ValueError):
                self._pending_creates.remove(event_id)
            fut.set_exception(llm.RealtimeError("generate_reply timed out"))

        handle = asyncio.get_event_loop().call_later(GENERATE_REPLY_TIMEOUT, _on_timeout)

        def _on_done(f: asyncio.Future[llm.GenerationCreatedEvent]) -> None:
            handle.cancel()
            self._response_futures.pop(event_id, None)
            if f.cancelled():
                # The caller gave up, say on an interruption. If the response is already
                # open, cancel it now; if not, drop it when it arrives.
                self.send_event({"type": "response.cancel"})
                if event_id in self._pending_creates:
                    self._discarded_event_ids.add(event_id)

        fut.add_done_callback(_on_done)
        return fut

    def commit_audio(self) -> None:
        if self._pushed_duration > 0.1:
            self.send_event({"type": "input_audio_buffer.commit"})
            self._pushed_duration = 0.0

    def clear_audio(self) -> None:
        # Note: the service also cancels any reply it's generating for the cleared audio.
        self.send_event({"type": "input_audio_buffer.clear"})
        self._pushed_duration = 0.0

    def interrupt(self) -> None:
        if self._current_generation is not None or self._response_futures:
            self.send_event({"type": "response.cancel"})

    def truncate(
        self,
        *,
        message_id: str,
        modalities: list[Literal["text", "audio"]],
        audio_end_ms: int,
        audio_transcript: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        # Rewrite the interrupted reply to the words the caller heard, so the model's next
        # reply follows on from those. conversation.item.truncate is a no-op on this service.
        # This can run before response.done fills in the mirrored text, so it doesn't go
        # through the diff, which skips items whose remote text is still empty.
        if not is_given(audio_transcript) or self._remote_chat_ctx.get(message_id) is None:
            return
        self.send_event(
            {
                "type": "conversation.item.delete",
                "event_id": utils.shortuuid("truncate_delete_"),
                "item_id": message_id,
            }
        )
        if audio_transcript:
            heard = llm.ChatMessage(id=message_id, role="assistant", content=[audio_transcript])
            self.send_event(
                {
                    "type": "conversation.item.create",
                    "event_id": utils.shortuuid("truncate_create_"),
                    "item": protocol.to_wire_item(heard),
                }
            )

    async def aclose(self) -> None:
        self._closing = True
        self._close_current_generation()
        self._msg_ch.close()
        # a connection failure was already reported through the "error" event
        await asyncio.gather(self._main_atask, return_exceptions=True)

    # endregion

    def send_event(self, event: dict[str, Any]) -> None:
        """Queue a raw client event. Useful for ``poly_*`` extensions the plugin doesn't wrap."""
        with contextlib.suppress(utils.aio.channel.ChanClosed):
            self._msg_ch.send_nowait(event)

    def _update_model_options(
        self,
        *,
        turn_detection: NotGivenOr[TurnDetection],
        max_output_tokens: NotGivenOr[int | Literal["inf"]],
        response_language: NotGivenOr[str | None],
    ) -> None:
        session: dict[str, Any] = {"type": "realtime"}
        if is_given(turn_detection):
            self._opts.turn_detection = turn_detection
            session["audio"] = {
                "input": {
                    "turn_detection": {
                        "type": "server_vad",
                        "create_response": turn_detection.create_response,
                    }
                }
            }
        if not isinstance(max_output_tokens, NotGiven):
            self._opts.max_output_tokens = max_output_tokens
            session["max_output_tokens"] = max_output_tokens
        if is_given(response_language):
            self._opts.response_language = response_language
            session["poly_response_language"] = response_language
        if len(session) > 1:
            self.send_event(
                {
                    "type": "session.update",
                    "event_id": utils.shortuuid("options_update_"),
                    "session": session,
                }
            )

    def _session_update_event(
        self,
        *,
        initial: bool = False,
        instructions: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        opts = self._opts
        session = protocol.session_config(
            instructions=instructions,
            tools=tools,
            turn_detection=opts.turn_detection,
            max_output_tokens=opts.max_output_tokens,
            response_language=opts.response_language,
            initial=initial,
        )
        return {
            "type": "session.update",
            "event_id": utils.shortuuid("session_update_"),
            "session": session,
        }

    def _chat_ctx_update_events(self, chat_ctx: llm.ChatContext) -> list[dict[str, Any]]:
        """Events that make the service's conversation match ``chat_ctx``.

        The service appends each created item to the end of the conversation. That matches
        LiveKit's own updates, which add tool results and messages at the tail.
        """
        remote_ctx = self._remote_chat_ctx.to_chat_ctx()
        remote_ids = {item.id for item in remote_ctx.items}
        # An empty local message is a placeholder. An empty remote one is a caller turn whose
        # transcript hasn't arrived yet. Neither should be sent or deleted.
        chat_ctx = llm.ChatContext(
            [
                item
                for item in chat_ctx.items
                if item.type != "message" or item.content or item.id in remote_ids
            ]
        )
        diff = llm.utils.compute_chat_ctx_diff(remote_ctx, chat_ctx)

        events: list[dict[str, Any]] = []

        def _delete(item_id: str) -> None:
            events.append(
                {
                    "type": "conversation.item.delete",
                    "event_id": utils.shortuuid("chat_ctx_delete_"),
                    "item_id": item_id,
                }
            )

        def _create(item_id: str) -> None:
            item = chat_ctx.get_by_id(item_id)
            wire_item = protocol.to_wire_item(item) if item is not None else None
            if wire_item is None:
                return
            events.append(
                {
                    "type": "conversation.item.create",
                    "event_id": utils.shortuuid("chat_ctx_create_"),
                    "item": wire_item,
                }
            )

        for item_id in diff.to_remove:
            _delete(item_id)
        for _, item_id in diff.to_create:
            _create(item_id)
        for _, item_id in diff.to_update:
            remote = remote_ctx.get_by_id(item_id)
            if remote is not None and remote.type == "message" and not remote.content:
                # the remote copy just hasn't been filled in yet
                continue
            _delete(item_id)
            _create(item_id)
        return events

    def _resample(self, frame: rtc.AudioFrame) -> Iterator[rtc.AudioFrame]:
        if self._input_resampler and frame.sample_rate != self._input_resampler_rate:
            self._input_resampler = None
        if self._input_resampler is None and (
            frame.sample_rate != SAMPLE_RATE or frame.num_channels != NUM_CHANNELS
        ):
            self._input_resampler_rate = frame.sample_rate
            self._input_resampler = rtc.AudioResampler(
                input_rate=frame.sample_rate,
                output_rate=SAMPLE_RATE,
                num_channels=NUM_CHANNELS,
            )
        if self._input_resampler:
            yield from self._input_resampler.push(frame)
        else:
            yield frame

    def _warn_once(self, key: str, message: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            logger.warning(message)

    # region connection

    @utils.log_exceptions(logger=logger)
    async def _main_task(self) -> None:
        max_retries = self._opts.conn_options.max_retry
        num_retries = 0
        reconnecting = False
        try:
            while not self._msg_ch.closed:
                try:
                    ws = await self._connect()
                    if reconnecting:
                        await self._resume(ws)
                        num_retries = 0
                    await self._run_ws(ws)
                except APIError as e:
                    if not e.retryable or num_retries >= max_retries:
                        self._emit_error(e, recoverable=False)
                        raise
                    self._emit_error(e, recoverable=True)
                    conn = self._opts.conn_options
                    delay = 0.1 if num_retries == 0 else conn.retry_interval
                    logger.warning(
                        f"Dialog-RSN-1 connection failed, retrying in {delay}s",
                        exc_info=e,
                        extra={"attempt": num_retries},
                    )
                    await asyncio.sleep(delay)
                    num_retries += 1
                except Exception as e:
                    self._emit_error(e, recoverable=False)
                    raise
                reconnecting = True
        finally:
            self._close_current_generation()
            self._fail_pending_replies("realtime session closed")

    async def _connect(self) -> aiohttp.ClientWebSocketResponse:
        url = protocol.realtime_url(self._opts.base_url)
        headers = {
            "X-API-KEY": self._opts.api_key,
            "User-Agent": f"livekit-plugins-polyai/{__version__}",
        }
        t0 = time.perf_counter()
        try:
            ws = await asyncio.wait_for(
                self._realtime_model._ensure_http_session().ws_connect(url, headers=headers),
                self._opts.conn_options.timeout,
            )
        except aiohttp.WSServerHandshakeError as e:
            raise APIStatusError(
                f"Dialog-RSN-1 refused the connection ({e.status})"
                + (": check DIALOGUE_API_KEY" if e.status in (401, 403) else ""),
                status_code=e.status,
                retryable=e.status >= 500 or e.status == 429,
            ) from e
        except aiohttp.ClientError as e:
            raise APIConnectionError("Dialog-RSN-1 connection error") from e
        except asyncio.TimeoutError as e:
            raise APIConnectionError("Dialog-RSN-1 connection timed out") from e
        self._report_connection_acquired(time.perf_counter() - t0)
        return ws

    async def _resume(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        """Rebuild the conversation on a new connection. The service can't resume a session."""
        events = [
            self._session_update_event(
                initial=True,
                instructions=self._instructions,
                tools=protocol.to_wire_tools(self._tools.flatten()) or None,
            )
        ]
        chat_ctx = self.chat_ctx.copy(
            exclude_instructions=True,
            exclude_empty_message=True,
            exclude_handoff=True,
            exclude_config_update=True,
        )
        old_ctx = self._remote_chat_ctx
        self._remote_chat_ctx = llm.remote_chat_context.RemoteChatContext()
        self._speech_started_at.clear()
        self._transcript_deltas.clear()
        events.extend(self._chat_ctx_update_events(chat_ctx))
        try:
            for ev in events:
                await self._send(ws, ev)
        except Exception as e:
            self._remote_chat_ctx = old_ctx
            raise APIConnectionError("Dialog-RSN-1 failed to resume the session") from e

        self._fail_pending_replies("pending reply discarded on reconnection")
        self._close_current_generation()
        self.emit("session_reconnected", llm.RealtimeSessionReconnectedEvent())

    async def _send(self, ws: aiohttp.ClientWebSocketResponse, event: dict[str, Any]) -> None:
        if event["type"] == "response.create":
            # queued before the send, so a fast reply can't arrive ahead of its request
            self._pending_creates.append(event["event_id"])
        self.emit("polyai_client_event_queued", event)
        if _debug and event["type"] != "input_audio_buffer.append":
            logger.debug(">>> %s", event["type"], extra={"lk.pii.event": event})
        await ws.send_str(json.dumps(event))

    async def _run_ws(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        closing = False

        @utils.log_exceptions(logger=logger)
        async def _send_task() -> None:
            nonlocal closing
            async for event in self._msg_ch:
                try:
                    await self._send(ws, event)
                except Exception:
                    logger.exception("failed to send event to Dialog-RSN-1")
            closing = True
            await ws.close()

        @utils.log_exceptions(logger=logger)
        async def _recv_task() -> None:
            while True:
                msg = await ws.receive()
                if msg.type in (
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    if closing:
                        return
                    code = ws.close_code
                    reason = msg.extra if isinstance(msg.extra, str) else ""
                    if code in protocol.FATAL_CLOSE_CODES:
                        raise APIStatusError(
                            f"Dialog-RSN-1 closed the session ({code}): {reason}",
                            status_code=code or -1,
                            retryable=False,
                        )
                    raise APIConnectionError(
                        f"Dialog-RSN-1 closed the connection unexpectedly ({code}) {reason}"
                    )
                if msg.type != aiohttp.WSMsgType.TEXT or self._closing:
                    continue

                event = json.loads(msg.data)
                self.emit("polyai_server_event_received", event)
                if _debug and not event.get("type", "").endswith(".delta"):
                    logger.debug("<<< %s", event.get("type"), extra={"lk.pii.event": event})
                try:
                    self._handle_event(event)
                except APIError as e:
                    if not e.retryable:
                        raise
                    logger.exception("failed to handle a Dialog-RSN-1 event")
                except Exception:
                    logger.exception(
                        "failed to handle a Dialog-RSN-1 event", extra={"type": event.get("type")}
                    )

        tasks = [
            asyncio.create_task(_recv_task(), name="polyai._recv_task"),
            asyncio.create_task(_send_task(), name="polyai._send_task"),
        ]
        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
        finally:
            await utils.aio.cancel_and_wait(*tasks)
            await ws.close()

    # endregion

    # region server events

    def _handle_event(self, event: dict[str, Any]) -> None:
        handler = _HANDLERS.get(event.get("type", ""))
        if handler is not None:
            handler(self, event)

    def _on_speech_started(self, event: dict[str, Any]) -> None:
        if item_id := event.get("item_id"):
            self._speech_started_at[item_id] = time.time()
        self.emit("input_speech_started", llm.InputSpeechStartedEvent())

    def _on_speech_stopped(self, event: dict[str, Any]) -> None:
        self.emit(
            "input_speech_stopped",
            llm.InputSpeechStoppedEvent(user_transcription_enabled=True),
        )

    def _on_item_added(self, event: dict[str, Any]) -> None:
        item = event["item"]
        item_id = item["id"]
        # The service appends every item, whatever previous_item_id it echoes back.
        previous_item_id = self._remote_chat_ctx.tail_id
        try:
            lk_item = protocol.from_wire_item(item)
            self._remote_chat_ctx.insert(previous_item_id, lk_item)
            self.emit(
                "remote_item_added",
                llm.RemoteItemAddedEvent(previous_item_id=previous_item_id, item=lk_item),
            )
        except ValueError as e:
            logger.warning(f"failed to mirror item {item_id}: {e}")

        if (fut := self._item_create_futures.pop(item_id, None)) and not fut.done():
            fut.set_result(None)

    def _on_item_done(self, event: dict[str, Any]) -> None:
        # item.added carries an item as it starts. Fill in what it has by the time it's done.
        item = event["item"]
        remote = self._remote_chat_ctx.get(item["id"])
        if remote is None:
            return
        try:
            done_item = protocol.from_wire_item(item)
        except ValueError:
            return
        if isinstance(remote.item, llm.FunctionCall) and isinstance(done_item, llm.FunctionCall):
            remote.item.arguments = done_item.arguments
        elif (
            isinstance(remote.item, llm.ChatMessage)
            and isinstance(done_item, llm.ChatMessage)
            and done_item.content
        ):
            remote.item.content = done_item.content

    def _on_item_deleted(self, event: dict[str, Any]) -> None:
        item_id = event["item_id"]
        self._speech_started_at.pop(item_id, None)
        self._transcript_deltas.pop(item_id, None)
        with contextlib.suppress(ValueError):
            self._remote_chat_ctx.delete(item_id)
        if (fut := self._item_delete_futures.pop(item_id, None)) and not fut.done():
            fut.set_result(None)

    def _on_transcription_delta(self, event: dict[str, Any]) -> None:
        if not (delta := event.get("delta")):
            return
        item_id = event["item_id"]
        text = self._transcript_deltas.get(item_id, "") + delta
        self._transcript_deltas[item_id] = text
        self.emit(
            "input_audio_transcription_completed",
            llm.InputTranscriptionCompleted(item_id=item_id, transcript=text, is_final=False),
        )

    def _on_transcription_completed(self, event: dict[str, Any]) -> None:
        item_id = event["item_id"]
        transcript = event.get("transcript") or ""
        self._transcript_deltas.pop(item_id, None)
        if (remote := self._remote_chat_ctx.get(item_id)) and isinstance(
            remote.item, llm.ChatMessage
        ):
            remote.item.content = [transcript] if transcript else []
        self.emit(
            "input_audio_transcription_completed",
            llm.InputTranscriptionCompleted(
                item_id=item_id,
                transcript=transcript,
                is_final=True,
                # the transcript arrives after the reply, so place it by when the turn began
                turn_started_at=self._speech_started_at.pop(item_id, None),
            ),
        )

    def _on_response_created(self, event: dict[str, Any]) -> None:
        response_id = event["response"]["id"]
        event_id = self._pending_creates.popleft() if self._pending_creates else None

        # The oldest unanswered response.create is the one this answers (see module docs).
        # One race remains: if the service starts a reply to the caller's turn just before a
        # response.create reaches it, that reply is matched here and the create is refused
        # with conversation_already_has_active_response. The reply still plays.
        if event_id is not None and event_id in self._discarded_event_ids:
            self._discarded_event_ids.discard(event_id)
            self.send_event({"type": "response.cancel"})
            self._current_generation = _DiscardedGeneration(response_id)
            return

        generation = _ResponseGeneration(
            response_id=response_id,
            message_ch=utils.aio.Chan(),
            function_ch=utils.aio.Chan(),
            created_at=time.time(),
        )
        self._close_current_generation()
        self._current_generation = generation
        created_ev = llm.GenerationCreatedEvent(
            message_stream=generation.message_ch,
            function_stream=generation.function_ch,
            user_initiated=False,
            response_id=response_id,
        )
        reply_fut = self._response_futures.get(event_id) if event_id is not None else None
        if reply_fut is not None and not reply_fut.done():
            created_ev.user_initiated = True
            reply_fut.set_result(created_ev)
        self.emit("generation_created", created_ev)

    def _generation(self) -> _ResponseGeneration | None:
        gen = self._current_generation
        return gen if isinstance(gen, _ResponseGeneration) else None

    def _on_output_item_added(self, event: dict[str, Any]) -> None:
        if (gen := self._generation()) is None:
            return
        item = event["item"]
        if item.get("type") != "message":
            return
        modalities: asyncio.Future[list[Literal["text", "audio"]]] = asyncio.Future()
        modalities.set_result(["text"])
        msg = _MessageGeneration(
            message_id=item["id"],
            text_ch=utils.aio.Chan(),
            audio_ch=utils.aio.Chan(),
            modalities=modalities,
        )
        msg.audio_ch.close()
        gen.messages[msg.message_id] = msg
        gen.message_ch.send_nowait(
            llm.MessageGeneration(
                message_id=msg.message_id,
                text_stream=msg.text_ch,
                audio_stream=msg.audio_ch,
                modalities=msg.modalities,
            )
        )

    def _on_text_delta(self, event: dict[str, Any]) -> None:
        if (gen := self._generation()) is None:
            return
        msg = gen.messages.get(event["item_id"])
        if msg is None:
            return
        if gen.first_token_at is None:
            gen.first_token_at = time.time()
        msg.text += event["delta"]
        msg.text_ch.send_nowait(event["delta"])

    def _on_output_item_done(self, event: dict[str, Any]) -> None:
        if (gen := self._generation()) is None:
            return
        item = event["item"]
        if item.get("type") == "function_call":
            gen.function_ch.send_nowait(
                llm.FunctionCall(
                    id=item["id"],
                    call_id=item["call_id"],
                    name=item["name"],
                    arguments=item.get("arguments") or "",
                )
            )
        elif msg := gen.messages.get(item["id"]):
            msg.text_ch.close()

    def _on_response_done(self, event: dict[str, Any]) -> None:
        gen = self._current_generation
        response = event["response"]
        if gen is None or gen.response_id != response.get("id"):
            return
        self._current_generation = None
        if isinstance(gen, _DiscardedGeneration):
            return

        for item_id, msg in gen.messages.items():
            remote = self._remote_chat_ctx.get(item_id)
            if remote and isinstance(remote.item, llm.ChatMessage) and not remote.item.content:
                remote.item.content = [msg.text] if msg.text else []
        gen.close()

        status = response.get("status")
        if status == "failed":
            logger.warning(
                "Dialog-RSN-1 reply failed",
                extra={"status_details": response.get("status_details")},
            )
        self._emit_metrics(gen, response)

    def _on_error(self, event: dict[str, Any]) -> None:
        error = event.get("error") or {}
        code = error.get("code")
        event_id = error.get("event_id")
        message = protocol.describe_error(error)

        if event_id:
            if item_fut := self._chat_ctx_event_futures.pop(event_id, None):
                if not item_fut.done():
                    item_fut.set_exception(llm.RealtimeError(message, code=code))
                return
            if event_id in self._pending_creates:
                self._pending_creates.remove(event_id)
                self._discarded_event_ids.discard(event_id)
                reply_fut = self._response_futures.get(event_id)
                if reply_fut is not None and not reply_fut.done():
                    reply_fut.set_exception(llm.RealtimeError(message, code=code))
                return
            if code == "conversation_already_has_active_response":
                # the create was already matched to the reply that beat it (see above)
                return

        if code == "input_audio_buffer_commit_empty":
            # the service committed the turn itself, before our commit arrived
            return

        logger.error(f"Dialog-RSN-1 returned an error: {message}", extra={"error": error})
        fatal = protocol.is_fatal_error(error)
        err = APIError(
            f"Dialog-RSN-1 returned an error: {message}", body=error, retryable=not fatal
        )
        if fatal:
            raise err
        self._emit_error(err, recoverable=True)

    # endregion

    def _emit_metrics(self, gen: _ResponseGeneration, response: dict[str, Any]) -> None:
        usage = response.get("usage") or {}
        input_details = usage.get("input_token_details") or {}
        cached_details = input_details.get("cached_tokens_details") or {}
        duration = time.time() - gen.created_at
        output_tokens = usage.get("output_tokens", 0)
        self.emit(
            "metrics_collected",
            RealtimeModelMetrics(
                timestamp=gen.created_at,
                request_id=gen.response_id,
                ttft=gen.first_token_at - gen.created_at if gen.first_token_at else -1,
                duration=duration,
                cancelled=response.get("status") == "cancelled",
                label=self._realtime_model.label,
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=output_tokens,
                total_tokens=usage.get("total_tokens", 0),
                tokens_per_second=output_tokens / duration if duration > 0 else 0,
                input_token_details=RealtimeModelMetrics.InputTokenDetails(
                    audio_tokens=input_details.get("audio_tokens", 0),
                    text_tokens=input_details.get("text_tokens", 0),
                    cached_tokens=input_details.get("cached_tokens", 0),
                    cached_tokens_details=RealtimeModelMetrics.CachedTokenDetails(
                        text_tokens=cached_details.get("text_tokens", 0),
                        audio_tokens=cached_details.get("audio_tokens", 0),
                    ),
                ),
                output_token_details=RealtimeModelMetrics.OutputTokenDetails(
                    text_tokens=output_tokens,
                ),
                metadata=Metadata(
                    model_name=self._realtime_model.model,
                    model_provider=self._realtime_model.provider,
                ),
            ),
        )

    def _close_current_generation(self) -> None:
        gen = self._current_generation
        self._current_generation = None
        if isinstance(gen, _ResponseGeneration):
            gen.close()

    def _fail_pending_replies(self, reason: str) -> None:
        for fut in self._response_futures.values():
            if not fut.done():
                fut.set_exception(llm.RealtimeError(reason))
        self._response_futures.clear()
        self._pending_creates.clear()
        self._discarded_event_ids.clear()

    def _emit_error(self, error: Exception, *, recoverable: bool) -> None:
        self.emit(
            "error",
            llm.RealtimeModelError(
                timestamp=time.time(),
                label=self._realtime_model.label,
                error=error,
                recoverable=recoverable,
            ),
        )


_HANDLERS = {
    "input_audio_buffer.speech_started": RealtimeSession._on_speech_started,
    "input_audio_buffer.speech_stopped": RealtimeSession._on_speech_stopped,
    "conversation.item.added": RealtimeSession._on_item_added,
    "conversation.item.done": RealtimeSession._on_item_done,
    "conversation.item.deleted": RealtimeSession._on_item_deleted,
    "conversation.item.input_audio_transcription.delta": RealtimeSession._on_transcription_delta,
    "conversation.item.input_audio_transcription.completed": (
        RealtimeSession._on_transcription_completed
    ),
    "response.created": RealtimeSession._on_response_created,
    "response.output_item.added": RealtimeSession._on_output_item_added,
    "response.output_text.delta": RealtimeSession._on_text_delta,
    "response.output_item.done": RealtimeSession._on_output_item_done,
    "response.done": RealtimeSession._on_response_done,
    "error": RealtimeSession._on_error,
}
