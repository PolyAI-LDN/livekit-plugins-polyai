# livekit-plugins-polyai

Use PolyAI's Dialog-RSN-1 as the model in a [LiveKit Agents](https://docs.livekit.io/agents/) voice agent.

Dialog-RSN-1 takes the caller's audio, decides when they've finished speaking, and streams the reply as text. In an `AgentSession` it replaces three parts of a standard pipeline: the speech-to-text plugin, the turn detector and the LLM. You add any LiveKit TTS plugin to speak the replies.

## Install

The plugin isn't on PyPI yet. Install it from GitHub:

```bash
pip install "livekit-plugins-polyai @ git+https://github.com/PolyAI-LDN/livekit-plugins-polyai"
```

With uv:

```bash
uv add "livekit-plugins-polyai @ git+https://github.com/PolyAI-LDN/livekit-plugins-polyai"
```

You need Python 3.10 or later, `livekit-agents` 1.8.3 or later, and a Dialog-RSN-1 workspace API key from Agent Studio. Set the key as `DIALOG_API_KEY`.

## Quick start

```python
from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli
from livekit.plugins import elevenlabs, polyai

server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    session = AgentSession(
        llm=polyai.realtime.RealtimeModel(),
        tts=elevenlabs.TTS(),
    )
    await session.start(
        agent=Agent(instructions="You are a helpful voice assistant. Greet the caller first."),
        room=ctx.room,
    )
    await session.generate_reply()


if __name__ == "__main__":
    cli.run_app(server)
```

The session has no `stt` and no `turn_handling`. Dialog-RSN-1 hears the room's audio directly and judges each pause in the context of the conversation. LiveKit still shows the caller's transcript, because Dialog-RSN-1 sends it with each turn.

[`examples/basic_agent.py`](examples/basic_agent.py) is a runnable version with a tool. For a full walkthrough, see the companion repo [PolyAI-LDN/dialog-rsn-1-livekit](https://github.com/PolyAI-LDN/dialog-rsn-1-livekit): a hotel front desk with two tools, barge-in and a scripted test caller.

## Options

```python
polyai.realtime.RealtimeModel(
    api_key=None,  # defaults to DIALOG_API_KEY
    base_url=NOT_GIVEN,  # defaults to DIALOG_REALTIME_URL, then wss://api.studio.poly.ai/v1/realtime
    turn_detection=NOT_GIVEN,  # polyai.realtime.TurnDetection(create_response=True)
    max_output_tokens=NOT_GIVEN,  # cap per reply, reasoning included
    response_language=NOT_GIVEN,  # e.g. "es-ES"
)
```

| Option | What it does |
|---|---|
| `turn_detection` | `TurnDetection(create_response=False)` makes Dialog-RSN-1 commit the caller's turn without replying, so you call `generate_reply()` yourself. Turn detection itself can't be switched off. |
| `max_output_tokens` | Caps each reply. The cap includes the model's reasoning tokens, so a low value can leave a reply empty or cut short. |
| `response_language` | Asks Dialog-RSN-1 to reply in this language, for example `"es-ES"`. Pass `None` to `update_options` to clear it. |

Change options on a running agent with `model.update_options(...)`.

To make the agent more or less patient before it replies, say so in the instructions. The model reads them when it decides whether the caller has finished.

## How it maps to LiveKit

| LiveKit feature | With Dialog-RSN-1 |
|---|---|
| Tools (`@function_tool`) | Supported. Function tools only; other tool types are skipped with a warning. |
| Barge-in | Supported. Dialog-RSN-1 stops generating when the caller talks over it, and the plugin rewrites the interrupted reply to the words the caller heard. |
| User transcripts | Supported, sent with each turn. |
| `update_instructions`, `update_tools`, chat context edits | Supported mid-call. |
| Audio output | Not used. The plugin asks Dialog-RSN-1 for text, so add a TTS plugin. |
| `generate_reply(instructions=...)` | Not sent, with a warning. Dialog-RSN-1 honors per-response instructions, but the plugin doesn't pass them yet. Put the behavior in the agent's instructions. |
| `tool_choice` | `"auto"` only. |
| LiveKit turn detector, VAD, endpointing options | Not used. Dialog-RSN-1 decides when the turn ends. |
| Images and video | Dropped, with a warning. |

## Debugging

Set `LK_POLYAI_DEBUG=1` to log every event the plugin sends and receives, audio excepted.

Each `RealtimeSession` also emits the raw events as dicts, as `polyai_server_event_received` and `polyai_client_event_queued`. The tests use them, and you can too if you drive a session from `model.session()` yourself.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `Dialog-RSN-1 needs an API key` | `DIALOG_API_KEY` isn't set | Set it, or pass `api_key=` |
| `Dialog-RSN-1 refused the connection (401)` | The key is wrong or revoked | Create a new workspace API key in Agent Studio |
| `Dialog-RSN-1 returned an error: [unknown_parameter] ... (param: session.x)` | A `session.update` carried a field the service doesn't accept, so none of it applied | Open an issue with the `param` value: the plugin should never send one |
| The agent never speaks | No TTS in the `AgentSession` | Add a TTS plugin |
| The agent keeps interrupting itself | The microphone hears the speakers | Use headphones, or echo cancellation on the client |

## Known limitations

- **Playback reports.** Dialog-RSN-1 accepts `poly.output_audio.*` events that tell it what the caller has heard. With them, it checks whether the caller's speech during playback is a real interruption, or just "mm-hm", before it stops. The plugin doesn't send them yet, so Dialog-RSN-1 estimates playback instead. It assumes a typical speaking rate of 15 characters a second from the reply's first words, makes the same check for as long as it thinks the reply is playing, and on a barge-in cuts the reply in history back to the words the caller likely heard. Reports would replace that guess with real positions.
- **Reconnection.** The service can't resume a session. If the connection drops, the plugin opens a new one and replays the instructions, tools and conversation. The service closes a connection after 300 seconds with no client events, which happens only when no audio is flowing.

## Contributing

Bug reports and pull requests are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) to set up, run the tests, and test against the real service.

## License

[Apache 2.0](LICENSE)
