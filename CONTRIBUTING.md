# Contributing

Thanks for helping. Bug reports, fixes and new features are all welcome.

## Report a bug

Open an issue with the bug template. The most useful thing you can attach is a debug log: run your agent with `LK_POLYAI_DEBUG=1` and paste the events around the failure. Remove API keys and anything a caller said before you paste.

## Set up

You need [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/PolyAI-LDN/livekit-plugins-polyai
cd livekit-plugins-polyai
uv sync
```

## Check your change

CI runs these on every pull request:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
```

The tests don't need an API key. `tests/fake_server.py` stands in for Dialog-RSN-1: it records every event the session sends and replies only with what each test scripts. Add a test there for any change in behavior. If you're fixing a bug, write the test first and watch it fail.

CI also runs the tests on the oldest `livekit-agents` the package supports. To run that locally:

```bash
uv run --isolated --python 3.10 --resolution lowest-direct --group dev pytest
```

## Test against the real service

Unit tests can't catch a field the service rejects, so test protocol changes against Dialog-RSN-1 too:

```bash
cp .env.example .env              # add POLY_API_KEY and ELEVEN_API_KEY
uv sync --group examples
uv run lk agent console examples/basic_agent.py --text
```

Say in your pull request what you ran.

## How the code is laid out

| File | What it holds |
|---|---|
| `livekit/plugins/polyai/realtime/protocol.py` | The wire format: session config, tool and item conversion. Pure functions, no I/O. |
| `livekit/plugins/polyai/realtime/realtime_model.py` | `RealtimeModel` and `RealtimeSession`: the connection, reconnects, and mapping server events to LiveKit's. |
| `tests/fake_server.py` | The scripted fake service the tests run against. |

Two rules keep the plugin working against the service:

1. **Only send session fields the service accepts.** Dialog-RSN-1 rejects a whole `session.update` if any field is unknown, and the agent then runs without its instructions or tools. `protocol.session_config` is the one place session fields are built. `test_initial_session_sends_only_accepted_fields` guards it.
2. **Match replies by order, not metadata.** The service doesn't echo `response.metadata`. Each `generate_reply` is matched to the next `response.created`, so every path that drops a pending request (errors, timeouts, reconnects) has to remove it from `_pending_creates`.

## Pull requests

- Keep each pull request to one change.
- Add an entry to `CHANGELOG.md` under "Unreleased".
- Write commit messages in the imperative: "Send playback reports", not "Sent playback reports".

## Releasing

Maintainers only:

1. Move the "Unreleased" entries in `CHANGELOG.md` under the new version.
2. Bump `livekit/plugins/polyai/version.py`.
3. Merge, then tag the merge commit `vX.Y.Z` and push the tag. The release workflow builds the package and publishes it to PyPI.
