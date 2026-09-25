from __future__ import annotations

from collections.abc import AsyncIterator

import aiohttp
import pytest
from livekit.agents import APIConnectOptions

from livekit.plugins import polyai

from .fake_server import FakeDialogRSN1

FAST_RETRY = APIConnectOptions(max_retry=2, retry_interval=0.01, timeout=2.0)


@pytest.fixture
async def server() -> AsyncIterator[FakeDialogRSN1]:
    fake = FakeDialogRSN1()
    await fake.start()
    yield fake
    await fake.stop()


@pytest.fixture
async def model(server: FakeDialogRSN1) -> AsyncIterator[polyai.realtime.RealtimeModel]:
    async with aiohttp.ClientSession() as http:
        m = polyai.realtime.RealtimeModel(
            api_key="test-key", base_url=server.url, http_session=http, conn_options=FAST_RETRY
        )
        yield m
        await m.aclose()


@pytest.fixture
async def session(
    model: polyai.realtime.RealtimeModel, server: FakeDialogRSN1
) -> AsyncIterator[polyai.realtime.RealtimeSession]:
    sess = model.session()
    await server.wait_for("session.update")
    yield sess
    await sess.aclose()
