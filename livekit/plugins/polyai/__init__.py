"""PolyAI plugin for LiveKit Agents.

Dialog-RSN-1 hears the caller, decides when they've finished, and streams the reply as text.
Pair ``polyai.realtime.RealtimeModel`` with any LiveKit TTS in an ``AgentSession``.
"""

from livekit.agents import Plugin

from . import realtime
from .log import logger
from .version import __version__

__all__ = ["__version__", "realtime"]


class PolyAIPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logger)


Plugin.register_plugin(PolyAIPlugin())
