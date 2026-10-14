"""A minimal voice agent on Dialog-RSN-1, with one tool.

    uv sync --group examples
    cp .env.example .env               # add POLY_API_KEY and ELEVEN_API_KEY
    uv run lk agent console examples/basic_agent.py

Or run it against a LiveKit server with `uv run examples/basic_agent.py dev`.
"""

from datetime import datetime

from dotenv import load_dotenv
from livekit.agents import Agent, AgentServer, AgentSession, JobContext, RunContext, cli
from livekit.agents.llm import function_tool

from livekit.plugins import elevenlabs, polyai

load_dotenv()


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a friendly voice assistant. Keep replies to one or two short "
                "sentences. Start the call by saying: Hi, how can I help?"
            )
        )

    @function_tool
    async def current_time(self, context: RunContext) -> str:
        """Get the current local time."""
        return datetime.now().strftime("%H:%M")


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    session = AgentSession(
        # Dialog-RSN-1 hears the caller, decides when they've finished, and streams the reply
        # as text. No STT, turn detector or separate LLM.
        llm=polyai.realtime.RealtimeModel(),
        # Dialog-RSN-1 replies in text, so a TTS speaks each reply as it streams.
        # Sarah works on the ElevenLabs free tier.
        tts=elevenlabs.TTS(voice_id="EXAVITQu4vr4xnSDxMaL", model="eleven_flash_v2_5"),
    )
    await session.start(agent=Assistant(), room=ctx.room)
    # the greeting comes from the instructions: Dialog-RSN-1 ignores per-reply instructions
    await session.generate_reply()


if __name__ == "__main__":
    cli.run_app(server)
