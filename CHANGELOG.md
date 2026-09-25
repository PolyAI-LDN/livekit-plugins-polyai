# Changelog

All notable changes to this project are listed here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses [semantic versioning](https://semver.org/).

## Unreleased

## 0.1.0

First release.

- `polyai.realtime.RealtimeModel`: Dialog-RSN-1 as a LiveKit Agents realtime model, with text output for a TTS plugin.
- Tools, barge-in, user transcripts, and instruction, tool and chat context updates mid-call.
- Replies to `generate_reply` matched by order, as Dialog-RSN-1 doesn't echo response metadata.
- Automatic reconnection that replays the instructions, tools and conversation.
- `TurnDetection`, `max_output_tokens` and `response_language` options.
