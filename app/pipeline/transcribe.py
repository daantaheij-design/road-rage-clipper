"""Speech-to-text transcription via the ElevenLabs Scribe API, with
word-level timestamps."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

from elevenlabs.client import ElevenLabs

from app.config import get_settings


@dataclass
class TranscriptWord:
    text: str
    start: float
    end: float
    kind: str  # word | spacing | audio_event


@dataclass
class Transcript:
    text: str
    words: list[TranscriptWord]

    def words_in_range(self, start: float, end: float) -> list[TranscriptWord]:
        return [w for w in self.words if w.kind == "word" and w.end > start and w.start < end]

    def text_in_range(self, start: float, end: float) -> str:
        return " ".join(w.text for w in self.words_in_range(start, end))


def _client() -> ElevenLabs:
    return ElevenLabs(api_key=get_settings().elevenlabs_api_key)


def _transcribe_sync(audio_path: Path, model_id: str) -> Transcript:
    client = _client()
    with open(audio_path, "rb") as f:
        result = client.speech_to_text.convert(
            model_id=model_id,
            file=f,
            timestamps_granularity="word",
            tag_audio_events=True,
        )

    words = [
        TranscriptWord(text=w.text, start=w.start or 0.0, end=w.end or 0.0, kind=w.type or "word")
        for w in (result.words or [])
    ]
    return Transcript(text=result.text or "", words=words)


async def transcribe_audio(audio_path: Path) -> Transcript:
    settings = get_settings()
    return await asyncio.to_thread(_transcribe_sync, audio_path, settings.elevenlabs_stt_model)
