"""English narrator voice-over generation via the ElevenLabs Text-to-Speech
API.

Uses `convert_with_timestamps` rather than plain `convert` - ElevenLabs
returns character-level alignment (start/end time for every character
spoken) alongside the audio, which is what makes word-by-word synced
captions possible without a separate, expensive forced-alignment pass. See
`_characters_to_words` for how that gets turned into word timings.
"""

from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass
from pathlib import Path

from elevenlabs.client import ElevenLabs
from elevenlabs.types.voice_settings import VoiceSettings

from app.config import get_settings


@dataclass
class WordTiming:
    text: str
    start: float  # seconds, relative to the start of this narration audio clip
    end: float


@dataclass
class NarrationAudio:
    audio_path: Path
    words: list[WordTiming]


def _client() -> ElevenLabs:
    return ElevenLabs(api_key=get_settings().elevenlabs_api_key)


def _characters_to_words(characters: list[str], starts: list[float], ends: list[float]) -> list[WordTiming]:
    """Group ElevenLabs' per-character alignment into per-word timing,
    splitting on whitespace characters (which ElevenLabs includes as their
    own entries in the alignment) and keeping attached punctuation."""
    words: list[WordTiming] = []
    current_chars: list[str] = []
    current_start: float | None = None
    current_end: float = 0.0

    for ch, start, end in zip(characters, starts, ends, strict=False):
        if ch.isspace():
            if current_chars:
                words.append(WordTiming(text="".join(current_chars), start=current_start, end=current_end))
                current_chars = []
                current_start = None
            continue
        if current_start is None:
            current_start = start
        current_chars.append(ch)
        current_end = end

    if current_chars:
        words.append(WordTiming(text="".join(current_chars), start=current_start, end=current_end))

    return words


def _synthesize_sync(text: str, out_path: Path, settings) -> NarrationAudio:
    client = _client()
    result = client.text_to_speech.convert_with_timestamps(
        voice_id=settings.elevenlabs_voice_id,
        text=text,
        model_id=settings.elevenlabs_tts_model,
        output_format="mp3_44100_128",
        voice_settings=VoiceSettings(
            stability=settings.elevenlabs_voice_stability,
            similarity_boost=settings.elevenlabs_voice_similarity,
            style=settings.elevenlabs_voice_style,
            speed=settings.elevenlabs_voice_speed,
            use_speaker_boost=True,
        ),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(base64.b64decode(result.audio_base_64))

    words: list[WordTiming] = []
    if result.alignment is not None:
        words = _characters_to_words(
            result.alignment.characters,
            result.alignment.character_start_times_seconds,
            result.alignment.character_end_times_seconds,
        )

    return NarrationAudio(audio_path=out_path, words=words)


async def synthesize_narration(text: str, out_path: Path) -> NarrationAudio:
    settings = get_settings()
    text = text.strip()
    if not text:
        raise ValueError("narration text is empty")
    return await asyncio.to_thread(_synthesize_sync, text, out_path, settings)
