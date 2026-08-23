"""English narrator voice-over generation via the ElevenLabs Text-to-Speech
API."""

from __future__ import annotations

import asyncio
from pathlib import Path

from elevenlabs.client import ElevenLabs
from elevenlabs.types.voice_settings import VoiceSettings

from app.config import get_settings


def _client() -> ElevenLabs:
    return ElevenLabs(api_key=get_settings().elevenlabs_api_key)


def _synthesize_sync(text: str, out_path: Path, voice_id: str, model_id: str) -> Path:
    client = _client()
    audio_chunks = client.text_to_speech.convert(
        voice_id=voice_id,
        text=text,
        model_id=model_id,
        output_format="mp3_44100_128",
        voice_settings=VoiceSettings(
            stability=0.45,
            similarity_boost=0.8,
            style=0.35,
            use_speaker_boost=True,
        ),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        for chunk in audio_chunks:
            if chunk:
                f.write(chunk)
    return out_path


async def synthesize_narration(text: str, out_path: Path) -> Path:
    settings = get_settings()
    text = text.strip()
    if not text:
        raise ValueError("narration text is empty")
    return await asyncio.to_thread(
        _synthesize_sync, text, out_path, settings.elevenlabs_voice_id, settings.elevenlabs_tts_model
    )
