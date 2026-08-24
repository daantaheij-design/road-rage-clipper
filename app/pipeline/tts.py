"""English narrator voice-over generation via the ElevenLabs Text-to-Speech
API.

Uses `convert_with_timestamps` rather than plain `convert` - ElevenLabs
returns character-level alignment (start/end time for every character
spoken) alongside the audio, which is what makes word-by-word synced
captions possible without a separate, expensive forced-alignment pass. See
`_characters_to_words` for how that gets turned into word timings.

`alignment` (timing against the original text) can legitimately come back
None even on a successful call: ElevenLabs applies text normalization
server-side (spelling out numbers, expanding abbreviations, etc. -
`apply_text_normalization` defaults to "auto" and we don't override it),
and when normalization changes the text, only `normalized_alignment`
(timing against the *normalized* text) is populated. Checking only
`alignment` silently produced an empty word list on every such cue in
production, which made app.pipeline.captions fall back to a single
whole-sentence caption for that cue - exactly the "not literally one word
at a time" bug this fixes. Below, `alignment` is preferred when present
(it matches Claude's original wording, which callers already have) and
`normalized_alignment` is the fallback; if genuinely neither is present,
`_estimate_word_timings` provides an approximate but still strictly
per-word breakdown rather than ever handing back an empty word list.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from dataclasses import dataclass
from pathlib import Path

from elevenlabs.client import ElevenLabs
from elevenlabs.types.character_alignment_response_model import CharacterAlignmentResponseModel
from elevenlabs.types.voice_settings import VoiceSettings

from app.config import get_settings

logger = logging.getLogger(__name__)

# Rough characters-per-second speaking pace at normal (1.0x) TTS speed, used
# only by _estimate_word_timings - the rare fallback when ElevenLabs returns
# neither alignment nor normalized_alignment. Not meant to be frame-exact,
# just good enough that captions still land roughly on their words instead
# of degrading to a whole-sentence block.
_ESTIMATED_CHARS_PER_SECOND = 15.0
_ESTIMATED_MIN_WORD_SECONDS = 0.15
_ESTIMATED_WORD_GAP_SECONDS = 0.05


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


def _estimate_word_timings(text: str, speed: float) -> list[WordTiming]:
    """Approximate per-word timing when ElevenLabs returned no alignment at
    all (neither original nor normalized) - proportional to word length at
    a rough speaking pace, adjusted for the configured voice speed. Never
    used when real alignment is available; exists so a caption is still
    strictly one-word-at-a-time even in that rare case, instead of one
    block of text for the whole cue."""
    chars_per_second = _ESTIMATED_CHARS_PER_SECOND * max(0.5, speed)
    timings: list[WordTiming] = []
    t = 0.0
    for word in text.split():
        duration = max(_ESTIMATED_MIN_WORD_SECONDS, len(word) / chars_per_second)
        timings.append(WordTiming(text=word, start=t, end=t + duration))
        t += duration + _ESTIMATED_WORD_GAP_SECONDS
    return timings


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

    alignment: CharacterAlignmentResponseModel | None = result.alignment or result.normalized_alignment
    if alignment is not None:
        words = _characters_to_words(
            alignment.characters, alignment.character_start_times_seconds, alignment.character_end_times_seconds
        )
    else:
        logger.warning(
            "ElevenLabs returned no character alignment (neither original nor normalized) for narration "
            "text %r - using estimated word timings so captions still render one word at a time",
            text,
        )
        words = []
    if not words:
        words = _estimate_word_timings(text, settings.elevenlabs_voice_speed)

    return NarrationAudio(audio_path=out_path, words=words)


async def synthesize_narration(text: str, out_path: Path) -> NarrationAudio:
    settings = get_settings()
    text = text.strip()
    if not text:
        raise ValueError("narration text is empty")
    return await asyncio.to_thread(_synthesize_sync, text, out_path, settings)
