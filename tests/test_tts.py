from __future__ import annotations

import base64
from types import SimpleNamespace

import pytest

from app.pipeline import tts as tts_mod
from app.pipeline.tts import WordTiming, _characters_to_words, synthesize_narration


def test_characters_to_words_splits_on_whitespace():
    text = "I thought no way"
    chars = list(text)
    starts = [i * 0.1 for i in range(len(chars))]
    ends = [s + 0.1 for s in starts]
    words = _characters_to_words(chars, starts, ends)
    assert [w.text for w in words] == ["I", "thought", "no", "way"]


def test_characters_to_words_keeps_attached_punctuation():
    text = "No way, seriously."
    chars = list(text)
    starts = [i * 0.1 for i in range(len(chars))]
    ends = [s + 0.1 for s in starts]
    words = _characters_to_words(chars, starts, ends)
    assert [w.text for w in words] == ["No", "way,", "seriously."]


def test_characters_to_words_timing_spans_the_whole_word():
    chars = list("hi there")
    starts = [i * 0.05 for i in range(len(chars))]
    ends = [s + 0.05 for s in starts]
    words = _characters_to_words(chars, starts, ends)
    assert words[0].text == "hi"
    assert words[0].start == pytest.approx(starts[0])
    assert words[0].end == pytest.approx(ends[1])  # end of 'i'
    assert words[1].text == "there"
    assert words[1].start == pytest.approx(starts[3])  # start of 't'


def test_characters_to_words_handles_leading_trailing_whitespace():
    chars = list("  hi  ")
    starts = [i * 0.1 for i in range(len(chars))]
    ends = [s + 0.1 for s in starts]
    words = _characters_to_words(chars, starts, ends)
    assert [w.text for w in words] == ["hi"]


def test_characters_to_words_empty_input():
    assert _characters_to_words([], [], []) == []


async def test_synthesize_narration_rejects_empty_text(tmp_path):
    with pytest.raises(ValueError):
        await synthesize_narration("   ", tmp_path / "out.mp3")


async def test_synthesize_narration_writes_audio_and_returns_word_timings(settings, monkeypatch, tmp_path):
    fake_audio_bytes = b"fake mp3 audio bytes"

    def fake_convert_with_timestamps(*, voice_id, text, model_id, output_format, voice_settings):
        assert voice_id == settings.elevenlabs_voice_id
        assert model_id == settings.elevenlabs_tts_model
        assert voice_settings.stability == settings.elevenlabs_voice_stability
        assert voice_settings.similarity_boost == settings.elevenlabs_voice_similarity
        assert voice_settings.style == settings.elevenlabs_voice_style
        assert voice_settings.speed == settings.elevenlabs_voice_speed

        chars = list(text)
        starts = [i * 0.08 for i in range(len(chars))]
        ends = [s + 0.08 for s in starts]
        alignment = SimpleNamespace(
            characters=chars, character_start_times_seconds=starts, character_end_times_seconds=ends
        )
        return SimpleNamespace(
            audio_base_64=base64.b64encode(fake_audio_bytes).decode("ascii"),
            alignment=alignment,
            normalized_alignment=alignment,
        )

    fake_client = SimpleNamespace(
        text_to_speech=SimpleNamespace(convert_with_timestamps=fake_convert_with_timestamps)
    )
    monkeypatch.setattr(tts_mod, "_client", lambda: fake_client)

    out_path = tmp_path / "narration.mp3"
    result = await synthesize_narration("Watch this now", out_path)

    assert out_path.read_bytes() == fake_audio_bytes
    assert result.audio_path == out_path
    assert [w.text for w in result.words] == ["Watch", "this", "now"]
    assert all(isinstance(w, WordTiming) for w in result.words)


async def test_synthesize_narration_falls_back_to_normalized_alignment(settings, monkeypatch, tmp_path):
    """Regression: ElevenLabs applies text normalization server-side and
    can return alignment=None with only normalized_alignment populated -
    this must NOT be treated the same as "no alignment at all" (which
    degrades to a whole-sentence caption block)."""
    text = "Watch this now"

    def fake_convert_with_timestamps(**kwargs):
        chars = list(text)
        starts = [i * 0.08 for i in range(len(chars))]
        ends = [s + 0.08 for s in starts]
        normalized = SimpleNamespace(
            characters=chars, character_start_times_seconds=starts, character_end_times_seconds=ends
        )
        return SimpleNamespace(
            audio_base_64=base64.b64encode(b"x").decode("ascii"), alignment=None, normalized_alignment=normalized
        )

    fake_client = SimpleNamespace(
        text_to_speech=SimpleNamespace(convert_with_timestamps=fake_convert_with_timestamps)
    )
    monkeypatch.setattr(tts_mod, "_client", lambda: fake_client)

    result = await synthesize_narration(text, tmp_path / "out.mp3")
    assert [w.text for w in result.words] == ["Watch", "this", "now"]


async def test_synthesize_narration_estimates_words_when_no_alignment_at_all(settings, monkeypatch, tmp_path):
    """Neither alignment nor normalized_alignment present - must still
    produce a strictly per-word breakdown (spec: never silently fall back
    to a whole-sentence caption), not an empty word list."""

    def fake_convert_with_timestamps(**kwargs):
        return SimpleNamespace(
            audio_base_64=base64.b64encode(b"x").decode("ascii"), alignment=None, normalized_alignment=None
        )

    fake_client = SimpleNamespace(
        text_to_speech=SimpleNamespace(convert_with_timestamps=fake_convert_with_timestamps)
    )
    monkeypatch.setattr(tts_mod, "_client", lambda: fake_client)

    result = await synthesize_narration("This driver got way too close", tmp_path / "out.mp3")
    assert [w.text for w in result.words] == ["This", "driver", "got", "way", "too", "close"]
    # Strictly increasing, non-overlapping - genuinely one word at a time.
    for a, b in zip(result.words, result.words[1:], strict=False):
        assert a.end <= b.start


def test_estimate_word_timings_never_overlaps():
    from app.pipeline.tts import _estimate_word_timings

    timings = _estimate_word_timings("a short narration sentence about a close call", speed=1.0)
    for a, b in zip(timings, timings[1:], strict=False):
        assert a.end <= b.start


def test_estimate_word_timings_respects_speed():
    from app.pipeline.tts import _estimate_word_timings

    normal = _estimate_word_timings("watch this driver", speed=1.0)
    fast = _estimate_word_timings("watch this driver", speed=2.0)
    assert fast[-1].end < normal[-1].end
