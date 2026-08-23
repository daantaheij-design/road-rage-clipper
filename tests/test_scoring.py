from __future__ import annotations

import pytest

from app.pipeline.scoring import MAX_CLIP_SECONDS, MIN_CLIP_SECONDS, select_clips
from app.pipeline.vision import MomentAnalysis


def _analysis(start, end, score, **overrides) -> MomentAnalysis:
    scores = {
        "visual_action": score // 10,
        "escalation": score // 10,
        "surprise": score // 10,
        "tension": score // 10,
        "hook_potential": score // 10,
        "understandable_context": score // 10,
        "payoff": score // 10,
        "retention": score // 10,
        "reaction": score // 10,
        "uniqueness": score - 9 * (score // 10),
    }
    return MomentAnalysis(
        is_moment=True,
        title=overrides.get("title", "clip"),
        explanation="",
        start_seconds=start,
        end_seconds=end,
        scores=scores,
        hook_text="hook",
        narration_cues=[],
    )


def test_selects_highest_scoring_non_overlapping():
    analyses = [
        _analysis(10, 50, score=90, title="best"),
        _analysis(20, 60, score=95, title="overlaps best but slightly higher"),
        _analysis(200, 240, score=80, title="separate"),
    ]
    selected = select_clips(analyses, number_of_clips=3, video_duration=600)
    titles = [a.title for a in selected]
    # The two overlapping candidates can't both be selected.
    assert "overlaps best but slightly higher" in titles
    assert "best" not in titles
    assert "separate" in titles
    assert len(selected) == 2


def test_respects_number_of_clips_limit():
    analyses = [_analysis(i * 100, i * 100 + 40, score=50 + i) for i in range(5)]
    selected = select_clips(analyses, number_of_clips=2, video_duration=1000)
    assert len(selected) == 2


def test_extends_short_clips_to_minimum_duration():
    a = _analysis(100, 110, score=70)  # only 10s
    selected = select_clips([a], number_of_clips=1, video_duration=1000)
    assert len(selected) == 1
    duration = selected[0].end_seconds - selected[0].start_seconds
    assert duration >= MIN_CLIP_SECONDS - 1  # allow small rounding


def test_caps_overly_long_clips_at_max_duration():
    a = _analysis(0, 500, score=70)
    selected = select_clips([a], number_of_clips=1, video_duration=1000)
    duration = selected[0].end_seconds - selected[0].start_seconds
    assert duration <= MAX_CLIP_SECONDS


def test_drops_candidates_that_cannot_reach_minimum_duration():
    # The whole source video is shorter than the minimum clip length, so no
    # amount of padding can produce a usable clip.
    a = _analysis(13, 15, score=99)
    selected = select_clips([a], number_of_clips=1, video_duration=15)
    assert selected == []


def test_empty_input_returns_empty():
    assert select_clips([], number_of_clips=3, video_duration=100) == []


def test_narration_cues_shift_with_head_extension():
    from app.pipeline.vision import NarrationCueDraft

    a = _analysis(100, 110, score=70)  # 10s, needs extending to reach MIN_CLIP_SECONDS
    a.narration_cues = [
        NarrationCueDraft(beat="hook", text="Hook line", start_seconds=0.0, skip=False),
        NarrationCueDraft(beat="payoff", text="Payoff line", start_seconds=8.0, skip=False),
    ]
    original_start = a.start_seconds

    selected = select_clips([a], number_of_clips=1, video_duration=1000)
    assert len(selected) == 1
    clip = selected[0]
    head_shift = original_start - clip.start_seconds
    assert head_shift > 0  # this candidate should have needed a head extension

    assert clip.narration_cues[0].start_seconds == pytest.approx(0.0 + head_shift)
    assert clip.narration_cues[1].start_seconds == pytest.approx(8.0 + head_shift)
