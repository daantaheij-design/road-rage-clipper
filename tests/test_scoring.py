from __future__ import annotations

import pytest

from app.pipeline.scoring import (
    HARD_MAX_CLIP_SECONDS,
    HARD_MIN_CLIP_SECONDS,
    MIN_SELECTABLE_SCORE,
    select_clips,
)
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


def test_short_clips_are_not_stretched_to_an_artificial_minimum():
    # An 11-second incident must stay ~11 seconds, not get padded out to a
    # longer "preferred" duration - that was the exact behavior being fixed.
    a = _analysis(100, 111, score=70)  # 11s
    selected = select_clips([a], number_of_clips=1, video_duration=1000)
    assert len(selected) == 1
    duration = selected[0].end_seconds - selected[0].start_seconds
    assert duration == pytest.approx(11.0)


def test_short_clips_below_hard_floor_are_padded_up_to_it():
    a = _analysis(100, 103, score=70)  # 3s - below HARD_MIN_CLIP_SECONDS
    selected = select_clips([a], number_of_clips=1, video_duration=1000)
    assert len(selected) == 1
    duration = selected[0].end_seconds - selected[0].start_seconds
    assert duration >= HARD_MIN_CLIP_SECONDS - 0.1


def test_caps_overly_long_clips_at_hard_max_duration():
    a = _analysis(0, 500, score=70)
    selected = select_clips([a], number_of_clips=1, video_duration=1000)
    duration = selected[0].end_seconds - selected[0].start_seconds
    assert duration <= HARD_MAX_CLIP_SECONDS


def test_drops_candidates_that_cannot_reach_hard_minimum_duration():
    # The whole source video is shorter than the hard minimum clip length,
    # so no amount of padding can produce a usable clip.
    a = _analysis(1, 2, score=99)
    selected = select_clips([a], number_of_clips=1, video_duration=3)
    assert selected == []


def test_empty_input_returns_empty():
    assert select_clips([], number_of_clips=3, video_duration=100) == []


def test_narration_cues_and_crop_keyframes_shift_with_head_extension():
    from app.pipeline.vision import CropKeyframeDraft, NarrationCueDraft

    a = _analysis(100, 103, score=70)  # 3s, needs padding up to HARD_MIN_CLIP_SECONDS
    a.narration_cues = [
        NarrationCueDraft(beat="hook", text="Hook line", start_seconds=0.0, skip=False),
        NarrationCueDraft(beat="payoff", text="Payoff line", start_seconds=2.0, skip=False),
    ]
    a.crop_keyframes = [CropKeyframeDraft(time_seconds=0.0, focus_x=0.5, focus_y=0.5, confidence=0.9)]
    original_start = a.start_seconds

    selected = select_clips([a], number_of_clips=1, video_duration=1000)
    assert len(selected) == 1
    clip = selected[0]
    head_shift = original_start - clip.start_seconds
    assert head_shift > 0  # this candidate should have needed a head extension

    assert clip.narration_cues[0].start_seconds == pytest.approx(0.0 + head_shift)
    assert clip.narration_cues[1].start_seconds == pytest.approx(2.0 + head_shift)
    assert clip.crop_keyframes[0].time_seconds == pytest.approx(0.0 + head_shift)


def test_quality_floor_excludes_weak_candidates_even_under_requested_count():
    # Prefer quality over hitting the requested count: three requested, but
    # only one candidate clears the quality floor - should return just that one.
    analyses = [
        _analysis(0, 30, score=85, title="excellent"),
        _analysis(100, 130, score=35, title="boring"),
        _analysis(200, 230, score=20, title="very boring"),
    ]
    selected = select_clips(analyses, number_of_clips=3, video_duration=1000)
    assert [a.title for a in selected] == ["excellent"]


def test_quality_floor_can_return_zero_clips():
    analyses = [_analysis(0, 30, score=10), _analysis(100, 130, score=25)]
    selected = select_clips(analyses, number_of_clips=3, video_duration=1000)
    assert selected == []


def test_score_exactly_at_floor_is_selectable():
    a = _analysis(0, 30, score=MIN_SELECTABLE_SCORE)
    selected = select_clips([a], number_of_clips=1, video_duration=1000)
    assert len(selected) == 1
