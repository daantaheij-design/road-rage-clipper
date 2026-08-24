from __future__ import annotations

import pytest

from app.pipeline import vision as vision_mod
from app.pipeline.ffmpeg_utils import Frame
from app.pipeline.transcribe import Transcript
from app.pipeline.vision import (
    _ANALYZE_TOOL,
    _FLAG_TOOL,
    CandidateWindow,
    MomentAnalysis,
    _batches,
    _safe_float,
    analyze_candidate,
    merge_windows,
)

_BASE_SCORES = {dim: 5 for dim in vision_mod.SCORE_DIMENSIONS}


def _base_tool_result(**overrides) -> dict:
    result = {
        "is_moment": True,
        "title": "Test",
        "explanation": "e",
        "incident_boundaries": {
            "pre_context_start": 4.0,
            "incident_start": 10.0,
            "incident_peak": 12.0,
            "escalation_start": 12.0,
            "payoff_end": 30.0,
            "incident_end": 33.0,
            "is_continuous_single_event": False,
        },
        "start_seconds": 10.0,
        "end_seconds": 15.0,
        "scores": _BASE_SCORES,
        "hook_text": "h",
        "narration_cues": [],
        "crop_keyframes": [],
        "effects": [],
        "teaser": {"enabled": False, "source_start": 0.0, "source_end": 0.0},
    }
    result.update(overrides)
    return result


@pytest.fixture
def dense_frames(tmp_path):
    img_path = tmp_path / "frame.jpg"
    img_path.write_bytes(b"\xff\xd8\xff\xe0fake jpeg bytes")
    return [Frame(timestamp=5.0, path=img_path), Frame(timestamp=6.0, path=img_path)]


def test_merge_windows_combines_overlapping_and_close():
    candidates = [
        CandidateWindow(10, 15, 60, ["a"]),
        CandidateWindow(14, 20, 70, ["b"]),  # overlaps first
        CandidateWindow(22, 25, 50, ["c"]),  # within gap_seconds of previous
        CandidateWindow(100, 110, 90, ["d"]),  # far away, separate
    ]
    merged = merge_windows(candidates, gap_seconds=4.0)
    assert len(merged) == 2
    assert merged[0].start_seconds == 10
    assert merged[0].end_seconds == 25
    assert merged[0].suspicion == 70
    assert merged[1].start_seconds == 100


def test_merge_windows_empty():
    assert merge_windows([]) == []


def test_batches_no_frames():
    assert _batches([], size=5, overlap=1) == []


def test_batches_creates_overlapping_chunks():
    frames = [Frame(timestamp=float(i), path=None) for i in range(10)]
    batches = _batches(frames, size=4, overlap=1)
    assert all(len(b) <= 4 for b in batches)
    # every frame should show up in at least one batch
    covered = {f.timestamp for b in batches for f in b}
    assert covered == {f.timestamp for f in frames}


def test_moment_analysis_total_score():
    scores = {
        "visual_action": 8,
        "escalation": 7,
        "surprise": 6,
        "tension": 9,
        "hook_potential": 8,
        "understandable_context": 7,
        "payoff": 8,
        "retention": 9,
        "reaction": 6,
        "uniqueness": 5,
    }
    analysis = MomentAnalysis(
        is_moment=True,
        title="t",
        explanation="e",
        start_seconds=0,
        end_seconds=30,
        scores=scores,
        hook_text="h",
        narration_cues=[],
    )
    assert analysis.total_score == sum(scores.values())


def test_tool_schemas_are_well_formed():
    for tool in (_FLAG_TOOL, _ANALYZE_TOOL):
        assert tool["input_schema"]["type"] == "object"
        assert "properties" in tool["input_schema"]
        assert isinstance(tool["input_schema"]["required"], list)


def test_analyze_tool_requires_incident_boundaries():
    assert "incident_boundaries" in _ANALYZE_TOOL["input_schema"]["required"]
    boundary_props = _ANALYZE_TOOL["input_schema"]["properties"]["incident_boundaries"]["properties"]
    for field_name in (
        "pre_context_start",
        "incident_start",
        "incident_peak",
        "escalation_start",
        "payoff_end",
        "incident_end",
        "is_continuous_single_event",
    ):
        assert field_name in boundary_props


def test_safe_float_handles_none_and_invalid():
    assert _safe_float(None) is None
    assert _safe_float("not a number") is None
    assert _safe_float("3.5") == 3.5
    assert _safe_float(3) == 3.0


async def test_analyze_candidate_widens_end_when_continuous_event_cut_short(monkeypatch, dense_frames):
    """The regression this whole feature exists for: Claude's own
    incident_boundaries say the collision and the following confrontation
    are one continuous event through payoff_end=30s, but start_seconds/
    end_seconds only cover the 10-15s collision peak - the safety net must
    widen end_seconds to the stated payoff_end rather than trust the
    narrower fields."""

    async def fake_call_tool(client, *, system, content, tool, model):
        return _base_tool_result(
            incident_boundaries={
                "pre_context_start": 4.0,
                "incident_start": 10.0,
                "incident_peak": 12.0,
                "escalation_start": 14.0,
                "payoff_end": 30.0,
                "incident_end": 33.0,
                "is_continuous_single_event": True,
            },
            start_seconds=10.0,
            end_seconds=15.0,  # too narrow - only the collision, not the confrontation
        )

    monkeypatch.setattr(vision_mod, "_call_tool", fake_call_tool)
    candidate = CandidateWindow(start_seconds=10.0, end_seconds=15.0, suspicion=90, reasons=["collision"])
    result = await analyze_candidate(candidate, dense_frames, Transcript(text="", words=[]), video_duration=45.0)

    assert result is not None
    assert result.is_continuous_single_event is True
    assert result.end_seconds == pytest.approx(30.0)
    assert result.start_seconds == pytest.approx(4.0)


async def test_analyze_candidate_does_not_widen_when_not_continuous(monkeypatch, dense_frames):
    async def fake_call_tool(client, *, system, content, tool, model):
        return _base_tool_result(
            incident_boundaries={
                "pre_context_start": 4.0,
                "incident_start": 10.0,
                "incident_peak": 12.0,
                "escalation_start": 12.0,
                "payoff_end": 30.0,
                "incident_end": 33.0,
                "is_continuous_single_event": False,
            },
            start_seconds=10.0,
            end_seconds=15.0,
        )

    monkeypatch.setattr(vision_mod, "_call_tool", fake_call_tool)
    candidate = CandidateWindow(start_seconds=10.0, end_seconds=15.0, suspicion=90, reasons=["moment"])
    result = await analyze_candidate(candidate, dense_frames, Transcript(text="", words=[]), video_duration=45.0)

    assert result is not None
    assert result.end_seconds == pytest.approx(15.0)  # not widened - not flagged as continuous


async def test_analyze_candidate_does_not_narrow_an_already_wide_end(monkeypatch, dense_frames):
    """The safety net only ever widens - if Claude already chose a wider
    end_seconds than payoff_end for good reason, don't shrink it back."""

    async def fake_call_tool(client, *, system, content, tool, model):
        return _base_tool_result(
            incident_boundaries={
                "pre_context_start": 4.0,
                "incident_start": 10.0,
                "incident_peak": 12.0,
                "escalation_start": 14.0,
                "payoff_end": 20.0,
                "incident_end": 22.0,
                "is_continuous_single_event": True,
            },
            start_seconds=4.0,
            end_seconds=35.0,  # already wider than payoff_end
        )

    monkeypatch.setattr(vision_mod, "_call_tool", fake_call_tool)
    candidate = CandidateWindow(start_seconds=4.0, end_seconds=35.0, suspicion=90, reasons=["moment"])
    result = await analyze_candidate(candidate, dense_frames, Transcript(text="", words=[]), video_duration=45.0)

    assert result is not None
    assert result.end_seconds == pytest.approx(35.0)


async def test_45_second_incident_is_not_collapsed_to_a_five_second_peak(monkeypatch, dense_frames):
    """Regression for the exact reported production failure: a 45s source
    with a buildup (~5s), a collision (~12s), a confrontation (14s-35s),
    and a payoff (~38s) must produce a clip that preserves the coherent
    incident - not a clip that collapses down to ~5 seconds around just
    the collision. Exercises analyze_candidate's incident_boundaries
    safety net AND scoring.select_clips together, since either one
    silently re-narrowing the window would reproduce the bug."""
    from app.pipeline.scoring import select_clips

    async def fake_call_tool(client, *, system, content, tool, model):
        return _base_tool_result(
            title="Collision and confrontation",
            incident_boundaries={
                "pre_context_start": 2.0,
                "incident_start": 5.0,
                "incident_peak": 12.0,
                "escalation_start": 14.0,
                "payoff_end": 38.0,
                "incident_end": 40.0,
                "is_continuous_single_event": True,
            },
            # Simulates the exact reported bug: the model's own start/end
            # fields only cover the collision peak, contradicting its own
            # stated incident_boundaries.
            start_seconds=10.5,
            end_seconds=15.5,
            scores={dim: 8 for dim in vision_mod.SCORE_DIMENSIONS},
        )

    monkeypatch.setattr(vision_mod, "_call_tool", fake_call_tool)
    candidate = CandidateWindow(start_seconds=10.5, end_seconds=15.5, suspicion=95, reasons=["collision"])
    analysis = await analyze_candidate(candidate, dense_frames, Transcript(text="", words=[]), video_duration=45.0)

    assert analysis is not None
    # The safety net must have widened this before it ever reaches scoring.
    assert (analysis.end_seconds - analysis.start_seconds) > 20.0

    selected = select_clips([analysis], number_of_clips=1, video_duration=45.0)
    assert len(selected) == 1
    final_duration = selected[0].end_seconds - selected[0].start_seconds

    # The exact failure mode: must NOT collapse to ~5s around just the peak.
    assert final_duration > 15.0, f"expected the full incident to be preserved, got a {final_duration:.1f}s clip"
    # And it should reasonably cover the buildup-through-payoff span.
    assert selected[0].start_seconds <= 5.0
    assert selected[0].end_seconds >= 35.0
