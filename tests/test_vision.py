from __future__ import annotations

from app.pipeline.ffmpeg_utils import Frame
from app.pipeline.vision import (
    _ANALYZE_TOOL,
    _FLAG_TOOL,
    CandidateWindow,
    MomentAnalysis,
    _batches,
    merge_windows,
)


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
