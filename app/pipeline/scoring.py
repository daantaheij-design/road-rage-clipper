"""Turn Claude's per-candidate analyses into a final, non-overlapping list of
clips to render."""

from __future__ import annotations

from app.pipeline.vision import MomentAnalysis

MIN_CLIP_SECONDS = 25
MAX_CLIP_SECONDS = 90
PREFERRED_MAX_SECONDS = 60


def _clamp_duration(analysis: MomentAnalysis, video_duration: float) -> tuple[float, float]:
    start, end = analysis.start_seconds, analysis.end_seconds
    duration = end - start

    if duration < MIN_CLIP_SECONDS:
        deficit = MIN_CLIP_SECONDS - duration
        # Prefer extending the tail (let the payoff/reaction breathe) but pull
        # from the head too if we're near the end of the source video.
        end = min(video_duration, end + deficit * 0.6)
        start = max(0.0, start - (deficit - (end - analysis.end_seconds)))
    elif duration > MAX_CLIP_SECONDS:
        end = start + MAX_CLIP_SECONDS

    start = max(0.0, start)
    end = min(video_duration, end)
    return start, end


def select_clips(
    analyses: list[MomentAnalysis],
    *,
    number_of_clips: int,
    video_duration: float,
    min_gap_seconds: float = 3.0,
) -> list[MomentAnalysis]:
    """Rank by total score and greedily pick the best non-overlapping set."""
    ranked = sorted(analyses, key=lambda a: a.total_score, reverse=True)

    selected: list[MomentAnalysis] = []
    for a in ranked:
        if len(selected) >= number_of_clips:
            break

        start, end = _clamp_duration(a, video_duration)
        if end - start < MIN_CLIP_SECONDS * 0.8:
            continue  # too close to the edges of the source video to salvage

        # Narration cue timing is relative to the clip start Claude saw when
        # it proposed this window - if clamping pulled the head earlier to
        # hit the minimum duration, shift cues by the same amount so they
        # stay aligned with the actual rendered clip start.
        head_shift = a.start_seconds - start
        if head_shift:
            for cue in a.narration_cues:
                cue.start_seconds += head_shift

        a.start_seconds, a.end_seconds = start, end

        overlaps = any(
            not (end + min_gap_seconds <= s.start_seconds or start - min_gap_seconds >= s.end_seconds)
            for s in selected
        )
        if overlaps:
            continue

        selected.append(a)

    selected.sort(key=lambda a: a.start_seconds)
    return selected
