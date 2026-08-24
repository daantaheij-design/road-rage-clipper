"""Turn Claude's per-candidate analyses into a final, non-overlapping list of
clips to render."""

from __future__ import annotations

from app.pipeline.vision import MomentAnalysis

# There is deliberately no "preferred" duration this module pushes clips
# toward - the vision system prompt already instructs Claude to pick the
# shortest COMPLETE, satisfying story (not the shortest clip, period):
# preserve a continuous incident (e.g. a collision immediately followed by
# a confrontation) through its natural payoff rather than cutting away at
# the single most dramatic frame - see incident_boundaries and
# is_continuous_single_event in vision.py, which analyze_candidate itself
# already enforces by widening end_seconds when a continuous event was cut
# short. Roughly 8-45s is preferred, up to ~90s when the story genuinely
# needs it. This module only guards the extremes: a true floor so a
# degenerate near-zero-length candidate can't reach rendering, and a
# ceiling so a runaway candidate doesn't produce an unreasonably long
# "short-form" clip. It never stretches a short, complete incident up to
# hit a target length, and it never shrinks a longer one either - only
# _clamp_duration's HARD_MAX truncation reduces an already-chosen window,
# and only when it exceeds the ceiling below.
HARD_MIN_CLIP_SECONDS = 6
HARD_MAX_CLIP_SECONDS = 90

# Below this, a candidate is closer to a glitch than a viable clip - not
# worth even padding up to the hard minimum.
UNSALVAGEABLE_SECONDS = HARD_MIN_CLIP_SECONDS * 0.6

# Quality floor for selection, matching the "0-40 ordinary/boring - nothing
# here should be selected" band the vision system prompt scores against.
# Claude is already instructed to set is_moment=false for genuinely weak
# footage, so this is defense-in-depth: prefer returning fewer clips (even
# zero) over padding out a requested count with filler. A video with three
# mediocre moments and one great one should produce one clip, not three.
MIN_SELECTABLE_SCORE = 40


def _clamp_duration(analysis: MomentAnalysis, video_duration: float) -> tuple[float, float]:
    start, end = analysis.start_seconds, analysis.end_seconds
    duration = end - start

    if duration < HARD_MIN_CLIP_SECONDS:
        deficit = HARD_MIN_CLIP_SECONDS - duration
        # Prefer extending the tail (let the payoff/reaction breathe) but pull
        # from the head too if we're near the end of the source video.
        end = min(video_duration, end + deficit * 0.6)
        start = max(0.0, start - (deficit - (end - analysis.end_seconds)))
    elif duration > HARD_MAX_CLIP_SECONDS:
        end = start + HARD_MAX_CLIP_SECONDS

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
    """Rank by total score and greedily pick the best non-overlapping set.

    Prefers quality over hitting the requested count exactly: candidates
    that Claude scored as weak (is_moment=True but a low total_score) are
    still eligible, but the ranking means a request for several clips on a
    video with only one genuinely good moment naturally returns just that
    one rather than padding out with mediocre filler - callers that want a
    hard quality floor can filter analyses by total_score before calling
    this.
    """
    ranked = sorted(
        (a for a in analyses if a.total_score >= MIN_SELECTABLE_SCORE),
        key=lambda a: a.total_score,
        reverse=True,
    )

    selected: list[MomentAnalysis] = []
    for a in ranked:
        if len(selected) >= number_of_clips:
            break

        start, end = _clamp_duration(a, video_duration)
        if end - start < UNSALVAGEABLE_SECONDS:
            continue  # too close to the edges of the source video to salvage

        # Narration cue timing is relative to the clip start Claude saw when
        # it proposed this window - if clamping pulled the head earlier to
        # hit the minimum duration, shift cues (and crop keyframes/effects,
        # which use the same clip-relative convention) by the same amount so
        # they stay aligned with the actual rendered clip start. Teaser uses
        # ABSOLUTE source-video seconds instead, so it doesn't need shifting -
        # app.pipeline.timeline validates it still falls within the final
        # [start, end] bounds at render time and disables it if not.
        head_shift = a.start_seconds - start
        if head_shift:
            for cue in a.narration_cues:
                cue.start_seconds += head_shift
            for kf in a.crop_keyframes:
                kf.time_seconds += head_shift
            for effect in a.effects:
                effect.start_seconds += head_shift
                effect.end_seconds += head_shift

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
