"""Turns a clip's persisted effects/teaser into a concrete render plan: an
ordered list of timeline segments (normal playback, freeze, slow motion,
replay, punch-zoom, teaser) plus the resulting expected output duration and
a function to remap any ORIGINAL clip-relative timestamp (as narration
cues, word timings, and crop keyframes are all expressed) onto its new
position in the final rendered timeline.

This is pure, deterministic, and render-time-only: it takes the persisted
Clip fields (effects, teaser) and recomputes the plan fresh on every render
attempt (including a Retry Render) rather than trusting a stale snapshot -
no Anthropic/ElevenLabs calls involved, so recomputing costs nothing and
guarantees a retry always matches what would be built from scratch.

Segment kinds and what they mean for the timeline:
- normal / slow_motion / zoom: each consumes a distinct sub-range of the
  clip's own source footage (zoom and slow_motion *replace* the normal
  playback for their window - zoom is the same source duration, just a
  tighter crop; slow_motion stretches the same source range over more
  output time).
- freeze / replay: point insertions - they don't consume new source range,
  they add extra output time at a specific point (freeze: a held frame;
  replay: a repeat of a source range that already played normally).
- teaser: prepended before everything else, its own (still within-clip)
  source range played early as a preview.

All effect windows/points are validated and clamped defensively here too
(not just at the point they were first persisted) since this module is the
one place that actually turns them into ffmpeg-facing segments - it must
never hand render.py a segment that would produce zero/negative duration
or reference time outside the clip.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.jobs.models import Clip, Effect, Teaser
from app.pipeline.crop import CropKeyframe, interpolate_focus, prepare_keyframes, slice_keyframes_local

FREEZE_MIN_SECONDS = 0.2
FREEZE_MAX_SECONDS = 1.0
SLOWMO_MIN_SPEED = 0.3
SLOWMO_MAX_SPEED = 0.9
REPLAY_MIN_SOURCE_SECONDS = 0.2
REPLAY_MAX_SOURCE_SECONDS = 3.0
REPLAY_MIN_SPEED = 0.5
REPLAY_MAX_SPEED = 1.0
PUNCH_ZOOM_MIN_SECONDS = 0.2
PUNCH_ZOOM_MAX_SECONDS = 2.0
ZOOM_MIN = 1.05
ZOOM_MAX = 1.5
TEASER_MIN_SECONDS = 0.3
TEASER_MAX_SECONDS = 3.0

# Audio treatment per segment kind - see module docstring's slow_motion/
# freeze notes on why these are muted/reduced rather than pitch-shifted.
_SEGMENT_VOLUME = {
    "normal": 1.0,
    "zoom": 1.0,
    "slow_motion": 0.0,
    "freeze": 0.0,
    "replay": 0.6,
    "teaser": 1.0,
}


@dataclass
class SegmentPlan:
    kind: str  # normal | freeze | slow_motion | replay | zoom | teaser
    source_start: float  # absolute seconds within the clip's own source window
    source_end: float
    output_duration: float
    speed: float = 1.0
    volume: float = 1.0
    zoom: float = 1.0
    crop_keyframes: list[CropKeyframe] = field(default_factory=list)  # local time, 0 = segment start


@dataclass
class RenderPlan:
    segments: list[SegmentPlan]
    expected_duration: float
    clip_start: float
    clip_end: float

    def remap(self, clip_relative_t: float) -> float:
        """Map a timestamp from the ORIGINAL (pre-effects) clip-relative
        timeline - as narration cue starts, word timings, and overlay
        effect windows are all expressed - onto the final rendered
        timeline (0 = start of the rendered file, after any teaser)."""
        abs_t = self.clip_start + clip_relative_t
        elapsed = 0.0
        for seg in self.segments:
            if seg.kind in ("normal", "slow_motion", "zoom") and seg.source_start <= abs_t <= seg.source_end:
                local = (abs_t - seg.source_start) / max(seg.speed, 1e-6)
                return elapsed + local
            elapsed += seg.output_duration
        return elapsed


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _valid_window(eff: Effect, clip_duration: float) -> tuple[float, float] | None:
    start = _clamp(eff.start_seconds, 0.0, clip_duration)
    end = _clamp(eff.end_seconds, 0.0, clip_duration)
    if end <= start:
        return None
    return start, end


def _select_non_overlapping(effects: list[Effect], clip_duration: float) -> list[tuple[Effect, float, float]]:
    """Timeline-restructuring effects (freeze/slow_motion/replay/punch_zoom)
    must not overlap each other - keep the earliest-starting of any
    conflicting pair."""
    windows: list[tuple[Effect, float, float]] = []
    for eff in effects:
        if eff.type not in ("freeze", "slow_motion", "replay", "punch_zoom"):
            continue
        w = _valid_window(eff, clip_duration)
        if w is None:
            continue
        windows.append((eff, w[0], w[1]))
    windows.sort(key=lambda item: item[1])

    kept: list[tuple[Effect, float, float]] = []
    for eff, start, end in windows:
        if kept and start < kept[-1][2]:
            continue  # overlaps the previously kept effect - drop it
        kept.append((eff, start, end))
    return kept


def _validate_teaser(teaser: Teaser | None, clip_start: float, clip_end: float, video_duration: float) -> Teaser | None:
    if teaser is None or not teaser.enabled:
        return None
    start, end = teaser.source_start, teaser.source_end
    if end <= start:
        return None
    # Must be footage from within this clip's own window (a preview of a
    # later moment in the SAME incident) and within the actual source video.
    if start < clip_start or end > clip_end or start < 0 or end > video_duration:
        return None
    duration = _clamp(end - start, TEASER_MIN_SECONDS, TEASER_MAX_SECONDS)
    end = start + duration
    if end > clip_end:
        return None
    return Teaser(enabled=True, source_start=start, source_end=end)


def _local_crop(base_plan: list[CropKeyframe], clip_start: float, abs_start: float, abs_end: float, speed: float) -> list[CropKeyframe]:
    return slice_keyframes_local(
        base_plan, abs_start - clip_start, abs_end - clip_start, time_scale=1.0 / max(speed, 1e-6)
    )


def _static_crop(base_plan: list[CropKeyframe], clip_start: float, abs_t: float, target_focus: tuple[float, float] | None) -> list[CropKeyframe]:
    if target_focus is not None:
        fx, fy = target_focus
        return [CropKeyframe(0.0, fx, fy, 1.0)]
    fx, fy, conf = interpolate_focus(base_plan, abs_t - clip_start)
    return [CropKeyframe(0.0, fx, fy, conf)]


def build_render_plan(
    clip: Clip,
    *,
    video_duration: float,
    raw_crop_keyframes: list[CropKeyframe] | None = None,
) -> RenderPlan:
    """The authoritative derivation of a clip's full render plan from its
    persisted effects/teaser. `raw_crop_keyframes` defaults to
    clip.crop_keyframes (converted) if not given."""
    clip_start, clip_end = clip.start_seconds, clip.end_seconds
    clip_duration = clip_end - clip_start
    if clip_duration <= 0:
        return RenderPlan(
            segments=[SegmentPlan("normal", clip_start, clip_end, max(0.0, clip_duration))],
            expected_duration=max(0.0, clip_duration),
            clip_start=clip_start,
            clip_end=clip_end,
        )

    if raw_crop_keyframes is None:
        raw_crop_keyframes = [
            CropKeyframe(kf.time_seconds, kf.focus_x, kf.focus_y, kf.confidence) for kf in clip.crop_keyframes
        ]
    base_plan = prepare_keyframes(raw_crop_keyframes, clip_duration)

    kept = _select_non_overlapping(clip.effects, clip_duration)

    # Phase A: substitution effects (zoom, slow_motion) replace a sub-range
    # of normal playback.
    substitutions = [(eff, s, e) for eff, s, e in kept if eff.type in ("slow_motion", "punch_zoom")]
    substitutions.sort(key=lambda item: item[1])

    segments: list[SegmentPlan] = []
    cursor = clip_start
    for eff, start, end in substitutions:
        abs_start, abs_end = clip_start + start, clip_start + end
        if abs_start > cursor:
            segments.append(_normal_segment(base_plan, clip_start, cursor, abs_start))
        if eff.type == "slow_motion":
            speed = _clamp(eff.speed, SLOWMO_MIN_SPEED, SLOWMO_MAX_SPEED)
            duration = (abs_end - abs_start) / speed
            segments.append(
                SegmentPlan(
                    "slow_motion",
                    abs_start,
                    abs_end,
                    duration,
                    speed=speed,
                    volume=_SEGMENT_VOLUME["slow_motion"],
                    crop_keyframes=_local_crop(base_plan, clip_start, abs_start, abs_end, speed),
                )
            )
        else:  # punch_zoom
            zoom = _clamp(eff.zoom, ZOOM_MIN, ZOOM_MAX)
            target_focus = None
            if eff.target and eff.target.bbox and eff.target.confidence >= 0.0:
                target_focus = _bbox_focus(eff.target)
            mid = (start + end) / 2
            crop = _static_crop(base_plan, clip_start, clip_start + mid, target_focus)
            segments.append(
                SegmentPlan(
                    "zoom",
                    abs_start,
                    abs_end,
                    abs_end - abs_start,
                    speed=1.0,
                    volume=_SEGMENT_VOLUME["zoom"],
                    zoom=zoom,
                    crop_keyframes=crop,
                )
            )
        cursor = max(cursor, abs_end)

    if cursor < clip_end:
        segments.append(_normal_segment(base_plan, clip_start, cursor, clip_end))

    # Phase B: point-insertion effects (freeze, replay) split whichever
    # segment currently contains their point and are inserted there.
    points = [(eff, s, e) for eff, s, e in kept if eff.type in ("freeze", "replay")]
    points.sort(key=lambda item: item[1])
    for eff, start, end in points:
        if eff.type == "freeze":
            duration = _clamp(end - start, FREEZE_MIN_SECONDS, FREEZE_MAX_SECONDS)
            abs_point = clip_start + start
            segments = _insert_point_segment(
                segments,
                abs_point,
                SegmentPlan(
                    "freeze",
                    abs_point,
                    abs_point,
                    duration,
                    speed=1.0,
                    volume=_SEGMENT_VOLUME["freeze"],
                    crop_keyframes=_static_crop(base_plan, clip_start, abs_point, None),
                ),
                base_plan,
                clip_start,
            )
        else:  # replay
            src_duration = _clamp(end - start, REPLAY_MIN_SOURCE_SECONDS, REPLAY_MAX_SOURCE_SECONDS)
            abs_start = clip_start + start
            abs_end = abs_start + src_duration
            speed = _clamp(eff.speed, REPLAY_MIN_SPEED, REPLAY_MAX_SPEED)
            abs_point = clip_start + end
            segments = _insert_point_segment(
                segments,
                abs_point,
                SegmentPlan(
                    "replay",
                    abs_start,
                    abs_end,
                    (abs_end - abs_start) / speed,
                    speed=speed,
                    volume=_SEGMENT_VOLUME["replay"],
                    crop_keyframes=_local_crop(base_plan, clip_start, abs_start, abs_end, speed),
                ),
                base_plan,
                clip_start,
            )

    valid_teaser = _validate_teaser(clip.teaser, clip_start, clip_end, video_duration)
    if valid_teaser is not None:
        teaser_crop = _local_crop(base_plan, clip_start, valid_teaser.source_start, valid_teaser.source_end, 1.0)
        segments.insert(
            0,
            SegmentPlan(
                "teaser",
                valid_teaser.source_start,
                valid_teaser.source_end,
                valid_teaser.source_end - valid_teaser.source_start,
                speed=1.0,
                volume=_SEGMENT_VOLUME["teaser"],
                crop_keyframes=teaser_crop,
            ),
        )

    segments = [s for s in segments if s.output_duration > 1e-6]
    expected_duration = sum(s.output_duration for s in segments)
    return RenderPlan(segments=segments, expected_duration=expected_duration, clip_start=clip_start, clip_end=clip_end)


def _bbox_focus(target) -> tuple[float, float]:
    b = target.bbox.clamped()
    return b.x + b.width / 2, b.y + b.height / 2


def _normal_segment(base_plan: list[CropKeyframe], clip_start: float, abs_start: float, abs_end: float) -> SegmentPlan:
    return SegmentPlan(
        "normal",
        abs_start,
        abs_end,
        abs_end - abs_start,
        speed=1.0,
        volume=_SEGMENT_VOLUME["normal"],
        crop_keyframes=_local_crop(base_plan, clip_start, abs_start, abs_end, 1.0),
    )


def _insert_point_segment(
    segments: list[SegmentPlan],
    abs_point: float,
    new_segment: SegmentPlan,
    base_plan: list[CropKeyframe],
    clip_start: float,
) -> list[SegmentPlan]:
    """Split whichever segment in `segments` currently spans abs_point and
    insert `new_segment` there. Falls back to appending at the end if no
    segment contains the point (shouldn't normally happen given upstream
    clamping, but never silently drop a validated effect)."""
    for i, seg in enumerate(segments):
        if seg.kind not in ("normal", "slow_motion") or not (seg.source_start <= abs_point <= seg.source_end):
            continue
        before = _resize_segment(seg, seg.source_start, abs_point, base_plan, clip_start)
        after = _resize_segment(seg, abs_point, seg.source_end, base_plan, clip_start)
        replacement = [s for s in (before,) if s.output_duration > 1e-6]
        replacement.append(new_segment)
        replacement += [s for s in (after,) if s.output_duration > 1e-6]
        return segments[:i] + replacement + segments[i + 1 :]
    return segments + [new_segment]


def _resize_segment(
    seg: SegmentPlan, new_start: float, new_end: float, base_plan: list[CropKeyframe], clip_start: float
) -> SegmentPlan:
    duration = (new_end - new_start) / max(seg.speed, 1e-6)
    return SegmentPlan(
        seg.kind,
        new_start,
        new_end,
        duration,
        speed=seg.speed,
        volume=seg.volume,
        zoom=seg.zoom,
        crop_keyframes=_local_crop(base_plan, clip_start, new_start, new_end, seg.speed),
    )
