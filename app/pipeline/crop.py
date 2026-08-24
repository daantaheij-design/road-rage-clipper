"""Smart pan-and-scan cropping.

Turns Claude's per-clip focus-point analysis (where the important action is,
over time) into a smooth, time-varying ffmpeg crop expression that fills the
entire 1080x1920 vertical canvas - no blurred-background letterboxing. This
is the standard TikTok/Shorts reformat technique: crop a vertical slice out
of the horizontal frame and pan that slice to follow the subject, rather
than shrinking the whole frame to fit inside a blurred box.

Note on punch-zoom: ffmpeg's `crop` filter only re-evaluates `x`/`y` per
frame in this build - `w`/`h` are fixed at filter init (confirmed
empirically; passing a `t`-dependent expression as crop's `w`/`h` raises
"Error when evaluating the expression" at filter-graph configuration time).
So a punch-zoom effect can't be a continuously-varying zoom inside one
`build_crop_filter` call. Instead (see app.pipeline.timeline), a punch-zoom
window becomes its own short timeline segment with a *constant* zoom for
that segment's duration - `build_crop_filter`'s `zoom` parameter below - and
gets concatenated back with the surrounding normal-pan segments. That keeps
every individual crop filter call using only the x/y panning this build
actually supports per-frame.
"""

from __future__ import annotations

from dataclasses import dataclass

# Below this confidence, a keyframe's focus point isn't trusted - it's
# pulled toward frame center instead of risking a crop that follows a wrong
# guess about where the action is.
MIN_TRUSTED_CONFIDENCE = 0.5

# How fast the crop window is allowed to move, in normalized focus units
# (0-1 = full frame width/height) per second. Keeps pans smooth and
# intentional instead of jittery/shaky.
MAX_PAN_SPEED_PER_SECOND = 0.35


@dataclass
class CropKeyframe:
    time_seconds: float  # relative to the clip's own start (0 = clip start)
    focus_x: float = 0.5  # normalized 0-1, fraction of frame width
    focus_y: float = 0.5  # normalized 0-1, fraction of frame height
    confidence: float = 0.0


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def prepare_keyframes(keyframes: list[CropKeyframe], clip_duration: float) -> list[CropKeyframe]:
    """Clean up raw, model-provided keyframes into a safe, monotonic,
    speed-limited sequence that spans the whole clip and is ready to turn
    into an ffmpeg expression.

    - Out-of-range/unsorted input is tolerated (sorted, clamped into [0, clip_duration]).
    - Low-confidence points are pulled to frame center rather than trusted.
    - No keyframes at all -> a single centered point (static center crop).
    - The sequence is extended to cover t=0 and t=clip_duration so the crop
      expression is defined for the entire clip.
    - Consecutive points are speed-limited so the pan can't jump abruptly.
    """
    if not keyframes or clip_duration <= 0:
        return [CropKeyframe(time_seconds=0.0, focus_x=0.5, focus_y=0.5, confidence=1.0)]

    cleaned: list[CropKeyframe] = []
    for kf in sorted(keyframes, key=lambda k: k.time_seconds):
        t = max(0.0, min(clip_duration, kf.time_seconds))
        if kf.confidence < MIN_TRUSTED_CONFIDENCE:
            fx, fy = 0.5, 0.5
        else:
            fx, fy = _clamp01(kf.focus_x), _clamp01(kf.focus_y)
        cleaned.append(CropKeyframe(time_seconds=t, focus_x=fx, focus_y=fy, confidence=kf.confidence))

    # Merge points that land on (almost) the same timestamp - keep the later one.
    deduped: list[CropKeyframe] = []
    for kf in cleaned:
        if deduped and kf.time_seconds - deduped[-1].time_seconds < 0.05:
            deduped[-1] = kf
        else:
            deduped.append(kf)

    if deduped[0].time_seconds > 0.0:
        first = deduped[0]
        deduped.insert(0, CropKeyframe(0.0, first.focus_x, first.focus_y, first.confidence))
    if deduped[-1].time_seconds < clip_duration:
        last = deduped[-1]
        deduped.append(CropKeyframe(clip_duration, last.focus_x, last.focus_y, last.confidence))

    smoothed = [deduped[0]]
    for kf in deduped[1:]:
        prev = smoothed[-1]
        dt = max(0.001, kf.time_seconds - prev.time_seconds)
        max_delta = MAX_PAN_SPEED_PER_SECOND * dt
        fx = prev.focus_x + max(-max_delta, min(max_delta, kf.focus_x - prev.focus_x))
        fy = prev.focus_y + max(-max_delta, min(max_delta, kf.focus_y - prev.focus_y))
        smoothed.append(CropKeyframe(kf.time_seconds, fx, fy, kf.confidence))

    return smoothed


def interpolate_focus(prepared: list[CropKeyframe], t: float) -> tuple[float, float, float]:
    """Numeric (pure-Python) linear interpolation of a prepared keyframe
    track at time t - the same math the ffmpeg x/y expressions perform per
    frame. Returns (focus_x, focus_y, confidence). Used by geometry.py to
    place overlays, and by slice_keyframes_local to retime sub-ranges."""
    if not prepared:
        return 0.5, 0.5, 0.0
    if t <= prepared[0].time_seconds:
        return prepared[0].focus_x, prepared[0].focus_y, prepared[0].confidence
    if t >= prepared[-1].time_seconds:
        return prepared[-1].focus_x, prepared[-1].focus_y, prepared[-1].confidence
    for a, b in zip(prepared, prepared[1:], strict=False):
        if a.time_seconds <= t <= b.time_seconds:
            span = b.time_seconds - a.time_seconds
            frac = 0.0 if span < 1e-9 else (t - a.time_seconds) / span
            return (
                a.focus_x + (b.focus_x - a.focus_x) * frac,
                a.focus_y + (b.focus_y - a.focus_y) * frac,
                max(a.confidence, b.confidence),
            )
    return prepared[-1].focus_x, prepared[-1].focus_y, prepared[-1].confidence


def slice_keyframes_local(
    prepared: list[CropKeyframe], orig_start: float, orig_end: float, *, time_scale: float = 1.0
) -> list[CropKeyframe]:
    """Extract the portion of an already-`prepare_keyframes`d pan track
    covering [orig_start, orig_end] (clip-relative seconds) and re-express
    it as its own standalone, LOCAL-time (0 = orig_start) track - used when
    a sub-range of the clip becomes its own timeline segment (slow motion,
    replay) so its crop filter keeps following the same pan, retimed to
    that segment's own local `t`. `time_scale` stretches the local
    timestamps (1/speed) so the pan stays in sync with a slow-motion
    segment's stretched playback duration."""
    orig_start, orig_end = min(orig_start, orig_end), max(orig_start, orig_end)
    fx0, fy0, c0 = interpolate_focus(prepared, orig_start)
    fx1, fy1, c1 = interpolate_focus(prepared, orig_end)
    points = [CropKeyframe(0.0, fx0, fy0, c0)]
    for kf in prepared:
        if orig_start < kf.time_seconds < orig_end:
            points.append(CropKeyframe((kf.time_seconds - orig_start) * time_scale, kf.focus_x, kf.focus_y, kf.confidence))
    points.append(CropKeyframe((orig_end - orig_start) * time_scale, fx1, fy1, c1))
    return points


def _piecewise_linear_expr(points: list[tuple[float, float]]) -> str:
    """Build an ffmpeg expression (usable as a `crop` x/y value, which is
    re-evaluated every frame using the `t` variable) that linearly
    interpolates `value` between consecutive `(time, value)` points."""
    if len(points) == 1:
        return f"{points[0][1]:.4f}"

    def segment(i: int) -> str:
        t0, v0 = points[i]
        t1, v1 = points[i + 1]
        if t1 - t0 < 1e-6:
            inner = f"{v1:.4f}"
        else:
            inner = f"({v0:.4f}+(({v1:.4f})-({v0:.4f}))*(t-({t0:.4f}))/(({t1:.4f})-({t0:.4f})))"
        if i + 2 == len(points):
            return inner
        return f"if(lt(t,{t1:.4f}),{inner},{segment(i + 1)})"

    return segment(0)


def _base_crop_dims(source_width: int, source_height: int, target_width: int, target_height: int) -> tuple[int, int, bool]:
    """The crop window size (at zoom=1.0) that fills target_width x
    target_height from source_width x source_height. Returns (crop_w,
    crop_h, pans_horizontally)."""
    target_aspect = target_width / target_height
    source_aspect = source_width / source_height
    if source_aspect >= target_aspect:
        crop_h = source_height - (source_height % 2)
        crop_w = int(round(crop_h * target_aspect))
        crop_w -= crop_w % 2
        crop_w = max(2, min(crop_w, source_width - (source_width % 2)))
        return crop_w, crop_h, True
    crop_w = source_width - (source_width % 2)
    crop_h = int(round(crop_w / target_aspect))
    crop_h -= crop_h % 2
    crop_h = max(2, min(crop_h, source_height - (source_height % 2)))
    return crop_w, crop_h, False


def crop_window_at(
    keyframes: list[CropKeyframe],
    t: float,
    *,
    clip_duration: float,
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
    zoom: float = 1.0,
) -> tuple[float, float, float, float]:
    """Numeric equivalent of what build_crop_filter's ffmpeg expression
    evaluates at time t: the crop window (x, y, w, h) in source pixels.
    Used by geometry.py to place overlays at the same position the crop
    filter will actually put them after cropping - must stay in lockstep
    with build_crop_filter's math below."""
    prepared = prepare_keyframes(keyframes, clip_duration)
    if source_width <= 0 or source_height <= 0:
        source_width, source_height = target_width, target_height
    base_w, base_h, horizontal = _base_crop_dims(source_width, source_height, target_width, target_height)
    zoom = max(1.0, zoom)
    crop_w = max(2.0, base_w / zoom)
    crop_h = max(2.0, base_h / zoom)
    fx, fy, _ = interpolate_focus(prepared, t)
    if horizontal:
        max_x = max(0.0, source_width - crop_w)
        x = min(max_x, max(0.0, fx * source_width - crop_w / 2))
        y = max(0.0, (source_height - crop_h) / 2)
    else:
        max_y = max(0.0, source_height - crop_h)
        y = min(max_y, max(0.0, fy * source_height - crop_h / 2))
        x = max(0.0, (source_width - crop_w) / 2)
    return x, y, crop_w, crop_h


def build_crop_filter(
    keyframes: list[CropKeyframe],
    *,
    clip_duration: float,
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
    zoom: float = 1.0,
) -> str:
    """Return an ffmpeg filter fragment (`crop=...,scale=...`) that always
    fills exactly target_width x target_height, panning within the source
    frame over time to follow `keyframes`. `zoom` (constant for the whole
    call - see module docstring for why it can't vary within one call)
    shrinks the crop window so the result reads as zoomed in; used for a
    punch-zoom effect's own short timeline segment."""
    prepared = prepare_keyframes(keyframes, clip_duration)

    if source_width <= 0 or source_height <= 0:
        source_width, source_height = target_width, target_height  # degenerate guard

    base_w, base_h, horizontal = _base_crop_dims(source_width, source_height, target_width, target_height)
    zoom = max(1.0, zoom)
    crop_w = max(2, int(round(base_w / zoom)))
    crop_h = max(2, int(round(base_h / zoom)))
    crop_w -= crop_w % 2
    crop_h -= crop_h % 2
    crop_w, crop_h = max(2, crop_w), max(2, crop_h)

    if horizontal:
        max_x = max(0, source_width - crop_w)
        x_points = [(kf.time_seconds, min(max_x, max(0.0, kf.focus_x * source_width - crop_w / 2))) for kf in prepared]
        x_expr = _piecewise_linear_expr(x_points)
        y_expr = f"{max(0, (source_height - crop_h) // 2)}"
    else:
        max_y = max(0, source_height - crop_h)
        y_points = [(kf.time_seconds, min(max_y, max(0.0, kf.focus_y * source_height - crop_h / 2))) for kf in prepared]
        y_expr = _piecewise_linear_expr(y_points)
        x_expr = f"{max(0, (source_width - crop_w) // 2)}"

    return f"crop=w={crop_w}:h={crop_h}:x='{x_expr}':y='{y_expr}',scale={target_width}:{target_height}:flags=bicubic"
