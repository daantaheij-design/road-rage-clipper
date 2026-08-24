"""Smart pan-and-scan cropping.

Turns Claude's per-clip focus-point analysis (where the important action is,
over time) into a smooth, time-varying ffmpeg crop expression that fills the
entire 1080x1920 vertical canvas - no blurred-background letterboxing. This
is the standard TikTok/Shorts reformat technique: crop a vertical slice out
of the horizontal frame and pan that slice to follow the subject, rather
than shrinking the whole frame to fit inside a blurred box.
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


def build_crop_filter(
    keyframes: list[CropKeyframe],
    *,
    clip_duration: float,
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
) -> str:
    """Return an ffmpeg filter fragment (`crop=...,scale=...`) that always
    fills exactly target_width x target_height, panning within the source
    frame over time to follow `keyframes`."""
    prepared = prepare_keyframes(keyframes, clip_duration)

    if source_width <= 0 or source_height <= 0:
        source_width, source_height = target_width, target_height  # degenerate guard

    target_aspect = target_width / target_height  # e.g. 9/16
    source_aspect = source_width / source_height

    if source_aspect >= target_aspect:
        # Source is at least as wide (relative to its height) as the target
        # - the normal horizontal-dashcam case. Use the full source height,
        # crop a narrower width, and pan that crop window left/right.
        crop_h = source_height - (source_height % 2)
        crop_w = int(round(crop_h * target_aspect))
        crop_w -= crop_w % 2
        crop_w = max(2, min(crop_w, source_width - (source_width % 2)))
        max_x = max(0, source_width - crop_w)
        x_points = [(kf.time_seconds, min(max_x, max(0.0, kf.focus_x * source_width - crop_w / 2))) for kf in prepared]
        x_expr = _piecewise_linear_expr(x_points)
        y_expr = f"{max(0, (source_height - crop_h) // 2)}"
    else:
        # Source is narrower/taller than the target aspect - crop height
        # instead, keep the full width, and pan up/down.
        crop_w = source_width - (source_width % 2)
        crop_h = int(round(crop_w / target_aspect))
        crop_h -= crop_h % 2
        crop_h = max(2, min(crop_h, source_height - (source_height % 2)))
        max_y = max(0, source_height - crop_h)
        y_points = [(kf.time_seconds, min(max_y, max(0.0, kf.focus_y * source_height - crop_h / 2))) for kf in prepared]
        y_expr = _piecewise_linear_expr(y_points)
        x_expr = f"{max(0, (source_width - crop_w) // 2)}"

    return f"crop=w={crop_w}:h={crop_h}:x='{x_expr}':y='{y_expr}',scale={target_width}:{target_height}:flags=bicubic"
