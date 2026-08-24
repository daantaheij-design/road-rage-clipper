from __future__ import annotations

import re

from app.pipeline.crop import (
    MAX_PAN_SPEED_PER_SECOND,
    MIN_TRUSTED_CONFIDENCE,
    CropKeyframe,
    build_crop_filter,
    prepare_keyframes,
)


def test_no_keyframes_falls_back_to_static_center():
    prepared = prepare_keyframes([], clip_duration=20.0)
    assert len(prepared) == 1
    assert prepared[0].focus_x == 0.5
    assert prepared[0].focus_y == 0.5


def test_low_confidence_keyframe_pulled_to_center():
    kfs = [CropKeyframe(time_seconds=0.0, focus_x=0.9, focus_y=0.9, confidence=0.1)]
    prepared = prepare_keyframes(kfs, clip_duration=10.0)
    assert prepared[0].confidence < MIN_TRUSTED_CONFIDENCE
    assert prepared[0].focus_x == 0.5
    assert prepared[0].focus_y == 0.5


def test_high_confidence_keyframe_is_trusted():
    kfs = [CropKeyframe(time_seconds=0.0, focus_x=0.8, focus_y=0.3, confidence=0.95)]
    prepared = prepare_keyframes(kfs, clip_duration=10.0)
    assert prepared[0].focus_x == 0.8
    assert prepared[0].focus_y == 0.3


def test_sequence_extended_to_cover_full_clip():
    kfs = [CropKeyframe(time_seconds=5.0, focus_x=0.7, focus_y=0.5, confidence=0.9)]
    prepared = prepare_keyframes(kfs, clip_duration=10.0)
    assert prepared[0].time_seconds == 0.0
    assert prepared[-1].time_seconds == 10.0


def test_out_of_order_and_out_of_range_keyframes_are_handled():
    kfs = [
        CropKeyframe(time_seconds=8.0, focus_x=0.9, focus_y=0.5, confidence=0.9),
        CropKeyframe(time_seconds=-2.0, focus_x=0.1, focus_y=0.5, confidence=0.9),
        CropKeyframe(time_seconds=100.0, focus_x=0.5, focus_y=0.5, confidence=0.9),
    ]
    prepared = prepare_keyframes(kfs, clip_duration=10.0)
    times = [kf.time_seconds for kf in prepared]
    assert times == sorted(times)
    assert times[0] == 0.0
    assert times[-1] == 10.0


def test_pan_speed_is_clamped_between_far_apart_focus_points():
    # A huge jump (0.0 -> 1.0) requested in a single second must be clamped.
    kfs = [
        CropKeyframe(time_seconds=0.0, focus_x=0.0, focus_y=0.5, confidence=0.9),
        CropKeyframe(time_seconds=1.0, focus_x=1.0, focus_y=0.5, confidence=0.9),
    ]
    prepared = prepare_keyframes(kfs, clip_duration=1.0)
    jump = prepared[1].focus_x - prepared[0].focus_x
    assert jump <= MAX_PAN_SPEED_PER_SECOND * 1.0 + 1e-6


def test_gradual_movement_is_not_clamped():
    kfs = [
        CropKeyframe(time_seconds=0.0, focus_x=0.4, focus_y=0.5, confidence=0.9),
        CropKeyframe(time_seconds=10.0, focus_x=0.6, focus_y=0.5, confidence=0.9),
    ]
    prepared = prepare_keyframes(kfs, clip_duration=10.0)
    assert prepared[-1].focus_x == 0.6


def test_build_crop_filter_fills_target_canvas_for_landscape_source():
    filt = build_crop_filter(
        [CropKeyframe(0.0, 0.5, 0.5, 0.9)],
        clip_duration=10.0,
        source_width=1920,
        source_height=1080,
        target_width=1080,
        target_height=1920,
    )
    assert "scale=1080:1920" in filt
    # crop height should be the full source height for a wide source.
    assert "h=1080" in filt
    # crop width should match the 9:16 aspect at that height (1080 * 9/16 = 607.5 -> even).
    w = int(re.search(r"w=(\d+)", filt).group(1))
    assert w % 2 == 0
    assert abs(w - 1080 * 9 / 16) <= 1


def test_build_crop_filter_pans_for_moving_focus():
    filt = build_crop_filter(
        [
            CropKeyframe(0.0, 0.1, 0.5, 0.9),
            CropKeyframe(10.0, 0.9, 0.5, 0.9),
        ],
        clip_duration=10.0,
        source_width=1920,
        source_height=1080,
        target_width=1080,
        target_height=1920,
    )
    # A time-varying x expression references t (not just a fixed constant) -
    # two keyframes collapse to a single linear segment, no `if` needed.
    x_expr = re.search(r"x='([^']+)'", filt).group(1)
    assert "(t-" in x_expr


def test_build_crop_filter_static_for_single_center_point():
    filt = build_crop_filter(
        [CropKeyframe(0.0, 0.5, 0.5, 0.9)],
        clip_duration=10.0,
        source_width=1920,
        source_height=1080,
        target_width=1080,
        target_height=1920,
    )
    x_expr = re.search(r"x='([^']+)'", filt).group(1)
    # A single, centered keyframe should produce a constant (no `if`/`t`).
    assert "if(" not in x_expr


def test_build_crop_filter_never_exceeds_source_bounds():
    # Focus point pinned at the extreme edge (1.0) must not push the crop
    # window past the source frame.
    filt = build_crop_filter(
        [CropKeyframe(0.0, 1.0, 0.5, 0.99)],
        clip_duration=1.0,
        source_width=1920,
        source_height=1080,
        target_width=1080,
        target_height=1920,
    )
    w = int(re.search(r"w=(\d+)", filt).group(1))
    x_expr = re.search(r"x='([^']+)'", filt).group(1)
    # focus_x=1.0 wants a crop center at the extreme right edge (x=1920),
    # which must be clamped to the last valid x position (1920 - w), not
    # left free to push the crop window past the source frame.
    assert f"{float(1920 - w):.4f}" in x_expr
    assert "1920.0000" not in x_expr


def test_build_crop_filter_handles_portrait_source():
    # A source narrower than the 9:16 target should crop height, keep full
    # width, and pan vertically instead of horizontally.
    filt = build_crop_filter(
        [CropKeyframe(0.0, 0.5, 0.2, 0.9)],
        clip_duration=5.0,
        source_width=1080,
        source_height=2400,  # taller than 9:16 relative to its width
        target_width=1080,
        target_height=1920,
    )
    assert "w=1080" in filt
    h = int(re.search(r"h=(\d+)", filt).group(1))
    assert h < 2400
    assert h % 2 == 0


def test_build_crop_filter_zero_duration_does_not_crash():
    filt = build_crop_filter(
        [CropKeyframe(0.0, 0.5, 0.5, 0.9)],
        clip_duration=0.0,
        source_width=1920,
        source_height=1080,
        target_width=1080,
        target_height=1920,
    )
    assert "crop=" in filt
