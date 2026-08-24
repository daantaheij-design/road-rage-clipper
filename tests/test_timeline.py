from __future__ import annotations

import pytest

from app.jobs.models import BBox, Clip, Effect, Target, Teaser
from app.pipeline.crop import CropKeyframe
from app.pipeline.timeline import (
    FREEZE_MAX_SECONDS,
    FREEZE_MIN_SECONDS,
    REPLAY_MAX_SOURCE_SECONDS,
    SLOWMO_MAX_SPEED,
    SLOWMO_MIN_SPEED,
    TEASER_MAX_SECONDS,
    build_render_plan,
)


def _clip(start=100.0, end=120.0, **kw) -> Clip:
    return Clip(start_seconds=start, end_seconds=end, duration_seconds=end - start, **kw)


def test_no_effects_produces_single_normal_segment_spanning_the_clip():
    clip = _clip()
    plan = build_render_plan(clip, video_duration=600.0)
    assert len(plan.segments) == 1
    seg = plan.segments[0]
    assert seg.kind == "normal"
    assert seg.source_start == 100.0
    assert seg.source_end == 120.0
    assert plan.expected_duration == pytest.approx(20.0)


def test_expected_duration_equals_clip_duration_with_no_effects():
    clip = _clip(start=0.0, end=30.0)
    plan = build_render_plan(clip, video_duration=600.0)
    assert plan.expected_duration == pytest.approx(30.0)


def test_remap_is_identity_with_no_effects():
    clip = _clip()
    plan = build_render_plan(clip, video_duration=600.0)
    for t in (0.0, 5.0, 19.9):
        assert plan.remap(t) == pytest.approx(t)


def test_freeze_effect_adds_its_duration_to_expected_output():
    clip = _clip()
    clip.effects = [Effect(type="freeze", start_seconds=10.0, end_seconds=10.5)]
    plan = build_render_plan(clip, video_duration=600.0)
    assert plan.expected_duration == pytest.approx(20.0 + 0.5)
    freeze_segs = [s for s in plan.segments if s.kind == "freeze"]
    assert len(freeze_segs) == 1
    assert freeze_segs[0].output_duration == pytest.approx(0.5)


def test_freeze_duration_clamped_to_max():
    clip = _clip()
    clip.effects = [Effect(type="freeze", start_seconds=10.0, end_seconds=10.0 + FREEZE_MAX_SECONDS + 5.0)]
    plan = build_render_plan(clip, video_duration=600.0)
    freeze_segs = [s for s in plan.segments if s.kind == "freeze"]
    assert freeze_segs[0].output_duration <= FREEZE_MAX_SECONDS + 1e-6


def test_freeze_duration_clamped_to_min():
    clip = _clip()
    clip.effects = [Effect(type="freeze", start_seconds=10.0, end_seconds=10.01)]
    plan = build_render_plan(clip, video_duration=600.0)
    freeze_segs = [s for s in plan.segments if s.kind == "freeze"]
    assert freeze_segs[0].output_duration >= FREEZE_MIN_SECONDS - 1e-6


def test_slow_motion_stretches_output_duration_by_speed():
    clip = _clip()
    clip.effects = [Effect(type="slow_motion", start_seconds=5.0, end_seconds=6.0, speed=0.5)]
    plan = build_render_plan(clip, video_duration=600.0)
    slow_segs = [s for s in plan.segments if s.kind == "slow_motion"]
    assert len(slow_segs) == 1
    assert slow_segs[0].output_duration == pytest.approx(1.0 / 0.5)
    assert plan.expected_duration == pytest.approx(20.0 - 1.0 + 1.0 / 0.5)


def test_slow_motion_speed_clamped_to_valid_range():
    clip = _clip()
    clip.effects = [Effect(type="slow_motion", start_seconds=5.0, end_seconds=6.0, speed=0.01)]
    plan = build_render_plan(clip, video_duration=600.0)
    slow_segs = [s for s in plan.segments if s.kind == "slow_motion"]
    assert slow_segs[0].speed >= SLOWMO_MIN_SPEED - 1e-6

    clip2 = _clip()
    clip2.effects = [Effect(type="slow_motion", start_seconds=5.0, end_seconds=6.0, speed=5.0)]
    plan2 = build_render_plan(clip2, video_duration=600.0)
    slow_segs2 = [s for s in plan2.segments if s.kind == "slow_motion"]
    assert slow_segs2[0].speed <= SLOWMO_MAX_SPEED + 1e-6


def test_replay_adds_extra_duration_after_original_playthrough():
    clip = _clip()
    clip.effects = [Effect(type="replay", start_seconds=5.0, end_seconds=6.0, speed=1.0)]
    plan = build_render_plan(clip, video_duration=600.0)
    replay_segs = [s for s in plan.segments if s.kind == "replay"]
    assert len(replay_segs) == 1
    assert replay_segs[0].output_duration == pytest.approx(1.0)
    assert plan.expected_duration == pytest.approx(20.0 + 1.0)


def test_replay_source_range_clamped_to_max():
    clip = _clip()
    clip.effects = [Effect(type="replay", start_seconds=5.0, end_seconds=5.0 + REPLAY_MAX_SOURCE_SECONDS + 5.0, speed=1.0)]
    plan = build_render_plan(clip, video_duration=600.0)
    replay_segs = [s for s in plan.segments if s.kind == "replay"]
    assert (replay_segs[0].source_end - replay_segs[0].source_start) <= REPLAY_MAX_SOURCE_SECONDS + 1e-6


def test_punch_zoom_does_not_add_output_duration():
    clip = _clip()
    clip.effects = [Effect(type="punch_zoom", start_seconds=5.0, end_seconds=6.0, zoom=1.3)]
    plan = build_render_plan(clip, video_duration=600.0)
    assert plan.expected_duration == pytest.approx(20.0)
    zoom_segs = [s for s in plan.segments if s.kind == "zoom"]
    assert len(zoom_segs) == 1
    assert zoom_segs[0].zoom == pytest.approx(1.3)


def test_overlapping_timeline_effects_keep_only_the_earliest():
    clip = _clip()
    clip.effects = [
        Effect(type="slow_motion", start_seconds=5.0, end_seconds=8.0, speed=0.5),
        Effect(type="freeze", start_seconds=6.0, end_seconds=6.5),  # overlaps the slow_motion above
    ]
    plan = build_render_plan(clip, video_duration=600.0)
    kinds = [s.kind for s in plan.segments]
    assert "slow_motion" in kinds
    assert "freeze" not in kinds


def test_effect_timestamps_are_clamped_into_clip_bounds():
    clip = _clip()
    clip.effects = [Effect(type="freeze", start_seconds=-5.0, end_seconds=1000.0)]
    plan = build_render_plan(clip, video_duration=600.0)
    freeze_segs = [s for s in plan.segments if s.kind == "freeze"]
    assert len(freeze_segs) == 1
    assert 0 < freeze_segs[0].output_duration <= FREEZE_MAX_SECONDS + 1e-6


def test_invalid_effect_window_is_dropped():
    clip = _clip()
    clip.effects = [Effect(type="freeze", start_seconds=10.0, end_seconds=10.0)]  # zero-width, invalid
    plan = build_render_plan(clip, video_duration=600.0)
    assert all(s.kind != "freeze" for s in plan.segments)


def test_teaser_is_prepended_and_adds_its_duration():
    clip = _clip()
    clip.teaser = Teaser(enabled=True, source_start=115.0, source_end=116.0)
    plan = build_render_plan(clip, video_duration=600.0)
    assert plan.segments[0].kind == "teaser"
    assert plan.segments[0].output_duration == pytest.approx(1.0)
    assert plan.expected_duration == pytest.approx(21.0)


def test_teaser_outside_clip_bounds_is_disabled():
    clip = _clip()
    clip.teaser = Teaser(enabled=True, source_start=50.0, source_end=51.0)  # before clip.start_seconds=100
    plan = build_render_plan(clip, video_duration=600.0)
    assert all(s.kind != "teaser" for s in plan.segments)
    assert plan.expected_duration == pytest.approx(20.0)


def test_teaser_beyond_video_duration_is_disabled():
    clip = _clip(start=590.0, end=598.0)
    clip.teaser = Teaser(enabled=True, source_start=596.0, source_end=610.0)  # source_end > video_duration
    plan = build_render_plan(clip, video_duration=600.0)
    assert all(s.kind != "teaser" for s in plan.segments)


def test_teaser_duration_clamped_to_max():
    clip = _clip(start=0.0, end=90.0)
    clip.teaser = Teaser(enabled=True, source_start=10.0, source_end=10.0 + TEASER_MAX_SECONDS + 10.0)
    plan = build_render_plan(clip, video_duration=600.0)
    teaser_segs = [s for s in plan.segments if s.kind == "teaser"]
    assert len(teaser_segs) == 1
    assert teaser_segs[0].output_duration <= TEASER_MAX_SECONDS + 1e-6


def test_disabled_teaser_produces_no_segment():
    clip = _clip()
    clip.teaser = Teaser(enabled=False, source_start=115.0, source_end=116.0)
    plan = build_render_plan(clip, video_duration=600.0)
    assert all(s.kind != "teaser" for s in plan.segments)


def test_remap_shifts_everything_after_a_freeze():
    clip = _clip()
    clip.effects = [Effect(type="freeze", start_seconds=10.0, end_seconds=10.5)]
    plan = build_render_plan(clip, video_duration=600.0)
    assert plan.remap(5.0) == pytest.approx(5.0)  # before the freeze - unaffected
    assert plan.remap(15.0) == pytest.approx(15.5)  # after the freeze - shifted by its duration


def test_remap_accounts_for_teaser_offset():
    clip = _clip()
    clip.teaser = Teaser(enabled=True, source_start=115.0, source_end=116.5)
    plan = build_render_plan(clip, video_duration=600.0)
    assert plan.remap(0.0) == pytest.approx(1.5)
    assert plan.remap(5.0) == pytest.approx(6.5)


def test_remap_within_slow_motion_window_is_stretched():
    clip = _clip()
    clip.effects = [Effect(type="slow_motion", start_seconds=5.0, end_seconds=6.0, speed=0.5)]
    plan = build_render_plan(clip, video_duration=600.0)
    assert plan.remap(5.0) == pytest.approx(5.0)
    assert plan.remap(5.5) == pytest.approx(6.0)  # halfway through source -> at 2x the stretched output time
    assert plan.remap(6.0) == pytest.approx(7.0)
    assert plan.remap(10.0) == pytest.approx(11.0)  # after the stretch, shifted by the extra 1.0s


def test_expected_duration_equals_sum_of_segment_output_durations():
    clip = _clip()
    clip.effects = [
        Effect(type="freeze", start_seconds=3.0, end_seconds=3.5),
        Effect(type="slow_motion", start_seconds=10.0, end_seconds=11.0, speed=0.5),
    ]
    clip.teaser = Teaser(enabled=True, source_start=117.0, source_end=118.0)
    plan = build_render_plan(clip, video_duration=600.0)
    assert plan.expected_duration == pytest.approx(sum(s.output_duration for s in plan.segments))


def test_zero_duration_clip_does_not_crash():
    clip = _clip(start=100.0, end=100.0)
    plan = build_render_plan(clip, video_duration=600.0)
    assert plan.expected_duration >= 0


def test_crop_keyframes_default_to_clip_crop_keyframes_when_not_overridden():
    from app.jobs.models import CropKeyframe as CropKeyframeRecord

    clip = _clip()
    clip.crop_keyframes = [CropKeyframeRecord(time_seconds=0.0, focus_x=0.7, focus_y=0.3, confidence=0.9)]
    plan = build_render_plan(clip, video_duration=600.0)
    seg = plan.segments[0]
    assert any(abs(kf.focus_x - 0.7) < 1e-6 for kf in seg.crop_keyframes)


def test_target_bbox_biases_zoom_segment_focus():
    clip = _clip()
    clip.effects = [
        Effect(
            type="punch_zoom",
            start_seconds=5.0,
            end_seconds=6.0,
            zoom=1.3,
            target=Target(description="car", bbox=BBox(x=0.6, y=0.3, width=0.2, height=0.2), confidence=0.9),
        )
    ]
    plan = build_render_plan(clip, video_duration=600.0)
    zoom_seg = next(s for s in plan.segments if s.kind == "zoom")
    assert zoom_seg.crop_keyframes[0].focus_x == pytest.approx(0.7)  # bbox center x = 0.6+0.2/2
    assert zoom_seg.crop_keyframes[0].focus_y == pytest.approx(0.4)


def test_freeze_and_replay_can_coexist_without_overlap():
    clip = _clip()
    clip.effects = [
        Effect(type="freeze", start_seconds=3.0, end_seconds=3.5),
        Effect(type="replay", start_seconds=10.0, end_seconds=11.0, speed=1.0),
    ]
    plan = build_render_plan(clip, video_duration=600.0)
    kinds = [s.kind for s in plan.segments]
    assert "freeze" in kinds
    assert "replay" in kinds


def test_crop_keyframes_used_instead_of_clip_field_when_raw_override_given():
    clip = _clip()
    override = [CropKeyframe(0.0, 0.9, 0.1, 0.9)]
    plan = build_render_plan(clip, video_duration=600.0, raw_crop_keyframes=override)
    assert any(abs(kf.focus_x - 0.9) < 1e-6 for kf in plan.segments[0].crop_keyframes)
