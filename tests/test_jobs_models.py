from __future__ import annotations

from app.jobs.models import BBox, Clip, Effect, Target


def test_bbox_clamped_keeps_valid_box_unchanged():
    b = BBox(x=0.2, y=0.3, width=0.1, height=0.15).clamped()
    assert b.x == 0.2
    assert b.y == 0.3
    assert b.width == 0.1
    assert b.height == 0.15


def test_bbox_clamped_clamps_out_of_range_coordinates():
    b = BBox(x=-0.5, y=1.5, width=0.1, height=0.1).clamped()
    assert 0.0 <= b.x <= 1.0
    assert 0.0 <= b.y <= 1.0


def test_bbox_clamped_clamps_width_and_height_to_positive():
    b = BBox(x=0.5, y=0.5, width=0.0, height=-1.0).clamped()
    assert b.width > 0
    assert b.height > 0


def test_bbox_clamped_never_extends_past_frame_edge():
    b = BBox(x=0.9, y=0.9, width=0.5, height=0.5).clamped()
    assert b.x + b.width <= 1.0 + 1e-9
    assert b.y + b.height <= 1.0 + 1e-9


def test_bbox_clamped_handles_fully_out_of_bounds_box():
    b = BBox(x=2.0, y=2.0, width=2.0, height=2.0).clamped()
    assert 0.0 <= b.x <= 1.0
    assert 0.0 <= b.y <= 1.0
    assert b.x + b.width <= 1.0 + 1e-9
    assert b.y + b.height <= 1.0 + 1e-9


def test_effect_defaults_are_sane():
    e = Effect(type="circle")
    assert e.start_seconds == 0.0
    assert e.end_seconds == 0.0
    assert e.target is None


def test_clip_to_output_summarizes_effect_types_deduplicated_and_sorted():
    clip = Clip(
        effects=[
            Effect(type="arrow", start_seconds=0, end_seconds=1),
            Effect(type="circle", start_seconds=1, end_seconds=2),
            Effect(type="circle", start_seconds=3, end_seconds=4),
        ]
    )
    out = clip.to_output(None)
    assert out["effects"] == ["arrow", "circle"]


def test_clip_to_output_effects_empty_when_no_effects():
    clip = Clip()
    assert clip.to_output(None)["effects"] == []


def test_target_defaults():
    t = Target()
    assert t.description == ""
    assert t.bbox is None
    assert t.confidence == 0.0
