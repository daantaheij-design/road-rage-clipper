from __future__ import annotations

from app.jobs.models import BBox
from app.pipeline.geometry import OutputRect, choose_arrow_geometry, transform_bbox_to_output


def test_transform_bbox_to_output_maps_full_frame_crop_correctly():
    # Crop window == full source frame, source == target aspect (already
    # 9:16) - the transform should just be a linear rescale.
    bbox = BBox(x=0.5, y=0.5, width=0.1, height=0.1)
    rect = transform_bbox_to_output(
        bbox,
        source_width=1080,
        source_height=1920,
        crop_x=0,
        crop_y=0,
        crop_w=1080,
        crop_h=1920,
        target_width=1080,
        target_height=1920,
    )
    assert rect is not None
    assert rect.x == 540  # bbox.x is the top-left corner, not center: 0.5*1080
    assert rect.width == 108


def test_transform_bbox_to_output_returns_none_when_outside_crop_window():
    bbox = BBox(x=0.05, y=0.05, width=0.05, height=0.05)  # near top-left
    rect = transform_bbox_to_output(
        bbox,
        source_width=1920,
        source_height=1080,
        crop_x=1000,  # crop window is over on the right side
        crop_y=0,
        crop_w=600,
        crop_h=1080,
        target_width=1080,
        target_height=1920,
    )
    assert rect is None


def test_transform_bbox_to_output_returns_none_when_mostly_cut_off():
    # Bbox is 90% outside the crop window - only a sliver overlaps.
    bbox = BBox(x=0.0, y=0.4, width=0.1, height=0.2)
    rect = transform_bbox_to_output(
        bbox,
        source_width=1000,
        source_height=1000,
        crop_x=90,  # crop starts at x=90, bbox spans x=[0,100] - only 10% visible
        crop_y=0,
        crop_w=500,
        crop_h=1000,
        target_width=500,
        target_height=1000,
    )
    assert rect is None


def test_transform_bbox_to_output_keeps_target_fully_inside_crop_bounds():
    # crop window spans source x=[500, 1108]; bbox spans x=[576, 729.6] -
    # fully inside the crop window.
    bbox = BBox(x=0.3, y=0.4, width=0.08, height=0.15)
    rect = transform_bbox_to_output(
        bbox,
        source_width=1920,
        source_height=1080,
        crop_x=500,
        crop_y=0,
        crop_w=608,
        crop_h=1080,
        target_width=1080,
        target_height=1920,
    )
    assert rect is not None
    assert 0 <= rect.x <= 1080
    assert 0 <= rect.y <= 1920
    assert rect.x + rect.width <= 1080 + 1  # allow tiny float slack
    assert rect.y + rect.height <= 1920 + 1


def test_transform_bbox_to_output_handles_degenerate_crop_window():
    bbox = BBox(x=0.5, y=0.5, width=0.1, height=0.1)
    rect = transform_bbox_to_output(
        bbox, source_width=1920, source_height=1080, crop_x=0, crop_y=0, crop_w=0, crop_h=0, target_width=1080, target_height=1920
    )
    assert rect is None


def test_choose_arrow_geometry_points_at_target_near_right_edge():
    target = OutputRect(x=950, y=900, width=100, height=100)  # near right edge
    tail, head = choose_arrow_geometry(target, canvas_width=1080, canvas_height=1920)
    # Per spec: near the right edge, arrow originates further left and points right.
    assert tail[0] < head[0]
    assert head[0] <= target.x + target.width + 20  # head is just outside/at the bbox edge


def test_choose_arrow_geometry_points_at_target_near_bottom_edge():
    target = OutputRect(x=450, y=1800, width=100, height=80)  # near bottom edge
    tail, head = choose_arrow_geometry(target, canvas_width=1080, canvas_height=1920)
    # Per spec: near the bottom, originate above and point down.
    assert tail[1] < head[1]


def test_choose_arrow_geometry_never_points_into_the_target_bbox():
    target = OutputRect(x=400, y=800, width=150, height=150)
    tail, head = choose_arrow_geometry(target, canvas_width=1080, canvas_height=1920)
    # Head should sit outside the bbox, not inside it.
    inside = (target.x <= head[0] <= target.x + target.width) and (target.y <= head[1] <= target.y + target.height)
    assert not inside


def test_choose_arrow_geometry_stays_within_canvas_bounds():
    target = OutputRect(x=20, y=20, width=60, height=60)  # near top-left corner
    tail, head = choose_arrow_geometry(target, canvas_width=1080, canvas_height=1920)
    for x, y in (tail, head):
        assert -1 <= x <= 1081
        assert -1 <= y <= 1921


def test_output_rect_center():
    rect = OutputRect(x=100, y=200, width=50, height=60)
    assert rect.center == (125, 230)


def test_choose_arrow_geometry_shaft_never_crosses_through_the_bbox():
    # Regression: an earlier version pointed the head at the target's FAR
    # edge, so a tail placed on the near side by ARROW_LENGTH_PX could
    # overshoot past the target and the shaft would cut straight through
    # the bbox (visually covering the subject the arrow is meant to point
    # at). For a horizontal shaft (constant y), both tail_x and head_x must
    # stay on the same side of the bbox's [x, x+w] span - never straddling
    # it - and likewise for a vertical shaft against [y, y+h].
    for x, y, w, h in [(400, 900, 150, 150), (100, 100, 80, 80), (900, 900, 100, 100), (500, 50, 100, 100), (500, 1800, 100, 100)]:
        target = OutputRect(x=x, y=y, width=w, height=h)
        (tail_x, tail_y), (head_x, head_y) = choose_arrow_geometry(target, canvas_width=1080, canvas_height=1920)
        if tail_y == head_y:  # horizontal shaft
            both_left = tail_x <= x and head_x <= x
            both_right = tail_x >= x + w and head_x >= x + w
            assert both_left or both_right, f"horizontal shaft straddles bbox: tail={tail_x} head={head_x} bbox_x=[{x},{x + w}]"
        else:  # vertical shaft
            both_above = tail_y <= y and head_y <= y
            both_below = tail_y >= y + h and head_y >= y + h
            assert both_above or both_below, f"vertical shaft straddles bbox: tail={tail_y} head={head_y} bbox_y=[{y},{y + h}]"
