from __future__ import annotations

from PIL import Image

from app.pipeline.geometry import OutputRect
from app.pipeline.overlays import CANVAS_HEIGHT, CANVAS_WIDTH, render_arrow_overlay, render_circle_overlay


def _count_reddish(img: Image.Image) -> int:
    count = 0
    for r, g, b, a in img.getdata():
        if a > 0 and r > 150 and g < 120 and b < 120:
            count += 1
    return count


def test_render_circle_overlay_is_full_canvas_transparent_png(tmp_path):
    out = tmp_path / "circle.png"
    target = OutputRect(x=400, y=800, width=200, height=150)
    render_circle_overlay(target, out)

    img = Image.open(out)
    assert img.size == (CANVAS_WIDTH, CANVAS_HEIGHT)
    assert img.mode == "RGBA"


def test_render_circle_overlay_draws_visible_red_pixels_near_target(tmp_path):
    out = tmp_path / "circle.png"
    target = OutputRect(x=400, y=800, width=200, height=150)
    render_circle_overlay(target, out)
    img = Image.open(out)
    assert _count_reddish(img) > 0


def test_render_circle_overlay_center_stays_transparent(tmp_path):
    # A proper ring, not a solid block: the exact center of the target
    # should not be painted over.
    out = tmp_path / "circle.png"
    target = OutputRect(x=400, y=800, width=200, height=150)
    render_circle_overlay(target, out)
    img = Image.open(out)
    cx, cy = int(target.x + target.width / 2), int(target.y + target.height / 2)
    assert img.getpixel((cx, cy))[3] == 0  # fully transparent alpha


def test_render_arrow_overlay_draws_visible_red_pixels(tmp_path):
    out = tmp_path / "arrow.png"
    target = OutputRect(x=900, y=900, width=100, height=100)
    render_arrow_overlay(target, out)
    img = Image.open(out)
    assert img.size == (CANVAS_WIDTH, CANVAS_HEIGHT)
    assert _count_reddish(img) > 0


def test_render_arrow_overlay_does_not_paint_over_target_center(tmp_path):
    out = tmp_path / "arrow.png"
    target = OutputRect(x=400, y=900, width=150, height=150)
    render_arrow_overlay(target, out)
    img = Image.open(out)
    cx, cy = int(target.x + target.width / 2), int(target.y + target.height / 2)
    assert img.getpixel((cx, cy))[3] == 0


def test_overlays_are_independent_files_not_overwritten(tmp_path):
    target = OutputRect(x=100, y=100, width=80, height=80)
    circle_path = tmp_path / "c.png"
    arrow_path = tmp_path / "a.png"
    render_circle_overlay(target, circle_path)
    render_arrow_overlay(target, arrow_path)
    assert circle_path.read_bytes() != arrow_path.read_bytes()
