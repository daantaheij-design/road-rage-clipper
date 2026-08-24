"""Generate small transparent PNG overlays (red circle / red arrow) for
visual-attention effects, composited onto the render via ffmpeg's `overlay`
filter with a start/end `enable` window.

Deliberately not done in pure ffmpeg filters: drawing a clean arrowhead
with ffmpeg's `drawbox`/`geq` primitives is fragile and hard to get looking
right on a phone screen, whereas Pillow gives real anti-aliased vector
drawing in a few lines. Each overlay is a single static full-canvas
(1080x1920) RGBA image - not a per-frame sequence - so this stays cheap on
a small Railway container: at most a handful of tiny PNGs per clip (0-3
effects), each one ffmpeg `-i` input away from a no-op until its `enable`
window is active.
"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw

from app.pipeline.geometry import OutputRect, choose_arrow_geometry

CANVAS_WIDTH = 1080
CANVAS_HEIGHT = 1920

RED = (255, 32, 32, 255)
RED_SOFT = (255, 32, 32, 90)

CIRCLE_OUTLINE_WIDTH = 10
CIRCLE_PADDING_PX = 18  # circle drawn a bit larger than the bbox itself

ARROW_SHAFT_WIDTH = 14
ARROW_HEAD_LENGTH = 46.0
ARROW_HEAD_WIDTH = 34.0


def _blank_canvas() -> Image.Image:
    return Image.new("RGBA", (CANVAS_WIDTH, CANVAS_HEIGHT), (0, 0, 0, 0))


def render_circle_overlay(target: OutputRect, out_path: Path) -> Path:
    """A bright red ellipse traced around `target`, thick outline,
    transparent center - never a solid block covering the subject."""
    img = _blank_canvas()
    draw = ImageDraw.Draw(img)

    cx, cy = target.center
    rx = target.width / 2 + CIRCLE_PADDING_PX
    ry = target.height / 2 + CIRCLE_PADDING_PX
    bbox = (cx - rx, cy - ry, cx + rx, cy + ry)

    # A faint soft halo behind the crisp outline reads better on busy
    # footage than a single thin line, without becoming a solid block.
    draw.ellipse(bbox, outline=RED_SOFT, width=CIRCLE_OUTLINE_WIDTH + 8)
    draw.ellipse(bbox, outline=RED, width=CIRCLE_OUTLINE_WIDTH)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, "PNG")
    return out_path


def render_arrow_overlay(target: OutputRect, out_path: Path) -> Path:
    """A thick red arrow whose tip points at `target`'s nearest edge,
    originating from open space chosen by choose_arrow_geometry so it never
    covers the target itself."""
    img = _blank_canvas()
    draw = ImageDraw.Draw(img)

    (tail_x, tail_y), (head_x, head_y) = choose_arrow_geometry(target, CANVAS_WIDTH, CANVAS_HEIGHT)
    draw.line((tail_x, tail_y, head_x, head_y), fill=RED, width=ARROW_SHAFT_WIDTH)

    # Arrowhead: a filled triangle at the head point, oriented along the
    # tail->head direction.
    dx, dy = head_x - tail_x, head_y - tail_y
    length = math.hypot(dx, dy) or 1.0
    ux, uy = dx / length, dy / length  # unit vector along the shaft
    px, py = -uy, ux  # perpendicular unit vector

    base_x = head_x - ux * ARROW_HEAD_LENGTH
    base_y = head_y - uy * ARROW_HEAD_LENGTH
    left = (base_x + px * ARROW_HEAD_WIDTH / 2, base_y + py * ARROW_HEAD_WIDTH / 2)
    right = (base_x - px * ARROW_HEAD_WIDTH / 2, base_y - py * ARROW_HEAD_WIDTH / 2)
    draw.polygon([(head_x, head_y), left, right], fill=RED)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, "PNG")
    return out_path
