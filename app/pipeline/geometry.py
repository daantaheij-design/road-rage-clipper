"""Pure coordinate-transform math for visual-attention effects (circle,
arrow, punch-zoom): turning a bounding box Claude reported in SOURCE frame
coordinates into a pixel rectangle in the OUTPUT (1080x1920, already
cropped+scaled) frame at a given moment.

The source footage is horizontal; the rendered clip is a smart-cropped,
panned 9:16 slice of it (see app.pipeline.crop). A bbox normalized against
the full source frame can't be drawn at the same normalized position in the
output - it has to be re-expressed relative to whatever the crop window
happens to be at that instant. That's what `transform_bbox_to_output` does;
everything here is plain arithmetic, no ffmpeg/Pillow involved, so it's
cheap to unit test exhaustively.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.jobs.models import BBox

# Below this fraction of the bbox's own area still visible inside the
# current crop window, the target is considered "not usefully on screen" -
# better to skip the annotation than draw a circle mostly off-canvas.
MIN_VISIBLE_AREA_FRACTION = 0.35

ARROW_LENGTH_PX = 220.0
ARROW_EDGE_GAP_PX = 10.0
ARROW_MIN_MARGIN_PX = 40.0


@dataclass
class OutputRect:
    """A rectangle in OUTPUT pixel space (0,0 = top-left of the rendered
    1080x1920 frame)."""

    x: float
    y: float
    width: float
    height: float

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.width / 2, self.y + self.height / 2


def bbox_to_source_pixels(bbox: BBox, source_width: int, source_height: int) -> tuple[float, float, float, float]:
    """Normalized (0-1) source-frame bbox -> absolute source pixels."""
    b = bbox.clamped()
    return b.x * source_width, b.y * source_height, b.width * source_width, b.height * source_height


def transform_bbox_to_output(
    bbox: BBox,
    *,
    source_width: int,
    source_height: int,
    crop_x: float,
    crop_y: float,
    crop_w: float,
    crop_h: float,
    target_width: int,
    target_height: int,
) -> OutputRect | None:
    """Map a source-frame bbox into output pixel space given the crop
    window (source pixels) active at the moment this annotation should
    appear. Returns None if the target isn't usefully visible in that crop
    window (mostly panned/zoomed out of frame) - callers should skip
    drawing an annotation in that case rather than place it off-screen."""
    if crop_w <= 0 or crop_h <= 0 or source_width <= 0 or source_height <= 0:
        return None

    bx, by, bw, bh = bbox_to_source_pixels(bbox, source_width, source_height)
    if bw <= 0 or bh <= 0:
        return None

    ix0, iy0 = max(bx, crop_x), max(by, crop_y)
    ix1, iy1 = min(bx + bw, crop_x + crop_w), min(by + bh, crop_y + crop_h)
    if ix1 <= ix0 or iy1 <= iy0:
        return None

    visible_fraction = ((ix1 - ix0) * (iy1 - iy0)) / (bw * bh)
    if visible_fraction < MIN_VISIBLE_AREA_FRACTION:
        return None

    scale_x = target_width / crop_w
    scale_y = target_height / crop_h
    return OutputRect(
        x=(bx - crop_x) * scale_x,
        y=(by - crop_y) * scale_y,
        width=bw * scale_x,
        height=bh * scale_y,
    )


def choose_arrow_geometry(
    target: OutputRect, canvas_width: int, canvas_height: int
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Pick a sensible tail->head line for an arrow pointing at `target`,
    approaching from whichever side of the frame has the most open room
    (mirrors the spec's own examples: near the right edge, originate
    further left and point right; near the bottom, originate above and
    point down). The head always lands on the target's NEAREST edge to
    that open space - never the far edge - so the shaft approaches and
    stops at the target instead of running straight through it. The tail
    sits further out in that same open space, clamped to stay on screen."""
    cx, cy = target.center
    space_left, space_right = target.x, canvas_width - (target.x + target.width)
    space_top, space_bottom = target.y, canvas_height - (target.y + target.height)

    max_horizontal_space = max(space_left, space_right)
    max_vertical_space = max(space_top, space_bottom)

    if max_vertical_space >= max_horizontal_space:
        if space_bottom >= space_top:
            # More open room below - approach from below, point up into the
            # target's bottom edge (the near edge from that direction).
            head = (cx, min(float(canvas_height), target.y + target.height + ARROW_EDGE_GAP_PX))
            tail_y = min(canvas_height - ARROW_MIN_MARGIN_PX, head[1] + ARROW_LENGTH_PX)
            tail = (cx, tail_y)
        else:
            # More open room above - approach from above, point down into
            # the target's top edge.
            head = (cx, max(0.0, target.y - ARROW_EDGE_GAP_PX))
            tail_y = max(ARROW_MIN_MARGIN_PX, head[1] - ARROW_LENGTH_PX)
            tail = (cx, tail_y)
    else:
        if space_right >= space_left:
            # More open room to the right - approach from the right, point
            # left into the target's right edge.
            head = (min(float(canvas_width), target.x + target.width + ARROW_EDGE_GAP_PX), cy)
            tail_x = min(canvas_width - ARROW_MIN_MARGIN_PX, head[0] + ARROW_LENGTH_PX)
            tail = (tail_x, cy)
        else:
            # More open room to the left - approach from the left, point
            # right into the target's left edge.
            head = (max(0.0, target.x - ARROW_EDGE_GAP_PX), cy)
            tail_x = max(ARROW_MIN_MARGIN_PX, head[0] - ARROW_LENGTH_PX)
            tail = (tail_x, cy)

    return tail, head
