"""Draw an agent's decisions on an image."""
from __future__ import annotations

from PIL import Image, ImageDraw

from tools.types import Detection, GTObject

COLORS = {"accepted": "lime", "human": "orange", "dropped": "red", "truth": "deepskyblue"}
SHOWN_TRUTH = ("Car", "Pedestrian", "Cyclist")


def draw_result(
    image: Image.Image,
    accepted: list[Detection],
    for_human: list[Detection],
    dropped: list[Detection],
    truth: list[GTObject] | None = None,
) -> Image.Image:
    """Return a copy of `image` with boxes: green = auto-accepted, orange = for a human,
    red = dropped, thin blue = KITTI ground truth (only when `truth` is given)."""
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    for g in truth or []:
        if g.cls in SHOWN_TRUTH:
            b = g.box
            draw.rectangle([b.x1, b.y1, b.x2, b.y2], outline=COLORS["truth"], width=1)
    for group, key, width in ((dropped, "dropped", 1), (for_human, "human", 2), (accepted, "accepted", 3)):
        for d in group:
            b = d.box
            draw.rectangle([b.x1, b.y1, b.x2, b.y2], outline=COLORS[key], width=width)
            draw.text((b.x1 + 2, max(b.y1 - 11, 0)), f"{d.cls} {d.score:.2f}", fill=COLORS[key])
    return canvas