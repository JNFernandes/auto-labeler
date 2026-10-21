"""Zoom retry: look at an uncertain box again, from closer."""
from __future__ import annotations

from PIL import Image

from tools.boxes import expand_box, iou
from tools.cyclists import merge_cyclists
from tools.types import Detection


def zoom_retry(
    image: Image.Image,
    det: Detection,
    proposer,
    verifier,
    factor: float = 3.0,
    min_overlap: float = 0.3,
) -> tuple[Detection, float] | None:
    """Run the proposer again on an enlarged window around `det`.

    Returns (refined detection, its CLIP score), or None when the zoomed view finds no
    box of the same class that overlaps the original one."""
    region = expand_box(det.box, factor, image.width, image.height)
    found = merge_cyclists(proposer.propose(image, region=region))
    same = [d for d in found if d.cls == det.cls and iou(d.box, det.box) >= min_overlap]
    if not same:
        return None
    best = max(same, key=lambda d: iou(d.box, det.box))
    return best, verifier.verify(image, best)