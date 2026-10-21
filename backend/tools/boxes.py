"""Box geometry: intersection and IoU."""
from tools.types import Box, Detection
from collections import defaultdict


def intersection_area(a: Box, b: Box) -> float:
    w = min(a.x2, b.x2) - max(a.x1, b.x1)   # overlap along x
    h = min(a.y2, b.y2) - max(a.y1, b.y1)   # overlap along y
    if w <= 0 or h <= 0:                    # boxes don't overlap
        return 0.0
    return w * h


def iou(a: Box, b: Box) -> float:
    inter = intersection_area(a, b)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0

def nms(dets: list[Detection], iou_thr: float = 0.5) -> list[Detection]:
    """Greedy NMS: keep the best box, drop boxes overlapping it more than iou_thr."""
    kept: list[Detection] = []
    for d in sorted(dets, key=lambda d: d.score, reverse=True):
        if all(iou(d.box, k.box) <= iou_thr for k in kept):
            kept.append(d)
    return kept

def nms_per_class(dets: list[Detection], iou_thr: float = 0.5) -> list[Detection]:
    """Run NMS separately inside each class, so different classes never suppress each other."""
    groups: dict[str, list[Detection]] = defaultdict(list)
    for d in dets:
        groups[d.cls].append(d)
    out: list[Detection] = []
    for group in groups.values():
        out.extend(nms(group, iou_thr))
    return sorted(out, key=lambda d: d.score, reverse=True)


def expand_box(box: Box, factor: float, img_w: float, img_h: float, min_size: float = 160.0) -> Box:
    """Crop region around `box`: scaled by `factor`, at least `min_size` wide and tall,
    clipped to the image."""
    cx, cy = box.center
    w = max(box.width * factor, min_size)
    h = max(box.height * factor, min_size)
    return Box(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2).clip(img_w, img_h)

def union_box(a: Box, b: Box) -> Box:
    return Box(min(a.x1, b.x1), min(a.y1, b.y1), max(a.x2, b.x2), max(a.y2, b.y2))