"""Detection evaluation: matching detections to ground truth, and Average Precision."""
import math

import numpy as np

from tools.boxes import intersection_area, iou
from tools.kitti import CLASSES, DIFFICULTIES, MIN_HEIGHT, meets_difficulty
from tools.types import Box, Detection, GTObject

IOU_THRESHOLD = {"Car": 0.7, "Pedestrian": 0.5, "Cyclist": 0.5}
NEIGHBOUR = {"Car": "Van", "Pedestrian": "Person_sitting", "Cyclist": None}

def match_image(dets: list[Detection], gts: list[Box], iou_thr: float) -> list[tuple[float, int]]:
    """Match one image's detections to its ground-truth boxes.

    Returns (score, 1 if true positive else 0) for every detection, best score first.
    Each ground-truth box can be claimed only once."""
    taken: set[int] = set()
    out: list[tuple[float, int]] = []
    for det in sorted(dets, key=lambda d: -d.score):
        free = [(iou(det.box, g), i) for i, g in enumerate(gts) if i not in taken]
        best = max(free, default=None)          # highest-IoU unclaimed ground truth
        if best is not None and best[0] >= iou_thr:
            taken.add(best[1])
            out.append((det.score, 1))
        else:
            out.append((det.score, 0))
    return out


def average_precision(scores: np.ndarray, is_tp: np.ndarray, n_pos: int) -> float:
    """AP (0 to 100) with 40 interpolated recall positions. NaN when there are no real objects."""
    if n_pos == 0:
        return float("nan")
    if len(scores) == 0:
        return 0.0
    order = np.argsort(-scores, kind="stable")           # best score first
    tp = np.cumsum(is_tp[order])
    fp = np.cumsum(1 - is_tp[order])
    recall = tp / n_pos
    precision = tp / np.maximum(tp + fp, 1)
    envelope = np.maximum.accumulate(precision[::-1])[::-1]   # best precision at this recall or higher
    total = 0.0
    for r in np.linspace(0, 1, 41)[1:]:                  # 1/40, 2/40, ... 40/40
        idx = int(np.searchsorted(recall, r - 1e-9, side="left"))   # first point reaching recall r
        total += envelope[idx] if idx < len(envelope) else 0.0      # never reached -> 0
    return 100.0 * total / 40.0


def _gt_status(gt: GTObject, cls: str, difficulty: str) -> int:
    """0 = valid (must be found), 1 = ignore (no penalty either way), -1 = irrelevant."""
    if gt.cls == cls:
        return 0 if meets_difficulty(gt, difficulty) else 1
    if NEIGHBOUR[cls] and gt.cls == NEIGHBOUR[cls]:
        return 1
    return -1


def _eval_image(
    dets: list[Detection], gts: list[GTObject], cls: str, difficulty: str
) -> tuple[list[tuple[float, int]], int]:
    """KITTI-style matching for one image, one class, one difficulty level.

    Returns ((score, 1 if TP else 0) for each detection that counts, number of valid objects)."""
    thr = IOU_THRESHOLD[cls]
    status = [_gt_status(g, cls, difficulty) for g in gts]
    valid = [i for i, s in enumerate(status) if s == 0]
    ignored = [i for i, s in enumerate(status) if s == 1]
    dontcare = [g.box for g in gts if g.cls == "DontCare"]

    taken: set[int] = set()
    out: list[tuple[float, int]] = []
    for det in sorted((d for d in dets if d.cls == cls), key=lambda d: -d.score):
        free = [(iou(det.box, gts[i].box), i) for i in valid if i not in taken]
        best = max(free, default=None)
        if best is not None and best[0] >= thr:                 # rule 3: true positive
            taken.add(best[1])
            out.append((det.score, 1))
            continue
        if any(iou(det.box, gts[i].box) >= thr for i in ignored):   # rule 2: sits on an ignore object
            continue
        if det.box.height < MIN_HEIGHT[difficulty]:                 # rule 2: too small to matter
            continue
        if det.box.area > 0 and any(intersection_area(det.box, b) / det.box.area > 0.5 for b in dontcare):
            continue                                                # rule 2: inside a DontCare zone
        out.append((det.score, 0))                                  # rule 3: false positive
    return out, len(valid)

def evaluate(
    detections: dict[str, list[Detection]],
    gts: dict[str, list[GTObject]],
    classes: tuple[str, ...] = CLASSES,
) -> dict[str, dict[str, float]]:
    """AP for every class and difficulty: {class: {difficulty: AP}}.

    `detections` and `gts` are keyed by image id. Only image ids present in `gts` are scored."""
    results: dict[str, dict[str, float]] = {}
    for cls in classes:
        results[cls] = {}
        for difficulty in DIFFICULTIES:
            scores: list[float] = []
            flags: list[int] = []
            n_pos = 0
            for image_id, gt_list in gts.items():
                rows, n_valid = _eval_image(detections.get(image_id, []), gt_list, cls, difficulty)
                n_pos += n_valid
                for score, flag in rows:
                    scores.append(score)
                    flags.append(flag)
            results[cls][difficulty] = average_precision(
                np.array(scores, dtype=np.float64), np.array(flags, dtype=np.int64), n_pos
            )
    return results


def mean_ap(result: dict[str, dict[str, float]], difficulty: str = "moderate") -> float:
    """Mean AP over classes at one level, skipping classes that have no objects (NaN)."""
    vals = [result[c][difficulty] for c in result if not math.isnan(result[c][difficulty])]
    return float(np.mean(vals)) if vals else float("nan")


def format_table(title: str, result: dict[str, dict[str, float]]) -> str:
    def cell(v: float) -> str:
        return "  n/a" if math.isnan(v) else f"{v:5.1f}"

    lines = [title, f"  {'class':<12}{'easy':>7}{'moderate':>10}{'hard':>7}"]
    for cls, row in result.items():
        lines.append(f"  {cls:<12}{cell(row['easy']):>7}{cell(row['moderate']):>10}{cell(row['hard']):>7}")
    lines.append(f"  {'mean (mod)':<12}{'':>7}{cell(mean_ap(result)):>10}")
    return "\n".join(lines)