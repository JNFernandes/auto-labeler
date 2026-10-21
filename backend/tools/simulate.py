"""Replay the policy on cached boxes with a simulated reviewer (reads ground truth: allowed here)."""
from __future__ import annotations

from tools.boxes import iou
from tools.evaluate import IOU_THRESHOLD, evaluate, mean_ap
from graph.policy import triage
from tools.types import Detection, GTObject


def is_true(det: Detection, gts: list[GTObject]) -> bool:
    """Does this box match a real object of its class? (the simulated reviewer's answer)"""
    return any(g.cls == det.cls and iou(det.box, g.box) >= IOU_THRESHOLD[det.cls] for g in gts)


def simulate_policy(
    boxes_by_image: dict[str, list[Detection]],
    gts: dict[str, list[GTObject]],
    accept_above: float,
    drop_below: float,
    min_height: float = 0.0,
) -> dict[str, float]:
    auto, final = {}, {}
    n_auto = n_ok = n_queue = 0
    for image_id, boxes in boxes_by_image.items():
        accepted, queued, _ = triage(boxes, accept_above, drop_below, min_height)
        approved = [d for d in queued if is_true(d, gts[image_id])]
        auto[image_id], final[image_id] = accepted, accepted + approved
        n_auto += len(accepted)
        n_ok += sum(is_true(d, gts[image_id]) for d in accepted)
        n_queue += len(queued)
    return {
        "auto": n_auto,
        "auto_precision": 100.0 * n_ok / max(n_auto, 1),
        "review_per_image": n_queue / max(len(boxes_by_image), 1),
        "ap_auto": mean_ap(evaluate(auto, gts)),
        "ap_final": mean_ap(evaluate(final, gts)),
    }