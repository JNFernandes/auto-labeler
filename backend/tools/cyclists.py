"""Turn raw model labels (car / person / bicycle) into KITTI classes, merging riders with bikes."""
from __future__ import annotations

from tools.boxes import intersection_area, union_box
from tools.types import Detection

MIN_COVER = 0.20     # fraction of the bicycle box the person must overlap
CENTER_SLACK = 0.25  # how far outside the bike the person's centre may sit (fraction of bike width)


def merge_cyclists(dets: list[Detection]) -> list[Detection]:
    bikes = sorted((d for d in dets if d.cls == "bicycle"), key=lambda d: -d.score)
    persons = [d for d in dets if d.cls == "person"]
    used: set[int] = set()
    out: list[Detection] = []

    for bike in bikes:
        best, best_cover = None, 0.0
        for i, p in enumerate(persons):
            if i in used or bike.box.area <= 0:
                continue
            cover = intersection_area(bike.box, p.box) / bike.box.area
            cx = p.box.center[0]
            slack = CENTER_SLACK * bike.box.width
            if cover >= MIN_COVER and bike.box.x1 - slack <= cx <= bike.box.x2 + slack and cover > best_cover:
                best, best_cover = i, cover
        if best is not None:
            used.add(best)
            rider = persons[best]
            out.append(Detection("Cyclist", union_box(bike.box, rider.box),
                                 min(bike.score, rider.score), "merged"))

    out += [Detection("Pedestrian", p.box, p.score, p.source) for i, p in enumerate(persons) if i not in used]
    out += [Detection("Car", d.box, d.score, d.source) for d in dets if d.cls == "car"]
    return out