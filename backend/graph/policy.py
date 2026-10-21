"""The accept / ask-a-human / discard decision, in one place."""
from __future__ import annotations

from tools.types import Detection


def triage(
    boxes: list[Detection], accept_above: float, drop_below: float, min_height: float = 0.0
) -> tuple[list[Detection], list[Detection], list[Detection]]:
    """Split boxes into (accepted, for_human, dropped).

    A box shorter than `min_height` pixels is dropped whatever its confidence: it is too
    small to label reliably (KITTI itself ignores objects under 25 px)."""
    accepted, for_human, dropped = [], [], []
    for d in boxes:
        if d.score < drop_below or d.box.height < min_height:
            dropped.append(d)
        elif d.score >= accept_above:
            accepted.append(d)
        else:
            for_human.append(d)
    return accepted, for_human, dropped