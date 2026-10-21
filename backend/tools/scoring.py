"""How the two model scores become one confidence."""

DINO_WEIGHT = 0.7


def fuse(dino_score: float, clip_score: float) -> float:
    """conf = 0.7 * dino + 0.3 * clip (the best rule from the fusion experiment)."""
    return DINO_WEIGHT * dino_score + (1.0 - DINO_WEIGHT) * clip_score