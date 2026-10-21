"""Shared data types."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Box:
    """Axis-aligned box in pixels: (x1, y1) top-left, (x2, y2) bottom-right."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def clip(self, width: float, height: float) -> "Box":
        """Force the box inside an image of the given size."""
        return Box(
            min(max(self.x1, 0.0), width),
            min(max(self.y1, 0.0), height),
            min(max(self.x2, 0.0), width),
            min(max(self.y2, 0.0), height),
        )

@dataclass(frozen=True)
class Detection:
    """A predicted box with a class label and a confidence score."""

    cls: str
    box: Box
    score: float
    source: str = ""     # where it came from; used later (e.g. "zero-shot", "human")

@dataclass(frozen=True)
class GTObject:
    """One ground-truth annotation from KITTI. `cls` is the raw KITTI type string
    (Car, Van, Truck, Pedestrian, Person_sitting, Cyclist, Tram, Misc, DontCare)."""

    cls: str
    box: Box
    truncated: float = 0.0
    occluded: int = 0

