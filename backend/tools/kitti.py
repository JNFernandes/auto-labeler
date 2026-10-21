"""Reading KITTI object-detection data."""
from pathlib import Path

from tools.types import Box, GTObject

from PIL import Image

import random

import warnings

CLASSES = ("Car", "Pedestrian", "Cyclist")
DIFFICULTIES = ("easy", "moderate", "hard")
MIN_HEIGHT = {"easy": 40, "moderate": 25, "hard": 25}
MAX_OCCLUSION = {"easy": 0, "moderate": 1, "hard": 2}
MAX_TRUNCATION = {"easy": 0.15, "moderate": 0.30, "hard": 0.50}

def parse_label_file(path: Path) -> list[GTObject]:
    """Parse one KITTI label file into ground-truth objects. Malformed lines are skipped."""
    objects: list[GTObject] = []
    text = Path(path).read_text(encoding="utf-8")
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 8:
            continue
        try:
            objects.append(
                GTObject(
                    cls=parts[0],
                    box=Box(float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])),
                    truncated=float(parts[1]),
                    occluded=int(float(parts[2])),
                )
            )
        except ValueError:
            continue
    return objects

def meets_difficulty(gt: GTObject, difficulty: str) -> bool:
    """True if this object counts at the given KITTI difficulty level."""
    return (
        gt.box.height >= MIN_HEIGHT[difficulty]
        and gt.occluded <= MAX_OCCLUSION[difficulty]
        and gt.truncated <= MAX_TRUNCATION[difficulty]
    )


class KittiDataset:
    """The KITTI training set. `root` can be the folder that contains `training/`
    or the `training/` folder itself."""

    def __init__(self, root, ids: list[str] | None = None):
        root = Path(root)
        if (root / "training").is_dir():
            self.root, train_dir = root, root / "training"
        else:                                  # we were given the training/ folder itself
            self.root, train_dir = root.parent, root
        self.image_dir = train_dir / "image_2"
        self.label_dir = train_dir / "label_2"
        if ids is None:
            ids = sorted(p.stem for p in self.label_dir.glob("*.txt"))
        self.ids = list(ids)

    def __len__(self) -> int:
        return len(self.ids)

    def image_path(self, image_id: str) -> Path:
        return self.image_dir / f"{image_id}.png"

    def load_image(self, image_id: str) -> Image.Image:
        return Image.open(self.image_path(image_id)).convert("RGB")

    def gts(self, image_id: str) -> list[GTObject]:
        return parse_label_file(self.label_dir / f"{image_id}.txt")


def load_split(root, name: str) -> list[str] | None:
    """Read ImageSets/<name>.txt (one image id per line). None if the file is missing."""
    path = Path(root) / "ImageSets" / f"{name}.txt"
    if not path.is_file():
        return None
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def fallback_split(ids: list[str], val_fraction: float = 0.5, seed: int = 0) -> tuple[list[str], list[str]]:
    """Random split, used only when the official files are missing."""
    warnings.warn(
        "ImageSets/train.txt and val.txt not found; using a random split. "
        "Frames from the same drive can land on both sides, which inflates validation results.",
        UserWarning,
        stacklevel=2,
    )
    shuffled = list(ids)
    random.Random(seed).shuffle(shuffled)
    n_val = round(len(shuffled) * val_fraction)
    return sorted(shuffled[n_val:]), sorted(shuffled[:n_val])


def get_splits(dataset: KittiDataset) -> tuple[list[str], list[str]]:
    """(train_ids, val_ids): the official split if available, else a warned random one."""
    available = set(dataset.ids)
    train = load_split(dataset.root, "train")
    val = load_split(dataset.root, "val")
    if train is None or val is None:
        return fallback_split(dataset.ids)
    return [i for i in train if i in available], [i for i in val if i in available]