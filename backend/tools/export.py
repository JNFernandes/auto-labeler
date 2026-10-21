"""Write the agent's labels in the formats training tools read: YOLO text files and one COCO json."""
import json
from pathlib import Path

from tools.types import Detection

CLASSES = ["Car", "Pedestrian", "Cyclist"]


def yolo_lines(dets: list[Detection], width: int, height: int) -> list[str]:
    """One line per box: class_id x_center y_center width height, all divided by the image size."""
    lines = []
    for d in dets:
        b = d.box
        lines.append(f"{CLASSES.index(d.cls)} {(b.x1 + b.x2) / 2 / width:.6f} {(b.y1 + b.y2) / 2 / height:.6f} "
                     f"{b.width / width:.6f} {b.height / height:.6f}")
    return lines


def save_yolo(folder: Path, name: str, dets: list[Detection], width: int, height: int) -> None:
    labels = folder / "yolo"
    labels.mkdir(parents=True, exist_ok=True)
    (labels / f"{name}.txt").write_text("\n".join(yolo_lines(dets, width, height)), encoding="utf-8")
    (labels / "classes.txt").write_text("\n".join(CLASSES), encoding="utf-8")


def coco_dict(items: list[tuple[str, int, int, list[Detection]]]) -> dict:
    """items = [(image name, width, height, boxes)]. Boxes are [x, y, width, height] in pixels."""
    images, annotations = [], []
    for image_id, (name, width, height, dets) in enumerate(items, 1):
        images.append({"id": image_id, "file_name": name, "width": width, "height": height})
        for d in dets:
            b = d.box
            annotations.append({"id": len(annotations) + 1, "image_id": image_id,
                                "category_id": CLASSES.index(d.cls) + 1,
                                "bbox": [round(b.x1, 1), round(b.y1, 1), round(b.width, 1), round(b.height, 1)],
                                "area": round(b.area, 1), "iscrowd": 0, "score": round(d.score, 3)})
    return {"images": images, "annotations": annotations,
            "categories": [{"id": i + 1, "name": c} for i, c in enumerate(CLASSES)]}


def save_coco(path: Path, items) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(coco_dict(items)), encoding="utf-8")