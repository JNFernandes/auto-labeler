import json

from tools.export import coco_dict, save_yolo, yolo_lines
from tools.types import Box, Detection
from tracing.logger import run_record, save_run

CAR = Detection("Car", Box(100, 50, 300, 150), 0.8, "dino")


def test_yolo_line_is_normalised_centre_and_size():
    assert yolo_lines([CAR], 400, 200) == ["0 0.500000 0.500000 0.500000 0.500000"]


def test_save_yolo_writes_label_and_class_files(tmp_path):
    save_yolo(tmp_path, "000010", [CAR], 400, 200)
    assert (tmp_path / "yolo" / "000010.txt").read_text().startswith("0 0.5")
    assert (tmp_path / "yolo" / "classes.txt").read_text().split() == ["Car", "Pedestrian", "Cyclist"]


def test_coco_uses_x_y_width_height_and_one_based_categories():
    out = coco_dict([("a.png", 400, 200, [CAR])])
    ann = out["annotations"][0]
    assert ann["bbox"] == [100.0, 50.0, 200.0, 100.0] and ann["category_id"] == 1 and ann["image_id"] == 1
    assert out["categories"][0] == {"id": 1, "name": "Car"}


def test_run_record_is_saved_as_json(tmp_path):
    state = {"accepted": [CAR], "for_human": [], "dropped": [], "trace": ["x"], "api_cost": 0.01}
    save_run(tmp_path / "run.json", [run_record("a", state)])
    saved = json.loads((tmp_path / "run.json").read_text())
    assert saved[0]["accepted"][0]["class"] == "Car" and saved[0]["claude_cost"] == 0.01