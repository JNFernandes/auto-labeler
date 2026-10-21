import math

import numpy as np
import pytest

from tools.evaluate import average_precision, match_image, _eval_image, evaluate, mean_ap, format_table
from tools.types import Box, Detection, GTObject

G = [Box(0, 0, 100, 100), Box(200, 0, 300, 100)]


def D(box, score):
    return Detection("Car", box, score)


def test_duplicate_detection_is_a_false_positive():
    dets = [D(Box(5, 5, 105, 105), 0.9), D(Box(0, 0, 98, 98), 0.8), D(Box(210, 10, 300, 100), 0.6)]
    assert match_image(dets, G, 0.7) == [(0.9, 1), (0.8, 0), (0.6, 1)]


def test_low_overlap_is_a_false_positive():
    assert match_image([D(Box(50, 0, 150, 100), 0.9)], G[:1], 0.7) == [(0.9, 0)]


def test_ap_worked_example():
    assert average_precision(np.array([0.9, 0.8, 0.6]), np.array([1, 0, 1]), 2) == pytest.approx(83.333, abs=0.01)


def test_ap_missed_object_costs_recall():
    assert average_precision(np.array([0.9, 0.8]), np.array([1, 0]), 2) == pytest.approx(50.0)


def test_ap_perfect_detector():
    assert average_precision(np.array([0.9, 0.8]), np.array([1, 1]), 2) == pytest.approx(100.0)


def test_ap_no_detections_is_zero():
    assert average_precision(np.array([]), np.array([]), 3) == 0.0


def test_ap_without_real_objects_is_nan():
    assert math.isnan(average_precision(np.array([0.9]), np.array([0]), 0))

def _scene():
    gts = [
        GTObject("Car", Box(100, 100, 300, 200), 0, 0),
        GTObject("Car", Box(500, 100, 530, 130), 0, 0),
        GTObject("Van", Box(600, 100, 800, 200), 0, 0),
        GTObject("DontCare", Box(900, 100, 1100, 200), -1, -1),
        GTObject("Pedestrian", Box(50, 300, 90, 380), 0, 0),
    ]
    boxes = [((105, 105, 305, 205), 0.9), ((502, 102, 531, 131), 0.8), ((605, 105, 800, 200), 0.7),
             ((920, 120, 1080, 180), 0.6), ((300, 250, 360, 300), 0.5), ((700, 300, 720, 320), 0.4)]
    return gts, [D(Box(*b), s) for b, s in boxes]


def test_easy_skips_small_car_van_dontcare_and_tiny_detection():
    gts, dets = _scene()
    assert _eval_image(dets, gts, "Car", "easy") == ([(0.9, 1), (0.5, 0)], 1)


def test_moderate_counts_the_small_car():
    gts, dets = _scene()
    assert _eval_image(dets, gts, "Car", "moderate") == ([(0.9, 1), (0.8, 1), (0.5, 0)], 2)


def test_detections_of_other_classes_are_not_judged():
    gts, _ = _scene()
    ped = [Detection("Pedestrian", Box(50, 300, 90, 380), 0.9)]
    assert _eval_image(ped, gts, "Car", "easy") == ([], 1)

def _car_scene():
    gts = {"a": [GTObject("Car", Box(100, 100, 300, 200), 0, 0)]}
    dets = {"a": [Detection("Car", Box(100, 100, 300, 200), 0.9)]}
    return gts, dets


def test_evaluate_perfect_detections():
    gts, dets = _car_scene()
    res = evaluate(dets, gts)
    assert res["Car"]["easy"] == pytest.approx(100.0)
    assert math.isnan(res["Pedestrian"]["easy"])          # no pedestrians in this scene


def test_evaluate_no_detections_is_zero():
    gts, _ = _car_scene()
    assert evaluate({}, gts)["Car"]["moderate"] == 0.0


def test_car_threshold_is_stricter_than_pedestrian_threshold():
    # both boxes are 200 wide and slid 40 px (20%), so their IoU with the truth is 0.67
    gts = {"a": [GTObject("Car", Box(100, 100, 300, 200)), GTObject("Pedestrian", Box(400, 100, 600, 200))]}
    dets = {"a": [Detection("Car", Box(140, 100, 340, 200), 0.9),
                  Detection("Pedestrian", Box(440, 100, 640, 200), 0.9)]}
    res = evaluate(dets, gts)
    assert res["Car"]["easy"] == 0.0                       # needs 0.7
    assert res["Pedestrian"]["easy"] == pytest.approx(100.0)   # needs 0.5


def test_mean_ap_skips_classes_without_objects():
    gts, dets = _car_scene()
    assert mean_ap(evaluate(dets, gts)) == pytest.approx(100.0)


def test_format_table_lists_every_class():
    gts, dets = _car_scene()
    text = format_table("title", evaluate(dets, gts))
    assert all(name in text for name in ("Car", "Pedestrian", "Cyclist", "n/a"))