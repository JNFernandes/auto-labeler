import pytest
from tools.types import Box, Detection
from tools.boxes import iou, nms, expand_box, nms_per_class

def _dets():
    return [
        Detection("Car", Box(0, 0, 100, 100), 0.90),
        Detection("Car", Box(10, 10, 110, 110), 0.80),
        Detection("Car", Box(300, 300, 400, 400), 0.70),
    ]

def test_identical_boxes():
    assert iou(Box(0, 0, 10, 10), Box(0, 0, 10, 10)) == pytest.approx(1.0)


def test_partial_overlap():
    assert iou(Box(0, 0, 10, 10), Box(5, 0, 15, 10)) == pytest.approx(50 / 150)


def test_no_overlap_far_apart():
    assert iou(Box(0, 0, 10, 10), Box(50, 50, 60, 60)) == 0.0


def test_touching_edges_is_zero():
    assert iou(Box(0, 0, 10, 10), Box(10, 0, 20, 10)) == 0.0


def test_containment():
    assert iou(Box(0, 0, 100, 100), Box(0, 0, 50, 50)) == pytest.approx(0.25)


def test_symmetric():
    a, b = Box(0, 0, 10, 10), Box(3, 4, 15, 12)
    assert iou(a, b) == pytest.approx(iou(b, a))

def test_nms_removes_duplicate_at_low_threshold():
    assert [d.score for d in nms(_dets(), 0.5)] == [0.90, 0.70]


def test_nms_keeps_duplicate_at_high_threshold():
    assert [d.score for d in nms(_dets(), 0.7)] == [0.90, 0.80, 0.70]


def test_nms_empty():
    assert nms([]) == []

def test_nms_per_class_keeps_different_classes():
    dets = [
        Detection("Car", Box(0, 0, 100, 100), 0.9),
        Detection("Pedestrian", Box(5, 5, 105, 105), 0.8),
    ]
    assert len(nms(dets)) == 1
    assert len(nms_per_class(dets)) == 2


def test_expand_box_raises_small_box_to_min_size():
    b = expand_box(Box(100, 150, 140, 190), 2.0, 1242, 375)
    assert (b.width, b.height) == (160, 160)
    assert b.center == (120, 170)


def test_expand_box_scales_large_box():
    b = expand_box(Box(400, 100, 600, 200), 2.0, 1242, 375)
    assert b == Box(300, 50, 700, 250)


def test_expand_box_is_clipped_to_image():
    b = expand_box(Box(0, 0, 40, 40), 2.0, 1242, 375)
    assert b == Box(0, 0, 100, 100)