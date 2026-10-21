from PIL import Image

from graph.policy import triage
from tools.types import Box, Detection, GTObject
from tools.visualize import draw_result


def det(score):
    return Detection("Car", Box(10, 10, 60, 40), score)


def test_triage_splits_by_confidence_with_inclusive_lower_edges():
    accepted, human, dropped = triage([det(0.9), det(0.6), det(0.5), det(0.2), det(0.1)], 0.6, 0.2)
    assert [d.score for d in accepted] == [0.9, 0.6]      # exactly at accept_above is accepted
    assert [d.score for d in human] == [0.5, 0.2]         # exactly at drop_below is still asked
    assert [d.score for d in dropped] == [0.1]


def test_draw_result_returns_a_copy_and_marks_pixels():
    image = Image.new("RGB", (200, 100), "black")
    out = draw_result(image, [det(0.9)], [det(0.5)], [det(0.1)], [GTObject("Car", Box(100, 10, 150, 50))])
    assert out.size == image.size and out is not image
    assert image.getpixel((10, 20)) == (0, 0, 0)          # the original is untouched
    assert out.getpixel((10, 20)) != (0, 0, 0)            # the box edge was drawn


def test_triage_drops_boxes_below_min_height_whatever_their_confidence():
    tall = Detection("Car", Box(0, 0, 80, 40), 0.9)
    tiny = Detection("Car", Box(0, 0, 30, 12), 0.9)
    accepted, human, dropped = triage([tall, tiny], 0.6, 0.2, min_height=25)
    assert accepted == [tall] and human == [] and dropped == [tiny]
    accepted, _, dropped = triage([tall, tiny], 0.6, 0.2)          # default: no size rule
    assert accepted == [tall, tiny] and dropped == []