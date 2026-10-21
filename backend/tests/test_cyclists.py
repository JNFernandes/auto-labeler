from tools.cyclists import merge_cyclists
from tools.types import Box, Detection


def det(cls, box, score=0.9):
    return Detection(cls, Box(*box), score)


def test_rider_and_bike_become_one_cyclist():
    out = merge_cyclists([det("bicycle", (100, 200, 180, 260), 0.8), det("person", (120, 130, 160, 240), 0.6)])
    assert len(out) == 1
    assert out[0].cls == "Cyclist"
    assert out[0].box == Box(100, 130, 180, 260)
    assert out[0].score == 0.6          # min of the two


def test_far_apart_person_stays_pedestrian():
    out = merge_cyclists([det("bicycle", (100, 200, 180, 260)), det("person", (400, 130, 440, 240))])
    assert [d.cls for d in out] == ["Pedestrian"]   # the lone bike is dropped


def test_two_bikes_one_person_only_best_bike_gets_it():
    out = merge_cyclists([
        det("bicycle", (100, 200, 180, 260), 0.9),
        det("bicycle", (110, 205, 190, 265), 0.5),
        det("person", (120, 130, 160, 240)),
    ])
    assert [d.cls for d in out] == ["Cyclist"]


def test_cars_pass_through():
    out = merge_cyclists([det("car", (0, 0, 50, 30))])
    assert [d.cls for d in out] == ["Car"]