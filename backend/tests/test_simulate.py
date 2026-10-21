import pytest

from tools.simulate import simulate_policy
from tools.types import Box, Detection, GTObject


def test_simulate_counts_auto_queue_and_the_size_rule():
    gts = {"a": [GTObject("Car", Box(0, 0, 100, 50))]}
    boxes = {"a": [
        Detection("Car", Box(0, 0, 100, 50), 0.9),       # right, confident -> auto
        Detection("Car", Box(200, 0, 230, 15), 0.5),     # tiny, unsure
        Detection("Car", Box(300, 0, 380, 60), 0.4),     # wrong, unsure
    ]}
    plain = simulate_policy(boxes, gts, accept_above=0.6, drop_below=0.2)
    assert plain["auto"] == 1 and plain["auto_precision"] == 100.0
    assert plain["review_per_image"] == 2.0
    sized = simulate_policy(boxes, gts, accept_above=0.6, drop_below=0.2, min_height=25)
    assert sized["review_per_image"] == 1.0              # the tiny box left the queue
    assert sized["ap_final"] == pytest.approx(100.0, abs=3)