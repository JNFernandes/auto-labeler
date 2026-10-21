import pytest
from PIL import Image

from graph.retry import zoom_retry
from tools.scoring import fuse
from tools.types import Box, Detection


class FakeProposer:
    def __init__(self, dets):
        self.dets, self.region = dets, None

    def propose(self, image, phrases=None, region=None):
        self.region = region
        return self.dets


class FakeVerifier:
    def __init__(self, score):
        self.score = score

    def verify(self, image, det):
        return self.score


IMAGE = Image.new("RGB", (1242, 375))
ORIGINAL = Detection("Car", Box(100, 100, 200, 160), 0.40)


def test_fuse():
    assert fuse(0.8, 0.4) == pytest.approx(0.68)


def test_retry_picks_overlapping_box_and_zooms_on_expanded_window():
    proposer = FakeProposer([
        Detection("car", Box(102, 98, 203, 162), 0.80),
        Detection("car", Box(500, 100, 600, 160), 0.90),
    ])
    best, clip = zoom_retry(IMAGE, ORIGINAL, proposer, FakeVerifier(0.7))
    assert best.box == Box(102, 98, 203, 162) and best.score == 0.80 and clip == 0.7
    assert proposer.region == Box(0, 40, 300, 220)


def test_retry_returns_none_when_nothing_overlaps():
    proposer = FakeProposer([Detection("car", Box(500, 100, 600, 160), 0.90)])
    assert zoom_retry(IMAGE, ORIGINAL, proposer, FakeVerifier(0.7)) is None


def test_retry_ignores_other_classes():
    proposer = FakeProposer([Detection("person", Box(102, 98, 203, 162), 0.90)])
    assert zoom_retry(IMAGE, ORIGINAL, proposer, FakeVerifier(0.7)) is None


def test_retry_cyclist_merges_inside_the_zoom_window():
    proposer = FakeProposer([
        Detection("bicycle", Box(100, 100, 180, 160), 0.7),
        Detection("person", Box(115, 60, 160, 140), 0.6),
    ])
    original = Detection("Cyclist", Box(100, 60, 180, 160), 0.35)
    best, _ = zoom_retry(IMAGE, original, proposer, FakeVerifier(0.5))
    assert best.cls == "Cyclist" and best.box == Box(100, 60, 180, 160) and best.score == 0.6