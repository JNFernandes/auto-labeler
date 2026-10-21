from PIL import Image

from agents.critic import Verdict
from graph.pipeline import build_graph, label_image
from graph.state import Settings
from tools.types import Box, Detection

IMAGE = Image.new("RGB", (400, 200))
SURE = Box(10, 10, 110, 60)       # confident
MAYBE = Box(150, 10, 250, 60)     # uncertain
WEAK = Box(300, 10, 330, 60)      # very low


class FakeProposer:
    """Full image: three boxes. Zoomed view of MAYBE: the same box with a high score."""
    def propose(self, image, phrases=None, region=None):
        if region is not None:
            return [Detection("car", MAYBE, 0.95, "dino")]
        return [Detection("car", SURE, 0.95, "dino"), Detection("car", MAYBE, 0.50, "dino"),
                Detection("car", WEAK, 0.10, "dino")]


class FakeVerifier:
    def verify(self, image, det):
        return 1.0 if det.score > 0.9 else 0.0   # fused: 0.7*d + 0.3*c


class FakeCritic:
    spent = 0.0

    def __init__(self, answer):
        self.answer, self.asked = answer, []

    def judge(self, image, det):
        self.asked.append(det)
        return Verdict(self.answer, "because", 0.0)


def run(critic=None, **settings):
    graph = build_graph(FakeProposer(), FakeVerifier(), critic, Settings(**settings))
    return label_image(graph, IMAGE, "test")


def test_plain_run_splits_into_three_groups():
    out = run()
    assert [d.box for d in out["accepted"]] == [SURE]
    assert [d.box for d in out["for_human"]] == [MAYBE]
    assert [d.box for d in out["dropped"]] == [WEAK]
    assert out["trace"][0].startswith("proposer") and out["trace"][-1].startswith("decision")


def test_critic_no_drops_the_uncertain_box_and_is_asked_only_about_it():
    critic = FakeCritic(False)
    out = run(critic)
    assert [d.box for d in critic.asked] == [MAYBE]
    assert out["for_human"] == [] and MAYBE in [d.box for d in out["dropped"]]


def test_critic_yes_changes_nothing():
    out = run(FakeCritic(True))
    assert [d.box for d in out["for_human"]] == [MAYBE]


def test_zoom_can_lift_a_box_into_accepted():
    out = run(zoom=True)
    assert {d.box for d in out["accepted"]} == {SURE, MAYBE}
    assert any(line.startswith("zoom") for line in out["trace"])