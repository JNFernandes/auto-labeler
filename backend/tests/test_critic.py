import pytest
from PIL import Image

from agents.critic import Critic, Verdict, apply_verdict, call_cost, parse_answer
from tools.types import Box, Detection


class FakeClient:
    """Stands in for anthropic.Anthropic(): counts calls and returns a canned answer."""

    def __init__(self, text="YES. A parked car."):
        self.text, self.n_calls, self.last_kwargs = text, 0, None
        self.messages = self

    def create(self, **kwargs):
        self.n_calls += 1
        self.last_kwargs = kwargs

        class Reply:
            pass

        reply = Reply()
        thinking = type("ThinkingBlock", (), {"type": "thinking"})()      # has no .text, like the real one
        answer = type("TextBlock", (), {"type": "text", "text": self.text})()
        reply.content = [thinking, answer] if self.text else [thinking]
        reply.usage = type("Usage", (), {"input_tokens": 1000, "output_tokens": 100})()
        return reply


IMAGE = Image.new("RGB", (400, 200), "gray")
DET = Detection("Car", Box(50, 50, 150, 110), 0.45)


def test_parse_answer():
    assert parse_answer("YES. Looks like a car.") == (True, "Looks like a car.")
    assert parse_answer("no, it is a poster")[0] is False
    assert parse_answer("NOT sure about this")[0] is None      # 'NOT' must not count as NO
    assert parse_answer("Maybe a car")[0] is None


def test_call_cost():
    assert call_cost("claude-haiku-5-5", 1000, 100) == pytest.approx(0.00015)
    assert call_cost("some-unknown-model", 1000, 100) == 0.0


def test_critic_calls_once_then_uses_the_cache(tmp_path):
    client = FakeClient("NO. It is a poster.")
    critic = Critic(client, model="claude-haiku-5-5", cache_path=tmp_path / "cache.json")
    first = critic.judge(IMAGE, DET)
    second = critic.judge(IMAGE, DET)
    assert client.n_calls == 1
    assert first.is_real is False and first.cost == pytest.approx(0.00015)
    assert second.is_real is False and second.cost == 0.0
    assert critic.is_cached(IMAGE, DET)
    again = Critic(FakeClient(), model="claude-haiku-5-5", cache_path=tmp_path / "cache.json")
    assert again.judge(IMAGE, DET).is_real is False             # the cache survives a restart


def test_apply_verdict():
    yes, no, unclear = Verdict(True, "", 0), Verdict(False, "", 0), Verdict(None, "", 0)
    assert apply_verdict(DET, yes, 0.6).score == 0.6
    assert apply_verdict(DET, yes, 0.4).score == 0.45          # never lowers a confidence
    assert apply_verdict(DET, no, 0.6).score == 0.0
    assert apply_verdict(DET, unclear, 0.6) is DET


def test_thinking_block_is_skipped_and_empty_answer_is_unclear(tmp_path):
    assert Critic(FakeClient("YES. ok"), cache_path=tmp_path / "a.json").judge(IMAGE, DET).is_real is True
    assert Critic(FakeClient(""), cache_path=tmp_path / "b.json").judge(IMAGE, DET).is_real is None


def test_thinking_is_switched_off_per_model(tmp_path):
    haiku, sonnet, other = FakeClient(), FakeClient(), FakeClient()
    Critic(haiku, model="claude-haiku-5-5").judge(IMAGE, DET)
    Critic(sonnet, model="claude-sonnet-5-5").judge(IMAGE, DET)
    Critic(other, model="some-other-model").judge(IMAGE, DET)
    assert haiku.last_kwargs["thinking"] == {"type": "disabled"}
    assert sonnet.last_kwargs["thinking"] == {"type": "between_tools"}
    assert "thinking" not in other.last_kwargs