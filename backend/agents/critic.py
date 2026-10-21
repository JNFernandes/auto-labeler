"""Critic: Claude (vision) looks at an uncertain crop and says whether it is a real object."""
from __future__ import annotations

import base64
import hashlib
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from tools.types import Box, Detection

NAMES = {"Car": "car", "Pedestrian": "pedestrian", "Cyclist": "cyclist riding a bicycle"}
PROMPT = (
    "This is a crop from a photo taken by a camera on a car. Is the main object a real {name} "
    "in the scene, and not a picture, poster, reflection or drawing of one? "
    "Answer YES or NO, then one short reason (at most 15 words)."
)
PRICES = {"claude-haiku-5-5": (0.10, 0.50), "claude-sonnet-5-5": (2.00, 10.00)}   # $ per million tokens (in, out)
ASSUMED_TOKENS = (250, 40)    # rough (input, output) tokens per call, used for cost estimates only
# These models think before answering by default, and thinking tokens are billed as output.
# A yes/no judgement does not need it, so we switch it off (each model has its own switch).
THINKING_OFF = {
    "claude-haiku-5-5": {"type": "disabled"},
    "claude-sonnet-5-5": {"type": "between_tools"},
}


@dataclass(frozen=True)
class Verdict:
    is_real: bool | None   # True = YES, False = NO, None = the answer was not a clear yes/no
    reason: str
    cost: float            # dollars spent on this call (0.0 when it came from the cache)


def parse_answer(text: str) -> tuple[bool | None, str]:
    match = re.match(r"^\W*(YES|NO)\b[\W]*(.*)$", text.strip(), flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return None, text.strip()
    return match.group(1).upper() == "YES", match.group(2).strip()


def answer_text(reply) -> str:
    """The visible answer: join the text blocks and skip any thinking block."""
    return "".join(b.text for b in reply.content if getattr(b, "type", None) == "text")


def call_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    price_in, price_out = PRICES.get(model, (0.0, 0.0))
    return (input_tokens * price_in + output_tokens * price_out) / 1e6


def estimate_cost(model: str, n_calls: int) -> float:
    return n_calls * call_cost(model, *ASSUMED_TOKENS)


class Critic:
    def __init__(self, client, model: str = "claude-haiku-5-5", cache_path: Path | None = None,
                 margin: float = 0.15, min_side: int = 256):
        self.client, self.model = client, model
        self.cache_path, self.margin, self.min_side = cache_path, margin, min_side
        self.cache: dict[str, dict] = {}
        if cache_path and cache_path.exists():
            self.cache = json.loads(cache_path.read_text(encoding="utf-8"))
        self.calls = 0
        self.spent = 0.0

    def _crop_png(self, image: Image.Image, box: Box) -> bytes:
        mx, my = box.width * self.margin, box.height * self.margin
        b = Box(box.x1 - mx, box.y1 - my, box.x2 + mx, box.y2 + my).clip(*image.size)
        crop = image.crop((int(b.x1), int(b.y1), int(b.x2), int(b.y2)))
        scale = max(1.0, self.min_side / min(crop.size))        # small crops are enlarged
        crop = crop.resize((int(crop.width * scale), int(crop.height * scale)))
        buf = io.BytesIO()
        crop.save(buf, format="PNG")
        return buf.getvalue()

    def _key(self, png: bytes, cls: str) -> str:
        return hashlib.sha256(self.model.encode() + cls.encode() + PROMPT.encode() + png).hexdigest()

    def is_cached(self, image: Image.Image, det: Detection) -> bool:
        return self._key(self._crop_png(image, det.box), det.cls) in self.cache

    def judge(self, image: Image.Image, det: Detection) -> Verdict:
        png = self._crop_png(image, det.box)
        key = self._key(png, det.cls)
        if key in self.cache:
            hit = self.cache[key]
            return Verdict(hit["is_real"], hit["reason"], 0.0)

        extra = {"thinking": THINKING_OFF[self.model]} if self.model in THINKING_OFF else {}
        reply = self.client.messages.create(
            model=self.model,
            max_tokens=300,
            **extra,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                             "data": base64.standard_b64encode(png).decode()}},
                {"type": "text", "text": PROMPT.format(name=NAMES[det.cls])},
            ]}],
        )
        is_real, reason = parse_answer(answer_text(reply))
        cost = call_cost(self.model, reply.usage.input_tokens, reply.usage.output_tokens)
        self.calls += 1
        self.spent += cost
        self.cache[key] = {"is_real": is_real, "reason": reason}
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self.cache), encoding="utf-8")
        return Verdict(is_real, reason, cost)


def apply_verdict(det: Detection, verdict: Verdict, accept_above: float) -> Detection:
    """YES promotes the box to the accepted zone, NO sinks it to 0 (dropped), unclear changes nothing."""
    if verdict.is_real is True:
        return Detection(det.cls, det.box, max(det.score, accept_above), "critic")
    if verdict.is_real is False:
        return Detection(det.cls, det.box, 0.0, "critic")
    return det