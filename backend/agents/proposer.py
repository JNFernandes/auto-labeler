"""Proposer: text prompt in, candidate boxes out (Grounding DINO)."""
from __future__ import annotations

import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from tools.types import Box, Detection

DEFAULT_PHRASES = {"car": "car", "person": "person", "bicycle": "bicycle"}


class GroundingDinoProposer:
    def __init__(self, model_id="IDEA-Research/grounding-dino-tiny", box_threshold=0.25, text_threshold=0.25):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device).eval()
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

    def propose(self, image: Image.Image, phrases: dict[str, str] | None = None,
                region: Box | None = None) -> list[Detection]:
        """Find boxes. With `region`, only that window is analysed (zoom), and the boxes are
        returned in coordinates of the full image."""
        phrases = phrases or DEFAULT_PHRASES
        ox = oy = 0
        view = image
        if region is not None:
            ox, oy = int(region.x1), int(region.y1)
            view = image.crop((ox, oy, int(region.x2), int(region.y2)))
        text = " ".join(f"{p.lower()}." for p in phrases.values())

        inputs = self.processor(images=view, text=text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        kwargs = dict(text_threshold=self.text_threshold, target_sizes=[view.size[::-1]])
        try:
            result = self.processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids, threshold=self.box_threshold, **kwargs)[0]
        except TypeError:
            result = self.processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids, box_threshold=self.box_threshold, **kwargs)[0]

        names = result.get("text_labels") or result["labels"]
        dets: list[Detection] = []
        for box, score, name in zip(result["boxes"].tolist(), result["scores"].tolist(), names):
            label = self._canonical(str(name).lower(), phrases)
            if label is None:
                continue
            x1, y1, x2, y2 = box
            dets.append(Detection(label, Box(x1 + ox, y1 + oy, x2 + ox, y2 + oy), float(score), "grounding-dino"))
        return dets

    @staticmethod
    def _canonical(name: str, phrases: dict[str, str]) -> str | None:
        for label, phrase in phrases.items():
            if phrase.lower() in name:
                return label
        return None