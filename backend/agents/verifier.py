"""Verifier: crop a box out of the image and ask CLIP how well it matches the class."""
from __future__ import annotations

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from tools.types import Box, Detection

CLASS_PROMPTS = {
    "Car": "a photo of a car",
    "Pedestrian": "a photo of a pedestrian walking",
    "Cyclist": "a photo of a cyclist riding a bicycle",
}
NEGATIVE_PROMPTS = [
    "a photo of a road",
    "a photo of a building",
    "a photo of trees or bushes",
    "a photo of a pole or traffic sign",
    "a photo of empty background",
]


class ClipVerifier:
    def __init__(self, model_id: str = "openai/clip-vit-base-patch32", margin: float = 0.10):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_id).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.margin = margin                      # extra context around the box
        self.names = list(CLASS_PROMPTS)
        self.prompts = [CLASS_PROMPTS[n] for n in self.names] + NEGATIVE_PROMPTS

    def verify(self, image: Image.Image, det: Detection) -> float:
        """Probability (0..1) that the crop is the class the detection claims."""
        b = det.box
        mx, my = b.width * self.margin, b.height * self.margin
        box = Box(b.x1 - mx, b.y1 - my, b.x2 + mx, b.y2 + my).clip(*image.size)
        if box.width < 8 or box.height < 8:
            return 0.0                            # too tiny to judge
        crop = image.crop((int(box.x1), int(box.y1), int(box.x2), int(box.y2)))
        inputs = self.processor(text=self.prompts, images=crop, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            probs = self.model(**inputs).logits_per_image.softmax(dim=-1)[0]
        return float(probs[self.names.index(det.cls)])