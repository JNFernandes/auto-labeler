"""What the agent knows while it works on one image, and its settings."""
import operator
from dataclasses import dataclass
from typing import Annotated, TypedDict

from PIL import Image

from tools.types import Detection


@dataclass(frozen=True)
class Settings:
    accept_above: float = 0.60   # confidence at or above this: accepted automatically
    drop_below: float = 0.20     # confidence below this: discarded
    min_height: float = 0.0      # boxes shorter than this (pixels) are discarded
    zoom: bool = False           # look again, from closer, at uncertain boxes

    def is_uncertain(self, d: Detection) -> bool:
        """A box the agent is not sure about: worth a second look (zoom, Claude)."""
        return self.drop_below <= d.score < self.accept_above and d.box.height >= self.min_height


class State(TypedDict, total=False):
    name: str                                  # image name, for messages
    image: Image.Image
    boxes: list[Detection]                     # current boxes, updated by every node
    accepted: list[Detection]                  # final: kept automatically
    for_human: list[Detection]                 # final: waiting for a human
    dropped: list[Detection]                   # final: discarded
    api_cost: float                            # dollars spent on Claude for this image
    trace: Annotated[list[str], operator.add]  # what each agent did, in order