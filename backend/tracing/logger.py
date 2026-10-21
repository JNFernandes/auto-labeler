"""A readable record of what the agent decided for each image (the UI will show this)."""
import json
from pathlib import Path

from tools.types import Detection


def box_record(d: Detection) -> dict:
    b = d.box
    return {"class": d.cls, "box": [round(b.x1, 1), round(b.y1, 1), round(b.x2, 1), round(b.y2, 1)],
            "confidence": round(d.score, 3), "source": d.source}


def run_record(name: str, state: dict) -> dict:
    return {"image": name,
            "accepted": [box_record(d) for d in state["accepted"]],
            "for_human": [box_record(d) for d in state["for_human"]],
            "dropped": [box_record(d) for d in state["dropped"]],
            "trace": state["trace"],
            "claude_cost": round(state.get("api_cost", 0.0), 5)}


def save_run(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=1), encoding="utf-8")