"""The API the UI talks to.

    uvicorn api:app --reload        # then open http://127.0.0.1:8000/docs
"""
import io
import os
import threading
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image

from graph.pipeline import build_graph, label_image
from graph.state import Settings
from tools.kitti import KittiDataset
from tracing.logger import box_record, run_record

load_dotenv(Path(__file__).resolve().parent / ".env")

ROOT = Path("data/kitti")
OUT = Path("runs")
CRITIC_MODEL = "claude-sonnet-5-5"

app = FastAPI(title="KITTI auto-labeler")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["*"], allow_headers=["*"])

_cache: dict = {}            # models are loaded on first use, then kept
_gpu_lock = threading.Lock() # one image at a time on the GPU


def get_dataset() -> KittiDataset:
    if "dataset" not in _cache:
        _cache["dataset"] = KittiDataset(ROOT)
    return _cache["dataset"]


def get_models():
    if "models" not in _cache:
        from agents.proposer import GroundingDinoProposer
        from agents.verifier import ClipVerifier
        _cache["models"] = (GroundingDinoProposer(), ClipVerifier())
    return _cache["models"]


def get_critic():
    if "critic" not in _cache:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise HTTPException(400, "ANTHROPIC_API_KEY is not set in backend/.env, so Claude cannot be used.")
        import anthropic
        from agents.critic import Critic
        _cache["critic"] = Critic(anthropic.Anthropic(), CRITIC_MODEL, cache_path=OUT / f"critic_{CRITIC_MODEL}.json")
    return _cache["critic"]


def run_agent(image: Image.Image, name: str, accept_above: float, drop_below: float,
              min_height: float, zoom: bool, critic: bool) -> dict:
    if not 0.0 <= drop_below < accept_above <= 1.0:
        raise HTTPException(400, "Need 0 <= drop_below < accept_above <= 1.")
    proposer, verifier = get_models()
    graph = build_graph(proposer, verifier, get_critic() if critic else None,
                        Settings(accept_above, drop_below, min_height, zoom))
    with _gpu_lock:
        state = label_image(graph, image, name)
    return {**run_record(name, state), "width": image.width, "height": image.height}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/images")
def list_images(limit: int = 50, offset: int = 0):
    """KITTI image ids the UI can pick from."""
    ids = get_dataset().ids
    return {"total": len(ids), "ids": ids[offset: offset + limit]}


@app.get("/image/{image_id}")
def get_image(image_id: str):
    dataset = get_dataset()
    if image_id not in dataset.ids:
        raise HTTPException(404, f"No KITTI image '{image_id}'.")
    buffer = io.BytesIO()
    dataset.load_image(image_id).save(buffer, format="JPEG", quality=92)
    return Response(buffer.getvalue(), media_type="image/jpeg")


@app.post("/label/{image_id}")
def label_kitti(image_id: str, accept_above: float = 0.60, drop_below: float = 0.20, min_height: float = 0.0,
                zoom: bool = False, critic: bool = False):
    """Run the agent on a KITTI image. The answer also contains KITTI's own boxes ("truth") for comparison."""
    dataset = get_dataset()
    if image_id not in dataset.ids:
        raise HTTPException(404, f"No KITTI image '{image_id}'.")
    result = run_agent(dataset.load_image(image_id), image_id, accept_above, drop_below, min_height, zoom, critic)
    result["truth"] = [{"class": g.cls, "box": [round(g.box.x1, 1), round(g.box.y1, 1), round(g.box.x2, 1), round(g.box.y2, 1)]}
                       for g in dataset.gts(image_id) if g.cls != "DontCare"]
    return result


@app.post("/label-upload")
async def label_upload(file: UploadFile = File(...), accept_above: float = 0.60, drop_below: float = 0.20,
                       min_height: float = 0.0, zoom: bool = False, critic: bool = False):
    """Run the agent on a photo you upload."""
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception:
        raise HTTPException(400, "That file is not an image.")
    return run_agent(image, Path(file.filename or "upload").stem, accept_above, drop_below, min_height, zoom, critic)