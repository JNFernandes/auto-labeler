"""Label one or two images with the agent and draw its decisions.

    python main.py 000010 000011
    python main.py 000010 --zoom --truth
    python main.py 000010 --critic                 # also asks Claude about uncertain boxes (costs a little)
    python main.py 000010 000011 --export runs/export
    python main.py C:\\photos\\street.jpg
"""
import argparse
from pathlib import Path

from PIL import Image

from agents.proposer import GroundingDinoProposer
from agents.verifier import ClipVerifier
from graph.pipeline import build_graph, label_image
from graph.state import Settings
from tools.export import save_coco, save_yolo
from tools.kitti import KittiDataset
from tools.visualize import draw_result
from tracing.logger import run_record, save_run

ROOT = Path("data/kitti")
OUT = Path("runs")

parser = argparse.ArgumentParser(description="Label one or two images and draw the agent's decisions.")
parser.add_argument("images", nargs="+", help="KITTI ids (e.g. 000010) or paths to image files")
parser.add_argument("--accept-above", type=float, default=0.60, help="auto-accept at or above this confidence")
parser.add_argument("--drop-below", type=float, default=0.20, help="discard below this confidence")
parser.add_argument("--min-height", type=float, default=0.0, help="discard boxes shorter than this many pixels")
parser.add_argument("--zoom", action="store_true", help="zoom-retry the uncertain boxes")
parser.add_argument("--critic", action="store_true", help="ask Claude to reject fake or wrong uncertain boxes")
parser.add_argument("--critic-model", default="claude-sonnet-5-5", help="Claude model used by --critic")
parser.add_argument("--truth", action="store_true", help="also draw the KITTI ground truth (blue, for comparison)")
parser.add_argument("--export", type=Path, help="folder to write YOLO labels, a COCO json and the decision log")
args = parser.parse_args()

critic = None
if args.critic:
    import anthropic
    from dotenv import load_dotenv
    from agents.critic import Critic
    load_dotenv(Path(__file__).resolve().parent / ".env")
    critic = Critic(anthropic.Anthropic(), args.critic_model, cache_path=OUT / f"critic_{args.critic_model}.json")

dataset = KittiDataset(ROOT)
settings = Settings(args.accept_above, args.drop_below, args.min_height, args.zoom)
graph = build_graph(GroundingDinoProposer(), ClipVerifier(), critic, settings)
OUT.mkdir(exist_ok=True)


def load(arg):
    """A file path, or a KITTI id. Returns (name, image, ground truth or None)."""
    path = Path(arg)
    if path.is_file():
        return path.stem, Image.open(path).convert("RGB"), None
    if arg not in dataset.ids:
        raise SystemExit(f"'{arg}' is neither a file nor a KITTI id")
    return arg, dataset.load_image(arg), dataset.gts(arg)


coco_items, records = [], []
for arg in args.images:
    name, image, truth = load(arg)
    state = label_image(graph, image, name)

    print(f"\n{name}")
    for line in state["trace"]:
        print(f"  - {line}")
    for label, group in (("auto ", state["accepted"]), ("human", state["for_human"]), ("drop ", state["dropped"])):
        for d in group:
            b = d.box
            print(f"  {label} {d.cls:<10} conf={d.score:.2f}  box=({b.x1:.0f}, {b.y1:.0f}, {b.x2:.0f}, {b.y2:.0f})")
    if critic is not None:
        print(f"  Claude cost for this image: ${state.get('api_cost', 0.0):.4f}")

    result = draw_result(image, state["accepted"], state["for_human"], state["dropped"], truth if args.truth else None)
    path = OUT / f"label_{name}.png"
    result.save(path)
    print(f"  saved {path}")

    coco_items.append((f"{name}.png", image.width, image.height, state["accepted"]))
    records.append(run_record(name, state))
    if args.export:
        save_yolo(args.export, name, state["accepted"], image.width, image.height)

if args.export:
    save_coco(args.export / "coco.json", coco_items)
    save_run(args.export / "run.json", records)
    print(f"\nexported to {args.export}: yolo/, coco.json, run.json")