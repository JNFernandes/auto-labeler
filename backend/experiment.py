"""The experiment: how much work does the agent save, and how accurate is the result?

Runs the agent on a fixed set of validation images, once without Claude and once with the
Claude critic, and lets KITTI's ground truth play the human reviewer.

    python experiment.py --images 100            # dry run: shows the plan and the Claude cost, sends nothing
    python experiment.py --images 100 --go       # really asks Claude (answers are cached on disk)
"""
import argparse
from pathlib import Path

from graph.pipeline import build_graph, label_image
from graph.state import Settings
from tools.kitti import KittiDataset, get_splits
from tools.simulate import is_true, simulate_policy

ROOT = Path("data/kitti")
OUT = Path("runs")


def run_agent(graph, dataset, ids):
    """Run the agent on every image. Returns ({id: final boxes}, dollars spent on Claude)."""
    boxes, cost = {}, 0.0
    for n, image_id in enumerate(ids, 1):
        state = label_image(graph, dataset.load_image(image_id), image_id)
        boxes[image_id] = state["boxes"]
        cost += state.get("api_cost", 0.0)
        if n % 20 == 0:
            print(f"  {n}/{len(ids)} images")
    return boxes, cost


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", type=int, default=100, help="how many images (the first ones of the split)")
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--accept-above", type=float, default=0.60)
    parser.add_argument("--drop-below", type=float, default=0.20)
    parser.add_argument("--min-height", type=float, default=0.0)
    parser.add_argument("--critic-model", default="claude-sonnet-5-5")
    parser.add_argument("--go", action="store_true", help="really call Claude (otherwise: dry run)")
    args = parser.parse_args()

    import anthropic
    from dotenv import load_dotenv
    from agents.critic import Critic, estimate_cost
    from agents.proposer import GroundingDinoProposer
    from agents.verifier import ClipVerifier

    load_dotenv(Path(__file__).resolve().parent / ".env")
    dataset = KittiDataset(ROOT)
    train_ids, val_ids = get_splits(dataset)
    ids = (train_ids if args.split == "train" else val_ids)[: args.images]
    gts = {i: dataset.gts(i) for i in ids}
    settings = Settings(args.accept_above, args.drop_below, args.min_height)
    proposer, verifier = GroundingDinoProposer(), ClipVerifier()
    critic = Critic(anthropic.Anthropic(), args.critic_model, cache_path=OUT / f"critic_{args.critic_model}.json")

    print(f"{len(ids)} {args.split} images, accept_above={args.accept_above}, drop_below={args.drop_below}, "
          f"min_height={args.min_height}")
    print("agent without Claude:")
    plain_boxes, _ = run_agent(build_graph(proposer, verifier, None, settings), dataset, ids)

    # the plan: which boxes would Claude be asked about, and what does that cost?
    todo = [(i, d) for i in ids for d in plain_boxes[i] if settings.is_uncertain(d)]
    uncached = sum(not critic.is_cached(dataset.load_image(i), d) for i, d in todo)
    print(f"\nClaude would look at {len(todo)} uncertain boxes; {uncached} are not answered yet -> "
          f"about {uncached} calls with {args.critic_model}, estimated ${estimate_cost(args.critic_model, uncached):.2f}")
    if uncached and not args.go:
        raise SystemExit("Dry run: nothing was sent to Claude. Add --go to make the calls.")

    print("agent with Claude:")
    critic_boxes, cost = run_agent(build_graph(proposer, verifier, critic, settings), dataset, ids)

    # how good were Claude's rejections?
    vetoed = [(i, d) for i in ids for d in critic_boxes[i] if d.source == "critic"]
    right = sum(not is_true(d, gts[i]) for i, d in vetoed)

    print(f"\n{'':<22}{'auto':>6}{'auto prec':>11}{'human/img':>11}{'AP auto':>9}{'AP auto+human':>15}{'Claude $':>10}")
    for label, boxes, spent in (("agent without Claude", plain_boxes, 0.0), ("agent with Claude", critic_boxes, cost)):
        r = simulate_policy(boxes, gts, args.accept_above, args.drop_below, args.min_height)
        print(f"{label:<22}{r['auto']:>6}{r['auto_precision']:>10.0f}%{r['review_per_image']:>11.1f}"
              f"{r['ap_auto']:>9.1f}{r['ap_final']:>15.1f}{spent:>10.3f}")
    print(f"\nClaude rejected {len(vetoed)} boxes; {right} of them are really not in KITTI "
          f"({100 * right / max(len(vetoed), 1):.0f}%)")


if __name__ == "__main__":
    main()