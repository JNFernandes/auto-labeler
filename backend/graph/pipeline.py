"""The agent as a LangGraph graph.

    propose -> verify -> [zoom] -> [critic] -> triage

Zoom and critic only run when the router finds uncertain boxes and the step is switched on.
"""
from langgraph.graph import END, START, StateGraph

from agents.critic import apply_verdict
from graph.policy import triage
from graph.retry import zoom_retry
from graph.state import Settings, State
from tools.boxes import nms_per_class
from tools.cyclists import merge_cyclists
from tools.scoring import fuse
from tools.types import Detection


def build_graph(proposer, verifier, critic=None, settings: Settings = Settings()):
    """`critic=None` switches the Claude step off."""
    s = settings

    uncertain = s.is_uncertain

    # --- nodes: each returns only what it changes ---------------------------------------
    def propose(state: State):
        boxes = merge_cyclists(proposer.propose(state["image"]))
        return {"boxes": boxes, "trace": [f"proposer: {len(boxes)} candidate boxes"]}

    def verify(state: State):
        image = state["image"]
        scored = [Detection(d.cls, d.box, fuse(d.score, verifier.verify(image, d)), d.source)
                  for d in state["boxes"]]
        boxes = nms_per_class(scored, 0.5)
        return {"boxes": boxes, "trace": [f"verifier: scored them, {len(boxes)} left after removing duplicates"]}

    def zoom(state: State):
        image, out, better = state["image"], [], 0
        for d in state["boxes"]:
            if uncertain(d):
                result = zoom_retry(image, d, proposer, verifier)
                if result is not None:
                    best, clip = result
                    new_conf = fuse(best.score, clip)
                    if new_conf > d.score:
                        d, better = Detection(best.cls, best.box, new_conf, "zoom"), better + 1
            out.append(d)
        return {"boxes": nms_per_class(out, 0.5), "trace": [f"zoom: {better} boxes got a better score"]}

    def ask_claude(state: State):
        image, out, notes = state["image"], [], []
        spent_before = critic.spent
        for d in state["boxes"]:
            if uncertain(d):
                verdict = critic.judge(image, d)
                if verdict.is_real is False:        # Claude only vetoes; a YES changes nothing
                    notes.append(f"critic: rejected {d.cls} (conf {d.score:.2f}): {verdict.reason}")
                    d = apply_verdict(d, verdict, s.accept_above)
            out.append(d)
        return {"boxes": out, "api_cost": critic.spent - spent_before,
                "trace": notes or ["critic: no box rejected"]}

    def decide(state: State):
        accepted, for_human, dropped = triage(state["boxes"], s.accept_above, s.drop_below, s.min_height)
        return {"accepted": accepted, "for_human": for_human, "dropped": dropped,
                "trace": [f"decision: {len(accepted)} accepted, {len(for_human)} for a human, {len(dropped)} dropped"]}

    # --- routers: plain rules, no model involved ----------------------------------------
    def has_uncertain(state: State) -> bool:
        return any(uncertain(d) for d in state["boxes"])

    def after_verify(state: State) -> str:
        if s.zoom and has_uncertain(state):
            return "zoom"
        return after_zoom(state)

    def after_zoom(state: State) -> str:
        return "critic" if critic is not None and has_uncertain(state) else "triage"

    graph = StateGraph(State)
    graph.add_node("propose", propose)
    graph.add_node("verify", verify)
    graph.add_node("zoom", zoom)
    graph.add_node("critic", ask_claude)
    graph.add_node("triage", decide)
    graph.add_edge(START, "propose")
    graph.add_edge("propose", "verify")
    graph.add_conditional_edges("verify", after_verify, {"zoom": "zoom", "critic": "critic", "triage": "triage"})
    graph.add_conditional_edges("zoom", after_zoom, {"critic": "critic", "triage": "triage"})
    graph.add_edge("critic", "triage")
    graph.add_edge("triage", END)
    return graph.compile()


def label_image(graph, image, name: str = "image") -> dict:
    """Run the agent on one image and return its final state."""
    return graph.invoke({"image": image, "name": name, "trace": []})