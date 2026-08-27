from langgraph.graph import END, START, StateGraph

from visionrestore.graph.state import VisionRestoreState


def build_postprocess_subgraph(runtime):
    builder = StateGraph(VisionRestoreState)
    builder.add_node("prepare_denoise", runtime.prepare_denoise)
    builder.add_node("prepare_sr", runtime.prepare_sr)
    builder.add_node("wait_denoise", runtime.wait_postprocess_decision)
    builder.add_node("wait_sr", runtime.wait_postprocess_decision)
    builder.add_node("apply_postprocess_decision", runtime.apply_postprocess_decision)

    builder.add_conditional_edges(
        START,
        runtime.route_postprocess_entry,
        {"prepare_denoise": "prepare_denoise", "prepare_sr": "prepare_sr", "done": END},
    )
    builder.add_edge("prepare_denoise", "wait_denoise")
    builder.add_edge("prepare_sr", "wait_sr")
    builder.add_edge("wait_denoise", "apply_postprocess_decision")
    builder.add_edge("wait_sr", "apply_postprocess_decision")
    builder.add_conditional_edges(
        "apply_postprocess_decision",
        runtime.route_after_postprocess,
        {"prepare_denoise": "prepare_denoise", "prepare_sr": "prepare_sr", "done": END},
    )
    return builder.compile()
