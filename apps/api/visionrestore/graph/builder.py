from langgraph.graph import END, START, StateGraph

from visionrestore.graph.state import VisionRestoreState
from visionrestore.graph.subgraphs import build_candidate_execution_subgraph, build_postprocess_subgraph


def build_visionrestore_graph(runtime, *, checkpointer):
    builder = StateGraph(VisionRestoreState)
    builder.add_node("initialize_task", runtime.initialize_task)
    builder.add_node("analyze_input", runtime.analyze_input)
    builder.add_node("parse_intent", runtime.parse_intent)
    builder.add_node("inspect_runtime", runtime.inspect_runtime)
    builder.add_node("retrieve_context", runtime.retrieve_context)
    builder.add_node("semantic_analysis", runtime.semantic_analysis)
    builder.add_node("plan_candidates", runtime.plan_candidates)
    builder.add_node("candidate_execution", build_candidate_execution_subgraph(runtime))
    builder.add_node("aggregate_candidates", runtime.aggregate_candidates)
    builder.add_node("plan_region_retry", runtime.plan_region_retry)
    builder.add_node("rank_candidates", runtime.rank_candidates)
    builder.add_node("select_best", runtime.select_best)
    builder.add_node("diagnose_residual", runtime.diagnose_residual)
    builder.add_node("postprocess_workflow", build_postprocess_subgraph(runtime))
    builder.add_node("finalize", runtime.finalize)
    builder.add_node("fail_task", runtime.fail_task)
    builder.add_node("cancel_task", runtime.cancel_task)

    builder.add_edge(START, "initialize_task")
    builder.add_edge("initialize_task", "analyze_input")
    builder.add_edge("analyze_input", "parse_intent")
    builder.add_edge("parse_intent", "inspect_runtime")
    builder.add_edge("inspect_runtime", "retrieve_context")
    builder.add_edge("retrieve_context", "semantic_analysis")
    builder.add_edge("semantic_analysis", "plan_candidates")
    builder.add_conditional_edges("plan_candidates", runtime.dispatch_candidates)
    builder.add_edge("candidate_execution", "aggregate_candidates")
    builder.add_conditional_edges(
        "aggregate_candidates",
        runtime.route_after_aggregate,
        {"plan_region_retry": "plan_region_retry", "rank_candidates": "rank_candidates", "cancel_task": "cancel_task"},
    )
    builder.add_conditional_edges("plan_region_retry", runtime.dispatch_candidates)
    builder.add_edge("rank_candidates", "select_best")
    builder.add_conditional_edges(
        "select_best",
        lambda state: "diagnose_residual" if state.get("selected_candidate") else "fail_task",
        {"diagnose_residual": "diagnose_residual", "fail_task": "fail_task"},
    )
    builder.add_conditional_edges(
        "diagnose_residual",
        runtime.route_after_diagnosis,
        {"postprocess_workflow": "postprocess_workflow", "finalize": "finalize"},
    )
    builder.add_edge("postprocess_workflow", "finalize")
    builder.add_edge("finalize", END)
    builder.add_edge("fail_task", END)
    builder.add_edge("cancel_task", END)
    return builder.compile(checkpointer=checkpointer)
