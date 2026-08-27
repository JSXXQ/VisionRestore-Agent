from langgraph.graph import END, START, StateGraph

from visionrestore.graph.state import CandidateExecutionInput, CandidateExecutionOutput, CandidateExecutionState


def build_candidate_execution_subgraph(runtime):
    builder = StateGraph(
        CandidateExecutionState,
        input_schema=CandidateExecutionInput,
        output_schema=CandidateExecutionOutput,
    )
    builder.add_node("execute_candidate", runtime.execute_candidate)
    builder.add_edge(START, "execute_candidate")
    builder.add_edge("execute_candidate", END)
    return builder.compile()
