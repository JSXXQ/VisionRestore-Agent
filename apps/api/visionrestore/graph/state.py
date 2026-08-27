from __future__ import annotations

from typing import Annotated, Any

from typing_extensions import TypedDict


def append_unique(left: list[dict] | None, right: list[dict] | None) -> list[dict]:
    result = list(left or [])
    seen = {
        item.get("event_id") or item.get("message_id") or item.get("candidate_id")
        for item in result
        if isinstance(item, dict)
    }
    for item in right or []:
        if not isinstance(item, dict):
            continue
        key = item.get("event_id") or item.get("message_id") or item.get("candidate_id")
        if key and key in seen:
            continue
        result.append(item)
        if key:
            seen.add(key)
    return result


def merge_candidate_results(left: list[dict] | None, right: list[dict] | None) -> list[dict]:
    ordered: list[dict] = []
    positions: dict[str, int] = {}
    for item in [*(left or []), *(right or [])]:
        if not isinstance(item, dict):
            continue
        key = str(item.get("candidate_id") or item.get("output_file_id") or f"candidate-{len(ordered)}")
        if key in positions:
            ordered[positions[key]] = item
        else:
            positions[key] = len(ordered)
            ordered.append(item)
    return ordered


class VisionRestoreState(TypedDict, total=False):
    task_id: str
    run_id: str
    workflow_version: str
    task_projection: dict[str, Any]
    input_file: dict[str, Any]
    input_path: str

    image_analysis: dict[str, Any]
    user_intent: dict[str, Any]
    hardware_snapshot: dict[str, Any]
    model_snapshot: list[dict[str, Any]]
    retrieved_context: list[dict[str, Any]]
    ai_advisory: dict[str, Any]
    region_constraints: list[dict[str, Any]]

    candidate_plan: dict[str, Any]
    pending_candidates: list[dict[str, Any]]
    candidate_to_execute: dict[str, Any]
    candidate_results: Annotated[list[dict[str, Any]], merge_candidate_results]
    candidate_ranking: dict[str, Any]
    selected_candidate: dict[str, Any]

    retry_count: int
    residual_diagnosis: dict[str, Any]
    next_postprocess_operation: str
    pending_confirmation: dict[str, Any]
    postprocess_decision: dict[str, Any]
    last_postprocess_result: dict[str, Any]

    current_phase: str
    terminal_status: str | None
    error: dict[str, Any] | None
    events: Annotated[list[dict[str, Any]], append_unique]
    messages: Annotated[list[dict[str, Any]], append_unique]


class CandidateExecutionInput(TypedDict, total=False):
    task_id: str
    run_id: str
    workflow_version: str
    candidate_to_execute: dict[str, Any]
    input_path: str
    region_constraints: list[dict[str, Any]]
    task_projection: dict[str, Any]


class CandidateExecutionOutput(TypedDict, total=False):
    candidate_results: list[dict[str, Any]]
    events: list[dict[str, Any]]
    messages: list[dict[str, Any]]


class CandidateExecutionState(CandidateExecutionInput, CandidateExecutionOutput, total=False):
    pass
