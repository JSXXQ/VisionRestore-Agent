from copy import deepcopy
import re
from typing import Any

from visionrestore.core.model_config import get_routing_rules
from visionrestore.schemas.ai import CHECKPOINTS_BY_MODEL, CheckpointSuggestion, ModelSuggestion, MultimodalAnalysisResult

DEFAULT_MULTIMODAL_ROUTING = {
    "semantic_bonus_max": 20,
    "scene_confidence_threshold": 0.65,
    "minimum_overall_confidence": 0.55,
}

SECRET_OR_PATH_PATTERNS = [
    re.compile(r"\b[A-Z0-9_]*(API|TOKEN|SECRET|KEY)[A-Z0-9_]*\b", re.IGNORECASE),
    re.compile(r"\b[A-Za-z]:\\[^ \n\r\t]+"),
    re.compile(r"(?<![A-Za-z0-9_])/(home|users|var|etc|mnt|root)/[^ \n\r\t]+", re.IGNORECASE),
]


def validate_multimodal_result(
    result: MultimodalAnalysisResult,
    *,
    available_models: list[dict[str, Any]],
    hardware_summary: dict[str, Any],
    manual_model: str | None = None,
    manual_checkpoint: str | None = None,
) -> MultimodalAnalysisResult:
    validated = result.model_copy(deep=True)
    rules = _multimodal_rules()
    status = {item.get("model_id"): item for item in available_models}
    existing_weights = _existing_weights_by_model(available_models)
    errors: list[str] = []
    warnings: list[str] = list(validated.warnings or [])
    checks: dict[str, Any] = {
        "json_parse": "passed",
        "pydantic_schema": "passed",
        "model_whitelist": "passed",
        "checkpoint_whitelist": "passed",
        "model_checkpoint_combo": "passed",
        "local_existence": "passed",
        "health_status": "passed",
        "hardware_resources": "passed",
        "confidence_thresholds": "passed",
        "manual_override_protected": bool(manual_model or manual_checkpoint),
        "semantic_bonus_max": rules["semantic_bonus_max"],
        "scene_confidence_threshold": rules["scene_confidence_threshold"],
        "minimum_overall_confidence": rules["minimum_overall_confidence"],
        "original_scene": validated.scene,
    }

    if _contains_forbidden_output(validated):
        errors.append("FORBIDDEN_OUTPUT_CONTENT")
        checks["output_safety"] = "failed"
    else:
        checks["output_safety"] = "passed"

    if validated.scene_confidence < rules["scene_confidence_threshold"]:
        validated.scene = "unknown"
        checks["confidence_thresholds"] = "scene_low"
        warnings.append("Scene confidence is below threshold; scene was set to unknown.")
        validated.checkpoint_candidates = _default_retinexformer_candidate(validated.checkpoint_candidates)
    checks["adjusted_scene"] = validated.scene

    if validated.confidence < rules["minimum_overall_confidence"]:
        errors.append("OVERALL_CONFIDENCE_BELOW_THRESHOLD")
        checks["confidence_thresholds"] = "overall_low"

    valid_model_candidates: list[ModelSuggestion] = []
    for candidate in validated.model_candidates:
        if candidate.model_id not in CHECKPOINTS_BY_MODEL:
            errors.append(f"MODEL_NOT_ALLOWED:{candidate.model_id}")
            checks["model_whitelist"] = "failed"
            continue
        model_status = status.get(candidate.model_id)
        if not model_status or not model_status.get("available"):
            errors.append(f"MODEL_NOT_AVAILABLE:{candidate.model_id}")
            checks["local_existence"] = "failed"
            continue
        if _hardware_blocks_model(candidate.model_id, model_status, hardware_summary):
            errors.append(f"HARDWARE_RESOURCE_BLOCKED:{candidate.model_id}")
            checks["hardware_resources"] = "failed"
            continue
        valid_model_candidates.append(candidate)

    valid_checkpoint_candidates: list[CheckpointSuggestion] = []
    for candidate in validated.checkpoint_candidates:
        if candidate.model_id not in CHECKPOINTS_BY_MODEL:
            errors.append(f"CHECKPOINT_MODEL_NOT_ALLOWED:{candidate.model_id}")
            checks["checkpoint_whitelist"] = "failed"
            continue
        if candidate.checkpoint_id not in CHECKPOINTS_BY_MODEL[candidate.model_id]:
            errors.append(f"CHECKPOINT_NOT_ALLOWED:{candidate.model_id}:{candidate.checkpoint_id}")
            checks["checkpoint_whitelist"] = "failed"
            continue
        weight = existing_weights.get(candidate.model_id, {}).get(candidate.checkpoint_id)
        if not weight:
            errors.append(f"CHECKPOINT_NOT_AVAILABLE:{candidate.model_id}:{candidate.checkpoint_id}")
            checks["local_existence"] = "failed"
            continue
        if weight.get("health_check") == "failed":
            errors.append(f"CHECKPOINT_HEALTH_FAILED:{candidate.model_id}:{candidate.checkpoint_id}")
            checks["health_status"] = "failed"
            continue
        if weight.get("health_check") == "not_run":
            checks["health_status"] = "not_run"
        if _hardware_blocks_model(candidate.model_id, status.get(candidate.model_id, {}), hardware_summary):
            errors.append(f"CHECKPOINT_HARDWARE_BLOCKED:{candidate.model_id}:{candidate.checkpoint_id}")
            checks["hardware_resources"] = "failed"
            continue
        valid_checkpoint_candidates.append(candidate)

    validated.model_candidates = valid_model_candidates
    validated.checkpoint_candidates = valid_checkpoint_candidates

    if not valid_model_candidates:
        errors.append("NO_VALID_MODEL_CANDIDATE")
    if not valid_checkpoint_candidates:
        errors.append("NO_VALID_CHECKPOINT_CANDIDATE")

    if manual_model or manual_checkpoint:
        validated.adopted = False
        validated.rejection_reason = "USER_MANUAL_SELECTION_HAS_PRIORITY"
        warnings.append("User manual model or checkpoint selection has priority over multimodal advice.")
    elif errors:
        validated.adopted = False
        validated.rejection_reason = errors[0]
    else:
        validated.adopted = True
        validated.adoption_reason = "Accepted as bounded semantic advice only; final route remains local."

    validated.validation_errors = sorted(set(errors))
    validated.validation_passed = not bool(errors)
    validated.local_validation = checks
    validated.warnings = _dedupe(warnings)
    if errors:
        validated.fallback_used = True
        validated.failure_reason = validated.failure_reason or "MULTIMODAL_LOCAL_VALIDATION_FAILED"
    return validated


def mark_invalid_response(
    result: MultimodalAnalysisResult,
    *,
    code: str,
    message: str,
) -> MultimodalAnalysisResult:
    fallback = result.model_copy(deep=True)
    fallback.validation_passed = False
    fallback.validation_errors = [code]
    fallback.local_validation = {
        "json_parse": "unknown" if code != "AI_PROVIDER_INVALID_RESPONSE" else "failed",
        "pydantic_schema": "failed" if code == "AI_PROVIDER_INVALID_RESPONSE" else "unknown",
    }
    fallback.adopted = False
    fallback.rejection_reason = code
    fallback.fallback_used = True
    fallback.failure_reason = code
    fallback.warnings = _dedupe(list(fallback.warnings or []) + [message])
    return fallback


def _multimodal_rules() -> dict[str, float]:
    configured = deepcopy(get_routing_rules().get("multimodal_routing", {}) or {})
    return DEFAULT_MULTIMODAL_ROUTING | configured


def _existing_weights_by_model(models: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for model in models:
        model_id = model.get("model_id")
        out[model_id] = {}
        for weight in model.get("capabilities", {}).get("weights", []) or []:
            if weight.get("status") == "found" and weight.get("exists", True):
                out[model_id][weight.get("checkpoint_id")] = weight
    return out


def _default_retinexformer_candidate(candidates: list[CheckpointSuggestion]) -> list[CheckpointSuggestion]:
    safe = [
        item for item in candidates
        if not (item.model_id == "retinexformer" and item.checkpoint_id in {"sdsd_indoor", "sdsd_outdoor"})
    ]
    if not any(item.model_id == "retinexformer" and item.checkpoint_id == "lol_v2_real" for item in safe):
        safe.insert(0, CheckpointSuggestion(
            model_id="retinexformer",
            checkpoint_id="lol_v2_real",
            score=0,
            reason="Default Retinexformer candidate when scene confidence is low.",
        ))
    return safe


def _hardware_blocks_model(model_id: str, model_status: dict[str, Any], hardware: dict[str, Any]) -> bool:
    supported = set(model_status.get("supported_devices") or [])
    if "cuda" in supported and not hardware.get("cuda_available"):
        return True
    return False


def _contains_forbidden_output(result: MultimodalAnalysisResult) -> bool:
    payload = result.model_dump()
    text_parts: list[str] = []
    _collect_strings(payload, text_parts)
    return any(pattern.search(text) for text in text_parts for pattern in SECRET_OR_PATH_PATTERNS)


def _collect_strings(value: Any, out: list[str]) -> None:
    if isinstance(value, str):
        out.append(value)
    elif isinstance(value, dict):
        for item in value.values():
            _collect_strings(item, out)
    elif isinstance(value, list):
        for item in value:
            _collect_strings(item, out)


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out
