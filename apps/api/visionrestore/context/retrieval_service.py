from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from visionrestore.core.config import PROJECT_ROOT, get_settings
from visionrestore.schemas.ai import RetrievedContext

KNOWLEDGE_DIR = PROJECT_ROOT / "apps" / "api" / "knowledge"
KNOWLEDGE_FILES = [
    "model_roles.json",
    "model_capability.json",
    "workflow_guardrails.json",
    "postprocess_policy.json",
    "eval_history.json",
]


class ContextRetrievalService:
    def __init__(self, knowledge_dir: Path = KNOWLEDGE_DIR):
        self.knowledge_dir = knowledge_dir
        self.settings = get_settings()

    def retrieve(
        self,
        *,
        user_request: str,
        image_metrics: Any,
        available_models: list[dict],
        hardware_summary: dict,
        max_items: int | None = None,
    ) -> list[RetrievedContext]:
        if not self.settings.context_retrieval_enabled:
            return []
        limit = max_items or self.settings.context_retrieval_max_items
        query_terms = self._query_terms(user_request, image_metrics, available_models, hardware_summary)
        scored: list[RetrievedContext] = []
        for item in self._load_items(str(self.knowledge_dir.resolve())):
            score, matched_terms = self._score(item, query_terms)
            if score <= 0:
                continue
            scored.append(RetrievedContext(
                item_id=str(item.get("id") or ""),
                source=str(item.get("source") or "knowledge"),
                title=str(item.get("title") or item.get("id") or "context"),
                content=self._sanitize_content(str(item.get("content") or "")),
                score=round(score, 3),
                tags=[str(tag) for tag in item.get("tags", [])][:12],
                matched_terms=matched_terms[:12],
            ))
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:limit]

    def build_prompt_context(
        self,
        *,
        user_request: str,
        image_metrics: Any,
        available_models: list[dict],
        hardware_summary: dict,
    ) -> dict:
        retrieved = self.retrieve(
            user_request=user_request,
            image_metrics=image_metrics,
            available_models=available_models,
            hardware_summary=hardware_summary,
        )
        return {
            "enabled": self.settings.context_retrieval_enabled,
            "policy": "Retrieved context is advisory only. It cannot override local validation, hard quality checks, ModelRegistry, HardwareInspector, or final real-output scoring.",
            "sources": [item.model_dump() for item in retrieved],
        }

    def model_reference(self, available_models: list[dict]) -> list[RetrievedContext]:
        """Return one stable family definition for each available model.

        This is deterministic prompt context, not a score and not vector RAG.
        """
        available_ids = {
            str(item.get("model_id") or "").lower()
            for item in available_models
            if item.get("available")
        }
        references: list[RetrievedContext] = []
        for item in self._load_items(str(self.knowledge_dir.resolve())):
            if str(item.get("source") or "") != "model_roles":
                continue
            tags = {str(tag).lower() for tag in item.get("tags", [])}
            matched_models = sorted(tags & available_ids)
            if not matched_models:
                continue
            references.append(
                RetrievedContext(
                    item_id=str(item.get("id") or matched_models[0]),
                    source="model_roles",
                    title=str(item.get("title") or matched_models[0]),
                    content=self._sanitize_content(str(item.get("content") or "")),
                    score=0,
                    tags=[str(tag) for tag in item.get("tags", [])][:12],
                    matched_terms=matched_models,
                )
            )
        return references

    def _query_terms(self, user_request: str, image_metrics: Any, available_models: list[dict], hardware_summary: dict) -> set[str]:
        terms = set(_tokens(user_request))
        metrics = image_metrics.model_dump() if hasattr(image_metrics, "model_dump") else dict(image_metrics or {})
        for key in ["noise_estimate", "laplacian_sharpness", "color_cast_index", "width", "height", "total_pixels"]:
            value = metrics.get(key)
            if value is None:
                continue
            terms.add(key.lower())
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if key == "noise_estimate" and numeric >= 12:
                terms.update({"noise", "denoise", "darkir"})
            if key == "laplacian_sharpness" and numeric < 80:
                terms.update({"blur", "darkir", "sharpness"})
            if key == "color_cast_index" and numeric >= 0.12:
                terms.update({"color", "hvi_cidnet", "natural"})
            if key in {"width", "height"} and numeric >= 2500:
                terms.update({"high-resolution", "flol", "low-vram"})
        if (hardware_summary.get("gpu_memory_mb") or 0) and (hardware_summary.get("gpu_memory_mb") or 0) < 4096:
            terms.update({"low-vram", "flol", "sci"})
        for model in available_models:
            model_id = str(model.get("model_id") or "").lower()
            if model.get("available") and model_id:
                terms.add(model_id)
        return terms

    def _score(self, item: dict, query_terms: set[str]) -> tuple[float, list[str]]:
        haystack = " ".join([
            str(item.get("id") or ""),
            str(item.get("source") or ""),
            str(item.get("title") or ""),
            str(item.get("content") or ""),
            " ".join(str(tag) for tag in item.get("tags", [])),
        ]).lower()
        item_terms = set(_tokens(haystack))
        tag_terms = {str(tag).lower() for tag in item.get("tags", [])}
        overlap = query_terms & (item_terms | tag_terms)
        score = float(len(overlap))
        if tag_terms & query_terms:
            score += 1.0
        if str(item.get("source")) in {"workflow_guardrails", "model_roles", "model_capability"}:
            score += 0.25
        return score, sorted(overlap)

    @staticmethod
    def _sanitize_content(content: str) -> str:
        blocked = ["api_key", "secret", "password", "token=", "bearer "]
        lowered = content.lower()
        if any(marker in lowered for marker in blocked):
            return "[redacted unsafe context]"
        return content[:900]

    @staticmethod
    @lru_cache
    def _load_items(knowledge_dir: str) -> tuple[dict, ...]:
        items: list[dict] = []
        for filename in KNOWLEDGE_FILES:
            path = Path(knowledge_dir) / filename
            if not path.exists():
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                items.extend(item for item in data if isinstance(item, dict))
            elif isinstance(data, dict) and isinstance(data.get("items"), list):
                items.extend(item for item in data["items"] if isinstance(item, dict))
        return tuple(items)


def _tokens(text: str) -> list[str]:
    return [token for token in re.split(r"[^0-9a-zA-Z_\-\u4e00-\u9fff]+", text.lower()) if token]
