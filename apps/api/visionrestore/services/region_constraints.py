from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from visionrestore.schemas.region import RegionConstraint, RegionConstraintEvaluation


LIGHT_TARGET_WORDS = {
    "streetlight": ["streetlight", "street light", "lamp", "light pole", "路灯", "灯柱", "灯杆"],
    "lamp": ["lamp", "light", "灯", "光源"],
}
OVEREXPOSURE_WORDS = ["overexposure", "overexposed", "blown", "highlight", "clip", "过曝", "爆掉", "高光", "刺眼"]


class RegionConstraintService:
    def build_constraints(
        self,
        *,
        user_request: str,
        image_path: str,
        image_size: list[int] | tuple[int, int] | None,
        parameters: dict | None = None,
        ai_analysis: Any = None,
    ) -> list[RegionConstraint]:
        width, height = _image_size(image_path, image_size)
        constraints: list[RegionConstraint] = []
        for item in self._parameter_constraints(parameters or {}):
            constraint = self._coerce_constraint(item, width=width, height=height, source="api")
            if constraint:
                constraints.append(constraint)
        for item in getattr(ai_analysis, "region_constraints", []) or []:
            data = item.model_dump() if hasattr(item, "model_dump") else dict(item)
            constraint = self._coerce_constraint(data, width=width, height=height, source="multimodal")
            if constraint:
                constraints.append(constraint)
        if not constraints:
            constraints.extend(self._local_prompt_constraints(user_request, image_path, width, height))
        return self._dedupe(constraints)

    def evaluate(
        self,
        *,
        input_path: str,
        output_path: str,
        constraints: list[dict] | list[RegionConstraint],
    ) -> dict:
        normalized = [item if isinstance(item, RegionConstraint) else RegionConstraint(**item) for item in constraints]
        if not normalized:
            return {"enabled": False, "items": [], "score": 1.0, "hard_failed": False, "violations": []}
        with Image.open(input_path) as before_image, Image.open(output_path) as after_image:
            before = np.asarray(before_image.convert("RGB"))
            after = np.asarray(after_image.convert("RGB"))
        scale_x = after.shape[1] / before.shape[1]
        scale_y = after.shape[0] / before.shape[0]
        if abs(scale_x - scale_y) > 1e-6 or scale_x < 1:
            return {
                "enabled": True,
                "items": [],
                "score": 0.0,
                "hard_failed": True,
                "violations": ["region_constraint_size_mismatch"],
            }
        evaluations = [
            self._evaluate_one(before, after, item, output_scale=scale_x)
            for item in normalized
            if item.bbox
        ]
        scores = [item.score for item in evaluations]
        violations = [reason for item in evaluations if not item.passed for reason in item.reasons]
        hard_failed = any((not item.passed) and item.priority == "hard" for item in evaluations)
        severe_failure = any(self._is_severe_failure(item) for item in evaluations)
        return {
            "enabled": True,
            "items": [item.model_dump() for item in evaluations],
            "score": round(float(sum(scores) / len(scores)) if scores else 1.0, 4),
            "hard_failed": hard_failed,
            "severe_failure": severe_failure,
            "violations": violations,
            "policy": "region constraints are local checks after real model output; mild ROI violations affect user-match score, severe hard violations can eliminate a candidate",
        }

    def _parameter_constraints(self, parameters: dict) -> list[dict]:
        value = parameters.get("region_constraints") or parameters.get("roi_constraints") or []
        return value if isinstance(value, list) else []

    def _coerce_constraint(self, item: dict, *, width: int, height: int, source: str) -> RegionConstraint | None:
        if not isinstance(item, dict):
            return None
        data = dict(item)
        data.setdefault("source", source)
        data.setdefault("constraint_type", "avoid_overexposure")
        data.setdefault("priority", "hard")
        bbox = self._clamp_bbox(data.get("bbox") or [], width, height)
        if not bbox:
            return None
        data["bbox"] = bbox
        try:
            return RegionConstraint(**data)
        except Exception:
            return None

    def _local_prompt_constraints(self, user_request: str, image_path: str, width: int, height: int) -> list[RegionConstraint]:
        text = (user_request or "").lower()
        wants_light = any(word in text for words in LIGHT_TARGET_WORDS.values() for word in words)
        wants_overexposure = any(word in text for word in OVEREXPOSURE_WORDS)
        if not wants_light or not wants_overexposure:
            return []
        boxes = self._detect_bright_regions(image_path, width, height)
        constraints = []
        for index, bbox in enumerate(boxes[:2], start=1):
            constraints.append(RegionConstraint(
                target="streetlight" if any(word in text for word in LIGHT_TARGET_WORDS["streetlight"]) else "light_source",
                bbox=bbox,
                confidence=0.58 if index == 1 else 0.45,
                source="local_highlight_detector",
                constraint_type="avoid_overexposure",
                priority="hard",
                reason="User requested light-source highlight protection; local detector found a bright connected region.",
            ))
        return constraints

    def _detect_bright_regions(self, image_path: str, width: int, height: int) -> list[list[int]]:
        with Image.open(image_path) as image:
            arr = np.asarray(image.convert("RGB"))
        luminance = _luminance(arr)
        threshold = max(230.0, float(np.percentile(luminance, 99.2)))
        mask = luminance >= threshold
        visited = np.zeros(mask.shape, dtype=bool)
        boxes: list[tuple[int, list[int]]] = []
        min_area = max(12, int(width * height * 0.00003))
        max_area = max(min_area, int(width * height * 0.08))
        for y, x in np.argwhere(mask):
            if visited[y, x]:
                continue
            stack = [(int(y), int(x))]
            visited[y, x] = True
            xs: list[int] = []
            ys: list[int] = []
            while stack:
                cy, cx = stack.pop()
                xs.append(cx)
                ys.append(cy)
                for ny in range(max(0, cy - 1), min(mask.shape[0], cy + 2)):
                    for nx in range(max(0, cx - 1), min(mask.shape[1], cx + 2)):
                        if not visited[ny, nx] and mask[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))
            area = len(xs)
            if area < min_area or area > max_area:
                continue
            pad = max(8, int(max(width, height) * 0.015))
            bbox = self._clamp_bbox([min(xs) - pad, min(ys) - pad, max(xs) + pad + 1, max(ys) + pad + 1], width, height)
            if bbox:
                boxes.append((area, bbox))
        boxes.sort(key=lambda item: item[0], reverse=True)
        return [bbox for _, bbox in boxes]

    def _evaluate_one(
        self,
        before: np.ndarray,
        after: np.ndarray,
        constraint: RegionConstraint,
        output_scale: float = 1.0,
    ) -> RegionConstraintEvaluation:
        x1, y1, x2, y2 = constraint.bbox
        before_roi = before[y1:y2, x1:x2]
        sx1, sy1, sx2, sy2 = [int(round(value * output_scale)) for value in (x1, y1, x2, y2)]
        after_roi = after[sy1:sy2, sx1:sx2]
        before_lum = _luminance(before_roi)
        after_lum = _luminance(after_roi)
        before_over = float(np.mean(before_lum >= 250.0))
        after_over = float(np.mean(after_lum >= 250.0))
        before_p95 = float(np.percentile(before_lum, 95))
        after_p95 = float(np.percentile(after_lum, 95))
        before_mean = float(before_lum.mean())
        after_mean = float(after_lum.mean())
        reasons: list[str] = []
        if constraint.constraint_type == "avoid_overexposure":
            if after_over > constraint.max_overexposed_ratio:
                reasons.append(f"{constraint.target}区域过曝比例 {after_over:.3f} 超过阈值 {constraint.max_overexposed_ratio:.3f}")
            if after_p95 > constraint.max_luminance_p95 and after_p95 > before_p95 + 4:
                reasons.append(f"{constraint.target}区域高亮P95 {after_p95:.1f} 超过阈值 {constraint.max_luminance_p95:.1f}")
            if after_mean > before_mean + 45 and after_p95 > 248:
                reasons.append(f"{constraint.target}区域平均亮度增加 {after_mean - before_mean:.1f}，存在局部过增强风险")
        passed = not reasons
        over_penalty = max(0.0, after_over - constraint.max_overexposed_ratio) / max(0.01, 0.5 - constraint.max_overexposed_ratio)
        p95_penalty = max(0.0, after_p95 - constraint.max_luminance_p95) / max(1.0, 255.0 - constraint.max_luminance_p95)
        brightening_penalty = max(0.0, after_mean - before_mean - 35.0) / 120.0
        score = max(0.0, min(1.0, 1.0 - over_penalty * 0.55 - p95_penalty * 0.25 - brightening_penalty * 0.2))
        return RegionConstraintEvaluation(
            target=constraint.target,
            constraint_type=constraint.constraint_type,
            bbox=constraint.bbox,
            priority=constraint.priority,
            passed=passed,
            score=round(float(score), 4),
            overexposed_ratio_before=round(before_over, 4),
            overexposed_ratio_after=round(after_over, 4),
            luminance_p95_before=round(before_p95, 2),
            luminance_p95_after=round(after_p95, 2),
            mean_luminance_before=round(before_mean, 2),
            mean_luminance_after=round(after_mean, 2),
            reasons=reasons,
        )

    def _is_severe_failure(self, evaluation: RegionConstraintEvaluation) -> bool:
        if evaluation.priority != "hard" or evaluation.passed:
            return False
        if evaluation.overexposed_ratio_after >= 0.35 and evaluation.overexposed_ratio_after > evaluation.overexposed_ratio_before + 0.12:
            return True
        if evaluation.luminance_p95_after >= 254.0 and evaluation.luminance_p95_after > evaluation.luminance_p95_before + 12.0:
            return True
        return False

    def _clamp_bbox(self, bbox: list[int], width: int, height: int) -> list[int]:
        if not bbox or len(bbox) != 4 or width <= 0 or height <= 0:
            return []
        x1, y1, x2, y2 = [int(round(float(item))) for item in bbox]
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(x1 + 1, min(width, x2))
        y2 = max(y1 + 1, min(height, y2))
        if (x2 - x1) * (y2 - y1) < 4:
            return []
        return [x1, y1, x2, y2]

    def _dedupe(self, constraints: list[RegionConstraint]) -> list[RegionConstraint]:
        seen: set[tuple[str, tuple[int, int, int, int], str]] = set()
        out: list[RegionConstraint] = []
        for item in constraints:
            key = (item.target, tuple(item.bbox), item.constraint_type)
            if key in seen:
                continue
            seen.add(key)
            out.append(item)
        return out[:5]


def _image_size(image_path: str, provided: list[int] | tuple[int, int] | None) -> tuple[int, int]:
    if provided and len(provided) == 2:
        return int(provided[0]), int(provided[1])
    with Image.open(image_path) as image:
        return image.size


def _luminance(arr: np.ndarray) -> np.ndarray:
    rgb = arr.astype(np.float32)
    return rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114
