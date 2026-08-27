import json
from pathlib import Path
from typing import Any

from visionrestore.ai.prompts import prompt_registry_snapshot


class ReportService:
    def write_reports(self, task: dict, base_path: Path) -> dict:
        base_path.parent.mkdir(parents=True, exist_ok=True)
        json_path = base_path.with_suffix(".json")
        md_path = base_path.with_suffix(".md")
        json_path.write_text(json.dumps(task, ensure_ascii=False, indent=2), encoding="utf-8")

        best = task.get("best_result") or {}
        candidates = task.get("candidates") or []
        postprocess_candidates = [item for item in candidates if (item.get("parameters") or {}).get("role") == "postprocess"]
        main_candidates = [item for item in candidates if (item.get("parameters") or {}).get("role") != "postprocess"]
        ai = task.get("ai_analysis") or {}
        retrieved_context = task.get("retrieved_context") or []
        candidate_plan = task.get("candidate_plan") or {}

        lines = [
            "# VisionRestore Agent Report",
            "",
            "## Task Overview",
            f"- Task ID: `{task.get('task_id')}`",
            f"- Status: `{task.get('status')}`",
            f"- Mode: `{task.get('mode')}`",
            f"- Priority: `{task.get('priority')}`",
            f"- User goal: {task.get('user_goal') or '(empty)' }",
            "",
            "## Final Result",
            f"- Result type: `{self._role_name(best)}`",
            f"- Model: `{best.get('model_id') or '-'}`",
            f"- Checkpoint: `{best.get('checkpoint_id') or '-'}`",
            f"- Output file ID: `{best.get('output_file_id') or '-'}`",
            f"- Score: `{self._fmt(best.get('score'))}`",
            f"- Runtime: `{self._fmt(best.get('runtime_ms'))} ms`",
            f"- Peak memory: `{self._fmt(best.get('peak_memory_mb'))} MB`",
            f"- Final recommendation: {task.get('final_recommendation') or '-'}",
            "",
        ]

        input_file_id = self._postprocess_input_file_id(best)
        if input_file_id:
            lines += [
                "## Postprocess Adoption",
                "- Adopted: `yes`",
                f"- Operation: `{(best.get('parameters') or {}).get('operation') or (best.get('metrics') or {}).get('postprocess_operation') or 'denoise'}`",
                f"- Postprocess input file ID: `{input_file_id}`",
                "- Note: NAFNet is used only as a real-image denoising postprocess model, not as a low-light enhancement or super-resolution model.",
                "",
            ]
        elif postprocess_candidates:
            lines += [
                "## Postprocess Adoption",
                "- Adopted: `no`",
                "- Note: postprocess candidates exist, but the current final result has been rolled back to a main enhancement candidate.",
                "",
            ]

        if ai:
            lines += [
                "## Multimodal AI Analysis",
                f"- Provider/model: `{ai.get('provider') or '-'} / {ai.get('model') or '-'}`",
                f"- Scene: `{ai.get('scene') or 'unknown'}`",
                f"- Scene confidence: `{self._fmt(ai.get('scene_confidence'))}`",
                f"- Validation passed: `{ai.get('validation_passed')}`",
                f"- Adopted as LLMScore: `{ai.get('adopted')}`",
                f"- Rejection/fallback reason: {ai.get('rejection_reason') or ai.get('failure_reason') or '-' }",
                "",
            ]

        if retrieved_context:
            lines += [
                "## Retrieved Local Context",
                "- Context scope: local allowlisted knowledge only.",
                "- Model role definitions are sent to the configured LLM only as scoring reference standards.",
                "- Knowledge does not contribute an independent planning or final score.",
            ]
            for item in retrieved_context[:8]:
                lines.append(
                    f"- `{item.get('source')}` / {item.get('title')}: "
                    f"score={self._fmt(item.get('score'))}; tags={', '.join(item.get('tags') or [])}"
                )
            lines.append("")

        planned_candidates = candidate_plan.get("candidates") or []
        if planned_candidates:
            lines += [
                "## Candidate Planning",
                "- Formula with valid external advice: `planning_score = 0.5 * local_score + 0.5 * llm_score`.",
                "- Fallback formula: `planning_score = local_score` when the LLM is disabled, invalid, or unavailable.",
                "- Knowledge supplies model definitions to the LLM and is not an independent score.",
                "- Hardware/model/checkpoint readiness is a gate, not a quality score.",
            ]
            for item in planned_candidates:
                lines.append(
                    f"- `{item.get('model_id')}` / `{item.get('checkpoint_id')}`: "
                    f"local={self._fmt(item.get('local_score'))}, "
                    f"llm={self._fmt(item.get('llm_score')) if item.get('llm_score') is not None else '-'}, "
                    f"planning={self._fmt(item.get('planning_score'))}, "
                    f"mode={item.get('planning_mode') or 'local_fallback'}"
                )
                for evidence in item.get("planning_evidence") or []:
                    lines.append(
                        f"  - `{evidence.get('source')}:{evidence.get('signal')}` "
                        f"contribution={self._fmt(evidence.get('contribution'))}; "
                        f"{evidence.get('reason') or '-'}"
                    )
            lines.append("")

        prompts = prompt_registry_snapshot()
        lines += [
            "## Prompt Registry",
            f"- Prompt directory: `{prompts.get('prompt_dir')}`",
        ]
        for prompt in prompts.get("prompts", []):
            lines.append(
                f"- `{prompt.get('filename')}`: sha256=`{str(prompt.get('sha256') or '')[:12]}...`, "
                f"size={prompt.get('size_bytes')} bytes"
            )
        lines.append("")

        lines += [
            "## Agent Decision",
            (task.get("plan") or {}).get("selection_reason") or "No plan recorded.",
            "",
            "## Main Enhancement Candidates",
        ]
        lines.extend(self._candidate_lines(main_candidates))

        lines += ["", "## Postprocess Candidates"]
        lines.extend(self._candidate_lines(postprocess_candidates))

        lines += [
            "",
            "## Safety Notes",
            "- Multimodal AI suggestions are treated as bounded recommendations only.",
            "- Final execution still depends on ModelRegistry, local checkpoint health, hardware checks and routing rules.",
            "- No-reference metrics cannot fully replace human visual judgment.",
        ]
        md_path.write_text("\n".join(lines), encoding="utf-8")
        return {"json": str(json_path), "markdown": str(md_path)}

    def _candidate_lines(self, candidates: list[dict[str, Any]]) -> list[str]:
        if not candidates:
            return ["- None"]
        lines = []
        for cand in candidates:
            metrics = cand.get("metrics") or {}
            layers = metrics.get("score_layers") or {}
            lines.append(
                f"- `{cand.get('model_id')}` / `{cand.get('checkpoint_id') or '-'}`: "
                f"{cand.get('status')}, score={self._fmt(cand.get('score'))}, "
                f"runtime={self._fmt(cand.get('runtime_ms'))} ms, "
                f"memory={self._fmt(cand.get('peak_memory_mb'))} MB, "
                f"file=`{cand.get('output_file_id') or '-'}`, error={cand.get('error') or '-'}"
            )
            if layers:
                lines.append(
                    "  - FinalScore layers: "
                    f"image_quality={self._fmt(layers.get('image_quality'))}, "
                    f"restoration={self._fmt(layers.get('restoration'))}, "
                    f"constraint={self._fmt(layers.get('constraint'))}, "
                    f"stability={self._fmt(layers.get('stability'))}."
                )
                score_evidence = metrics.get("score_evidence") or {}
                image_source = (score_evidence.get("image_quality") or {}).get("source")
                anomalies = (score_evidence.get("stability") or {}).get("detected_anomalies") or []
                lines.append(
                    f"  - Scoring evidence: image_quality_source={image_source or '-'}; "
                    f"stability_anomalies={', '.join(anomalies) or 'none'}; "
                    "planning/knowledge/runtime excluded from final_score."
                )
        return lines

    def _postprocess_input_file_id(self, candidate: dict[str, Any]) -> str | None:
        parameters = candidate.get("parameters") or {}
        metrics = candidate.get("metrics") or {}
        return parameters.get("input_file_id") or metrics.get("postprocess_input_file_id")

    def _role_name(self, candidate: dict[str, Any]) -> str:
        return "postprocess" if (candidate.get("parameters") or {}).get("role") == "postprocess" else "main_enhancement"

    def _fmt(self, value: Any) -> str:
        if value is None:
            return "-"
        if isinstance(value, float):
            return f"{value:.3f}".rstrip("0").rstrip(".")
        return str(value)
