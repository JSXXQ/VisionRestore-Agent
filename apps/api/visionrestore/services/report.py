import json
from pathlib import Path

class ReportService:
    def write_reports(self, task: dict, base_path: Path) -> dict:
        base_path.parent.mkdir(parents=True, exist_ok=True)
        json_path = base_path.with_suffix(".json")
        md_path = base_path.with_suffix(".md")
        json_path.write_text(json.dumps(task, ensure_ascii=False, indent=2), encoding="utf-8")
        lines = [
            f"# VisionRestore Agent Report",
            "",
            f"- Task ID: `{task.get('task_id')}`",
            f"- Status: `{task.get('status')}`",
            f"- Mode: `{task.get('mode')}`",
            f"- Priority: `{task.get('priority')}`",
            f"- User goal: {task.get('user_goal') or '(empty)'}",
            "",
            "## Agent Decision",
            task.get("plan", {}).get("selection_reason", "No plan"),
            "",
            "## Candidates",
        ]
        for cand in task.get("candidates", []):
            lines.append(f"- `{cand.get('model_id')}`: {cand.get('status')}, score={cand.get('score')}, error={cand.get('error')}")
        lines += ["", "> 无参考指标不能完全替代人工主观判断。"]
        md_path.write_text("\n".join(lines), encoding="utf-8")
        return {"json": str(json_path), "markdown": str(md_path)}
