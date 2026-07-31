import argparse
import json
import shutil
import time
from pathlib import Path
from visionrestore.agent.enhancement_agent import EnhancementAgent
from visionrestore.schemas.common import now_iso
from visionrestore.schemas.task import EnhancementPlan, TaskRecord
from visionrestore.services.image_analyzer import ImageAnalyzer
from visionrestore.storage.database import Database
from visionrestore.utils.file_security import safe_image_upload


def analyze(args):
    result = ImageAnalyzer().analyze(args.input)
    print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))


def _run_task(input_path: str, output_path: str | None, request: str, mode: str, priority: str, model: str | None = None, weight: str | None = None) -> TaskRecord:
    db = Database()
    data = Path(input_path).read_bytes()
    rec = safe_image_upload(Path(input_path).name, data, None)
    db.put_file(rec.file_id, rec.model_dump())
    task = TaskRecord(
        task_id="cli-" + rec.file_id,
        status="queued",
        image_id=rec.file_id,
        user_goal=request or "",
        mode=mode,
        priority=priority,
        created_at=now_iso(),
        logs=["CLI task created"],
    )
    if model:
        task.plan = EnhancementPlan(
            input_id=rec.file_id,
            user_goal=request or "",
            mode=mode,
            priority=priority,
            selected_model=model,
            selected_checkpoint=weight,
            selection_reason=f"CLI 指定模型 {model}" + (f":{weight}" if weight else ""),
            parameters={"checkpoint_id": weight, "device": "cuda", "precision": "fp32"},
        )
    EnhancementAgent().run(task, rec.model_dump(), lambda t: db.put_task(t.task_id, t.model_dump()))
    if task.best_result and task.best_result.output_file_id and output_path:
        src = Path(__file__).resolve().parents[3] / "data" / "outputs" / f"{task.best_result.output_file_id}.png"
        shutil.copyfile(src, output_path)
    return task


def enhance(args):
    task = _run_task(args.input, args.output, args.request, args.mode, args.priority, args.model, args.weight)
    print(json.dumps(task.model_dump(), ensure_ascii=False, indent=2))
    if task.status != "completed":
        raise SystemExit(2)


def compare(args):
    # The first candidate seeds the manual request; selected additional candidates are recorded in request text for routing visibility.
    first = args.candidates[0]
    model, weight = first.split(":", 1) if ":" in first else (first, None)
    req = "compare " + " ".join(args.candidates)
    task = _run_task(args.input, args.output, req, "compare", args.priority, model, weight)
    print(json.dumps(task.model_dump(), ensure_ascii=False, indent=2))
    if task.status != "completed":
        raise SystemExit(2)


def main():
    parser = argparse.ArgumentParser(prog="visionrestore")
    sub = parser.add_subparsers(required=True)
    p = sub.add_parser("analyze")
    p.add_argument("--input", required=True)
    p.set_defaults(func=analyze)
    p = sub.add_parser("enhance")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--request", "--goal", default="")
    p.add_argument("--mode", choices=["auto", "manual", "compare"], default="auto")
    p.add_argument("--priority", choices=["quality", "balanced", "speed"], default="balanced")
    p.add_argument("--model")
    p.add_argument("--weight", "--checkpoint-id")
    p.set_defaults(func=enhance)
    p = sub.add_parser("compare")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--priority", choices=["quality", "balanced", "speed"], default="balanced")
    p.add_argument("--candidates", nargs="+", required=True)
    p.set_defaults(func=compare)
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()

