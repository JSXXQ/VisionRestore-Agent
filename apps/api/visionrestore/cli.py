import argparse
import json
import shutil
from pathlib import Path
from visionrestore.agent.enhancement_agent import EnhancementAgent
from visionrestore.schemas.common import now_iso
from visionrestore.schemas.task import TaskRecord
from visionrestore.services.image_analyzer import ImageAnalyzer
from visionrestore.storage.database import Database
from visionrestore.utils.file_security import safe_image_upload

def analyze(args):
    result = ImageAnalyzer().analyze(args.input)
    print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))

def enhance(args):
    db = Database()
    data = Path(args.input).read_bytes()
    rec = safe_image_upload(Path(args.input).name, data, None)
    db.put_file(rec.file_id, rec.model_dump())
    task = TaskRecord(
        task_id="cli-" + rec.file_id,
        status="queued",
        image_id=rec.file_id,
        user_goal=args.goal or "",
        mode=args.mode,
        priority=args.priority,
        created_at=now_iso(),
        logs=["CLI task created"],
    )
    if args.model:
        from visionrestore.schemas.task import EnhancementPlan
        task.plan = EnhancementPlan(
            input_id=rec.file_id,
            user_goal=args.goal or "",
            mode=args.mode,
            priority=args.priority,
            selected_model=args.model,
            selection_reason=f"CLI 指定模型 {args.model}",
            parameters={"precision": args.precision},
        )
    EnhancementAgent().run(task, rec.model_dump(), lambda t: db.put_task(t.task_id, t.model_dump()))
    if task.best_result and task.best_result.output_file_id and args.output:
        src = Path(__file__).resolve().parents[4] / "data" / "outputs" / f"{task.best_result.output_file_id}.png"
        shutil.copyfile(src, args.output)
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
    p.add_argument("--mode", choices=["auto", "manual", "compare"], default="auto")
    p.add_argument("--priority", choices=["quality", "balanced", "speed"], default="balanced")
    p.add_argument("--model")
    p.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    p.add_argument("--goal", default="")
    p.set_defaults(func=enhance)
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
