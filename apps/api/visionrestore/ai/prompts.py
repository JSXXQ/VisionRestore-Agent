import hashlib
from functools import lru_cache

from visionrestore.core.config import PROJECT_ROOT

PROMPT_DIR = PROJECT_ROOT / "apps" / "api" / "prompts"


@lru_cache
def load_prompt(filename: str) -> str:
    path = PROMPT_DIR / filename
    return path.read_text(encoding="utf-8").strip()


@lru_cache
def prompt_metadata(filename: str) -> dict:
    path = PROMPT_DIR / filename
    content = path.read_bytes()
    return {
        "filename": filename,
        "path": str(path),
        "sha256": hashlib.sha256(content).hexdigest(),
        "size_bytes": len(content),
    }


def prompt_registry_snapshot() -> dict:
    files = [
        "multimodal_system_prompt.txt",
        "multimodal_analysis_prompt.txt",
        "report_summary_prompt.txt",
    ]
    return {"prompt_dir": str(PROMPT_DIR), "prompts": [prompt_metadata(filename) for filename in files]}


def multimodal_system_prompt() -> str:
    return load_prompt("multimodal_system_prompt.txt")


def multimodal_analysis_prompt() -> str:
    return load_prompt("multimodal_analysis_prompt.txt")


def multimodal_prompt_metadata() -> dict:
    return {
        "system": prompt_metadata("multimodal_system_prompt.txt"),
        "analysis": prompt_metadata("multimodal_analysis_prompt.txt"),
    }
