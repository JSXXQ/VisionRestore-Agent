from functools import lru_cache

from visionrestore.core.config import PROJECT_ROOT

PROMPT_DIR = PROJECT_ROOT / "apps" / "api" / "prompts"


@lru_cache
def load_prompt(filename: str) -> str:
    path = PROMPT_DIR / filename
    return path.read_text(encoding="utf-8").strip()


def multimodal_system_prompt() -> str:
    return load_prompt("multimodal_system_prompt.txt")


def multimodal_analysis_prompt() -> str:
    return load_prompt("multimodal_analysis_prompt.txt")
