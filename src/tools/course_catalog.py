"""Read the factual course catalogue without filtering or ranking it."""

import json
from pathlib import Path

from langchain_core.tools import tool


DATA_PATH = Path(__file__).resolve().parents[2] / "local_data" / "course.jsonl"


def load_course_catalog() -> list[dict]:
    """Return every current course record exactly as stored."""
    return json.loads(DATA_PATH.read_text(encoding="utf-8"))


@tool("course_catalog", response_format="content_and_artifact")
def course_catalog_tool() -> tuple[str, list[dict]]:
    """Return every course with detailed descriptions for semantic assessment."""
    courses = load_course_catalog()
    return json.dumps(courses, ensure_ascii=False), courses
