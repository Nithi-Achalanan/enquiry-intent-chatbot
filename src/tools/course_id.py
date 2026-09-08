"""Exact course-ID lookup tool."""

import json
from pathlib import Path

from langchain_core.tools import tool


DATA_PATH = Path(__file__).resolve().parents[2] / "local_data" / "course.jsonl"


def load_courses() -> list[dict]:
    return json.loads(DATA_PATH.read_text(encoding="utf-8"))


def find_course_by_id(course_id: str) -> dict | None:
    normalized_id = course_id.strip().upper()
    return next((course for course in load_courses() if course["course_id"].upper() == normalized_id), None)


@tool("course_id", response_format="content_and_artifact")
def course_id_tool(course_id: str) -> tuple[str, dict]:
    """Return one exact course record or a clear not-found artifact."""
    course = find_course_by_id(course_id)
    artifact = course or {"course_id": course_id.strip().upper(), "found": False}
    if course is None:
        return f"Course ID {artifact['course_id']} was not found.", artifact
    return json.dumps(course, ensure_ascii=False), artifact
