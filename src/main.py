"""Small local entry point for the course enquiry chatbot."""

import json
import sys
from typing import Any

from src.graph import build_main_graph


def extract_related_courses(retrieved_context_raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expose only real retrieved courses, preserving retrieval rank/order."""
    related_courses: list[dict[str, Any]] = []
    seen: set[str] = set()
    ranked = sorted(
        (item for item in retrieved_context_raw if isinstance(item, dict) and isinstance(item.get("course"), dict)),
        key=lambda item: int(item.get("rank", 999)),
    )
    for item in ranked:
        course = item["course"]
        if course["course_id"] not in seen:
            related_courses.append(course)
            seen.add(course["course_id"])
    return related_courses


def run_chatbot(query: str, conversation: list[str] | None = None) -> dict[str, Any]:
    initial_state = {
        "conversation": conversation or [],
        "query": query,
        "guide_agent_state_memory": [],
        "search_agent_state_memory": [],
        "retrieved_context_raw": [],
        "search_attempts": 0,
        "max_search_attempts": 4,
    }
    result = build_main_graph().invoke(initial_state)
    return {
        "answer": result.get("final_answer", "ไม่สามารถสร้างคำตอบได้ในขณะนี้ กรุณาลองใหม่อีกครั้งครับ"),
        "related_courses": extract_related_courses(result.get("retrieved_context_raw", [])),
    }


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    query = " ".join(sys.argv[1:]) or input("Course enquiry: ").strip()
    print(json.dumps(run_chatbot(query), ensure_ascii=False, indent=2))
