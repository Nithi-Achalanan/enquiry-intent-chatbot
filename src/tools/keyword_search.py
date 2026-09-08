"""Inspectable local keyword retrieval over the course catalogue."""

import json
import re
from pathlib import Path
from typing import Any, TypedDict

from langchain_core.tools import tool
from rapidfuzz import fuzz


DATA_PATH = Path(__file__).resolve().parents[2] / "local_data" / "course.jsonl"


class SearchResult(TypedDict):
    course: dict[str, Any]
    matched_keywords: list[str]
    matched_keyword_count: int
    score: float
    rank: int


def load_courses() -> list[dict[str, Any]]:
    return json.loads(DATA_PATH.read_text(encoding="utf-8"))


def _searchable_text(course: dict[str, Any]) -> str:
    values = [str(course.get(field, "")) for field in (
        "course_id", "course_name", "instructor", "description", "category", "level", "target_audience"
    )]
    values.extend(str(value) for value in course.get("keywords", []))
    values.extend(str(value) for value in course.get("prerequisites", []))
    return " ".join(values).lower()


def _normalize_keywords(keywords: list[str]) -> list[str]:
    return list(dict.fromkeys(keyword.strip().lower() for keyword in keywords if keyword and keyword.strip()))[:8]


def multiple_keyword_search(keywords: list[str], top_k: int = 5, threshold: int = 65) -> list[SearchResult]:
    """Rank courses by matched-keyword count, then average fuzzy score."""
    normalized_keywords = _normalize_keywords(keywords)
    if not normalized_keywords:
        return []

    ranked: list[dict[str, Any]] = []
    for course in load_courses():
        searchable_text = _searchable_text(course)
        matched_keywords: list[str] = []
        scores: list[float] = []
        for keyword in normalized_keywords:
            score = float(fuzz.partial_ratio(keyword, searchable_text))
            if keyword in searchable_text or score >= threshold:
                matched_keywords.append(keyword)
                scores.append(100.0 if keyword in searchable_text else score)
        if matched_keywords:
            ranked.append({
                "course": course,
                "matched_keywords": matched_keywords,
                "matched_keyword_count": len(matched_keywords),
                "score": round(sum(scores) / len(scores), 2),
            })

    ranked.sort(key=lambda result: (result["matched_keyword_count"], result["score"]), reverse=True)
    return [{**result, "rank": index} for index, result in enumerate(ranked[:top_k], start=1)]


@tool("keyword_search", response_format="content_and_artifact")
def keyword_search_tool(keywords: list[str], top_k: int = 5) -> tuple[str, list[SearchResult]]:
    """Search local course data and return explainable ranked result artifacts."""
    results = multiple_keyword_search(keywords, top_k=top_k)
    return json.dumps(results, ensure_ascii=False), results


def course_ids_in_query(query: str) -> list[str]:
    return list(dict.fromkeys(re.findall(r"\b[A-Za-z]{2,}\d{2,}\b", query.upper())))
