"""FastAPI adapter and local entry point for the course enquiry chatbot."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.graph import graph
from src.reliability import ModelInvocationError


logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = PROJECT_ROOT / "frontend"
COURSE_DATA_PATH = PROJECT_ROOT / "local_data" / "course.jsonl"


class ConversationMessage(BaseModel):
    """A prior browser message adapted to the graph's string conversation state."""

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    query: str = Field(min_length=1)
    conversation: list[ConversationMessage] = Field(default_factory=list)


class RelatedCourse(BaseModel):
    course_id: str
    course_name: str
    description: str | None = None
    instructor: str | None = None
    category: str | None = None
    level: str | None = None
    duration: str | None = None
    schedule: str | None = None
    price: float | None = None


class ChatResponse(BaseModel):
    answer: str
    related_courses: list[RelatedCourse] = Field(default_factory=list)


def _course_catalog() -> dict[str, dict[str, Any]]:
    """Read the catalogue only to verify that returned artifacts are real courses."""
    courses = json.loads(COURSE_DATA_PATH.read_text(encoding="utf-8"))
    return {
        str(course.get("course_id", "")).upper(): course
        for course in courses
        if isinstance(course, dict) and course.get("course_id")
    }


def _rank(item: dict[str, Any]) -> int:
    try:
        return int(item.get("rank", 999))
    except (TypeError, ValueError):
        return 999


def extract_related_courses(retrieved_context_raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return deduplicated retrieved course artifacts in their retrieval order."""
    try:
        catalog = _course_catalog()
    except (OSError, ValueError, json.JSONDecodeError):
        logger.warning("Unable to validate retrieved courses against the local catalogue")
        return []

    related_courses: list[dict[str, Any]] = []
    seen: set[str] = set()
    artifacts = sorted(
        (
            item
            for item in retrieved_context_raw
            if isinstance(item, dict) and isinstance(item.get("course"), dict)
        ),
        key=_rank,
    )
    for artifact in artifacts:
        course_id = str(artifact["course"].get("course_id", "")).upper()
        course = catalog.get(course_id)
        if not course or course_id in seen:
            continue
        related_courses.append(course)
        seen.add(course_id)
    return related_courses


def adapt_conversation(messages: list[ConversationMessage]) -> list[str]:
    """Convert API messages to the existing GraphState.conversation format."""
    return [f"{message.role}: {message.content.strip()}" for message in messages if message.content.strip()]


def run_chatbot(query: str, conversation: list[str] | None = None) -> dict[str, Any]:
    """Execute the existing graph without moving agent logic into the API layer."""
    initial_state = {
        "conversation": conversation or [],
        "query": query,
        "guide_agent_state_memory": [],
        "search_agent_state_memory": [],
        "retrieved_context_raw": [],
        "search_attempts": 0,
        "max_search_attempts": 5,
        "tool_call_count": 0,
        "max_tool_calls": 5,
        "tool_call_artifacts": [],
    }
    result = graph.invoke(initial_state)
    if result.get("tool_call_limit_error"):
        raise RuntimeError(result["tool_call_limit_error"])
    return {
        "answer": result.get("final_answer", "ไม่สามารถสร้างคำตอบได้ในขณะนี้ กรุณาลองใหม่อีกครั้งครับ"),
        "related_courses": extract_related_courses(result.get("retrieved_context_raw", [])),
    }


app = FastAPI(title="Enquiry Intent Chatbot")
app.mount("/static", StaticFiles(directory=FRONTEND_DIR, check_dir=False), name="static")


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="Query must not be blank.")

    try:
        result = await run_in_threadpool(run_chatbot, query, adapt_conversation(request.conversation))
    except ModelInvocationError as error:
        logger.warning("chat_model_unavailable diagnostic=%s", error.artifact())
        if not error.diagnostic.retryable:
            raise HTTPException(
                status_code=502,
                detail="ระบบ AI ไม่สามารถประมวลผลคำขอได้ในขณะนี้",
            ) from None
        raise HTTPException(
            status_code=503,
            detail="ระบบ AI กำลังไม่พร้อมใช้งาน กรุณาลองใหม่อีกครั้ง",
            headers={"Retry-After": "2"},
        ) from None
    except Exception:
        logger.exception("Chat graph execution failed")
        raise HTTPException(status_code=500, detail="Unable to process the enquiry.") from None

    return ChatResponse(**result)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    query = " ".join(sys.argv[1:]) or input("Course enquiry: ").strip()
    print(json.dumps(run_chatbot(query), ensure_ascii=False, indent=2))
