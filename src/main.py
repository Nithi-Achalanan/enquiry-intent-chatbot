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
from src.state import DialogueState


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
    dialogue_state: DialogueState | None = None


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
    dialogue_state: DialogueState


def _course_catalog() -> dict[str, dict[str, Any]]:
    """Read the catalogue only to verify that returned artifacts are real courses."""
    courses = json.loads(COURSE_DATA_PATH.read_text(encoding="utf-8"))
    return {
        str(course.get("course_id", "")).upper(): course
        for course in courses
        if isinstance(course, dict) and course.get("course_id")
    }


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, BaseModel):
        return value.model_dump()
    return {}


def _evidence_course_ids(retrieved_context_raw: list[dict[str, Any]]) -> set[str]:
    """Collect course IDs from course-tool evidence, excluding unrelated artifacts."""
    course_ids: set[str] = set()
    for artifact in retrieved_context_raw:
        if not isinstance(artifact, dict):
            continue
        tool_name = artifact.get("tool_name")
        if tool_name not in {"course_catalog", "course_id"}:
            continue
        course = artifact.get("course")
        if isinstance(course, dict) and course.get("course_id"):
            course_ids.add(str(course["course_id"]).strip().upper())
        courses = artifact.get("courses")
        if isinstance(courses, list):
            course_ids.update(
                str(item["course_id"]).strip().upper()
                for item in courses
                if isinstance(item, dict) and item.get("course_id")
            )
    return course_ids


def extract_related_courses(
    final_result: Any,
    retrieved_context_raw: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Materialize only Agent 2's selected, evidence-backed related courses."""
    structured_result = _as_dict(final_result)
    if structured_result.get("final_response_mode") in {"no_result", "refuse", "clarify"}:
        return []
    selected_ids = structured_result.get("related_course_ids", [])
    if not isinstance(selected_ids, list):
        return []

    try:
        catalog = _course_catalog()
    except (OSError, ValueError, json.JSONDecodeError):
        logger.warning("Unable to validate retrieved courses against the local catalogue")
        return []

    evidence_ids = _evidence_course_ids(retrieved_context_raw)
    related_courses: list[dict[str, Any]] = []
    seen: set[str] = set()
    for selected_id in selected_ids:
        course_id = str(selected_id).strip().upper()
        course = catalog.get(course_id)
        if not course or course_id not in evidence_ids or course_id in seen:
            continue
        related_courses.append(course)
        seen.add(course_id)
    return related_courses


def _response_dialogue_state(result: dict[str, Any], previous: DialogueState) -> DialogueState:
    """Prefer graph-produced state and provide a compatibility fallback during migration."""
    base = previous
    graph_state = result.get("dialogue_state")
    if graph_state is not None:
        try:
            base = DialogueState.model_validate({
                **previous.model_dump(),
                **_as_dict(graph_state),
            })
        except (TypeError, ValueError):
            logger.warning("Graph returned an invalid dialogue state; using derived state")

    plan = _as_dict(result.get("guide_plan"))
    final_result = _as_dict(result.get("final_result"))
    primary_course_id = final_result.get("primary_course_id", result.get("primary_course_id"))
    related_course_ids = final_result.get("related_course_ids", result.get("related_course_ids"))
    value = base.model_dump()
    value.update({
        "resolved_course_ids": result.get(
            "resolved_course_ids",
            plan.get("resolved_course_ids", base.resolved_course_ids),
        ),
        "last_primary_course_id": primary_course_id or base.last_primary_course_id,
        "last_related_course_ids": (
            related_course_ids
            if isinstance(related_course_ids, list)
            else base.last_related_course_ids
        ),
        "active_constraints": result.get(
            "active_constraints",
            plan.get("active_constraints", base.active_constraints),
        ),
        "unresolved_references": result.get(
            "unresolved_references",
            plan.get("unresolved_references", base.unresolved_references),
        ),
        "current_goal": plan.get("semantic_intent", base.current_goal),
        "last_intent_family": plan.get("intent_family", base.last_intent_family),
        "last_response_mode": final_result.get(
            "final_response_mode",
            result.get("final_response_mode", base.last_response_mode),
        ),
    })
    return DialogueState.model_validate(value)


def adapt_conversation(messages: list[ConversationMessage]) -> list[str]:
    """Convert API messages to the existing GraphState.conversation format."""
    return [f"{message.role}: {message.content.strip()}" for message in messages if message.content.strip()]


def run_chatbot(
    query: str,
    conversation: list[str] | None = None,
    dialogue_state: DialogueState | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute the existing graph without moving agent logic into the API layer."""
    prior_dialogue_state = DialogueState.model_validate(dialogue_state or {})
    initial_state = {
        "conversation": conversation or [],
        "query": query,
        "dialogue_state": prior_dialogue_state.model_dump(),
        "resolved_course_ids": prior_dialogue_state.resolved_course_ids,
        "active_constraints": prior_dialogue_state.active_constraints,
        "unresolved_references": prior_dialogue_state.unresolved_references,
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
        "related_courses": extract_related_courses(
            result.get("final_result"),
            result.get("retrieved_context_raw", []),
        ),
        "dialogue_state": _response_dialogue_state(result, prior_dialogue_state).model_dump(),
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
        result = await run_in_threadpool(
            run_chatbot,
            query,
            adapt_conversation(request.conversation),
            request.dialogue_state,
        )
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
