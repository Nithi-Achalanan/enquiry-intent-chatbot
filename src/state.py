from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field, field_validator


AgentMessage = HumanMessage | SystemMessage | ToolMessage | AIMessage


class PendingClarification(BaseModel):
    """The actual clarification question most recently shown to the user."""

    target: str = Field(min_length=1)
    reason: str | None = None
    question: str = Field(min_length=1)
    options: list[str] = Field(default_factory=list)
    supporting_course_ids: list[str] = Field(default_factory=list)
    retrieval_required: bool = False
    attempt: int = Field(default=1, ge=1)

    @field_validator("supporting_course_ids")
    @classmethod
    def normalize_supporting_course_ids(cls, values: list[str]) -> list[str]:
        return normalize_course_ids(values)


class DialogueState(BaseModel):
    """Structured cross-turn context sent alongside the raw conversation."""

    resolved_course_ids: list[str] = Field(default_factory=list)
    last_primary_course_id: str | None = None
    last_related_course_ids: list[str] = Field(default_factory=list)
    active_constraints: dict[str, Any] = Field(default_factory=dict)
    unresolved_references: list[str] = Field(default_factory=list)
    current_goal: str | None = None
    pending_clarification: PendingClarification | None = None
    clarification_count: int = Field(default=0, ge=0)
    last_intent_family: str | None = None
    last_response_mode: str | None = None


def normalize_course_ids(values: list[str] | None) -> list[str]:
    normalized: list[str] = []
    for value in values or []:
        course_id = str(value).strip().upper()
        if course_id and course_id not in normalized:
            normalized.append(course_id)
    return normalized


def merge_dialogue_state(
    previous: DialogueState | dict[str, Any] | None,
    *,
    resolved_course_ids: list[str] | None = None,
    active_constraints: dict[str, Any] | None = None,
    unresolved_references: list[str] | None = None,
    current_goal: str | None = None,
    primary_course_id: str | None = None,
    related_course_ids: list[str] | None = None,
    pending_clarification: PendingClarification | dict[str, Any] | None = None,
    clear_pending_clarification: bool = False,
    clarification_asked: bool = False,
    reset_clarification_count: bool = False,
    last_intent_family: str | None = None,
    last_response_mode: str | None = None,
) -> DialogueState:
    """Merge a model-resolved turn into prior state without interpreting semantics."""
    prior = previous if isinstance(previous, DialogueState) else DialogueState.model_validate(previous or {})
    constraints = dict(prior.active_constraints)
    for key, value in (active_constraints or {}).items():
        if value is None:
            constraints.pop(key, None)
        else:
            constraints[key] = value
    resolved = normalize_course_ids([*prior.resolved_course_ids, *(resolved_course_ids or [])])
    primary = str(primary_course_id).strip().upper() if primary_course_id else prior.last_primary_course_id
    related = normalize_course_ids(related_course_ids) if related_course_ids is not None else prior.last_related_course_ids
    if primary and primary not in resolved:
        resolved.append(primary)
    for course_id in related:
        if course_id not in resolved:
            resolved.append(course_id)
    pending = prior.pending_clarification
    if clear_pending_clarification:
        pending = None
    elif pending_clarification is not None:
        pending = (
            pending_clarification
            if isinstance(pending_clarification, PendingClarification)
            else PendingClarification.model_validate(pending_clarification)
        )
    clarification_count = 0 if reset_clarification_count else prior.clarification_count
    if clarification_asked:
        clarification_count += 1
    return DialogueState(
        resolved_course_ids=resolved,
        last_primary_course_id=primary,
        last_related_course_ids=related,
        active_constraints=constraints,
        unresolved_references=(
            list(unresolved_references)
            if unresolved_references is not None
            else prior.unresolved_references
        ),
        current_goal=current_goal if current_goal is not None else prior.current_goal,
        pending_clarification=pending,
        clarification_count=clarification_count,
        last_intent_family=(
            last_intent_family if last_intent_family is not None else prior.last_intent_family
        ),
        last_response_mode=(
            last_response_mode if last_response_mode is not None else prior.last_response_mode
        ),
    )


class GraphState(TypedDict, total=False):
    """State shared by the guide agent and retrieval subgraph."""

    conversation: list[str]
    query: str
    guide_plan: dict[str, Any]
    dialogue_state: dict[str, Any]
    resolved_course_ids: list[str]
    active_constraints: dict[str, Any]
    unresolved_references: list[str]
    pending_clarification: dict[str, Any] | None
    clarification_count: int
    last_intent_family: str | None
    last_response_mode: str | None
    guide_agent_state_memory: Annotated[list[AgentMessage], operator.add]
    search_agent_state_memory: Annotated[list[AgentMessage], operator.add]
    retrieved_context_raw: list[dict[str, Any]]
    search_attempts: int
    max_search_attempts: int
    tool_call_count: int
    max_tool_calls: int
    tool_call_artifacts: Annotated[list[dict[str, Any]], operator.add]
    tool_call_limit_error: str
    final_answer: str
    final_result: dict[str, Any]
    final_response_mode: str
    primary_course_id: str | None
    referenced_course_ids: list[str]
    related_course_ids: list[str]
    evidence_course_ids: list[str]
    grounding_status: str
    grounding_issues: list[str]
