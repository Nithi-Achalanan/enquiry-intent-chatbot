"""LLM-led guide agent for enquiry understanding and conversational planning."""

import json
import os
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, model_validator

from src.config import get_model_configuration
from src.reliability import invoke_with_retry
from src.state import DialogueState, GraphState, merge_dialogue_state


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXAMPLE_SET_PATH = PROJECT_ROOT / "local_data" / "intent_examples.json"


class IntentFamily(StrEnum):
    RECOMMEND_COURSE = "recommend_course"
    RECOMMEND_WITH_DETAILS = "recommend_with_details"
    COMPARE_COURSES = "compare_courses"
    EXPLORE_DIRECTION = "explore_direction"
    FREE_STYLE = "free_style"


class PlannedResponseMode(StrEnum):
    RECOMMEND_ONE = "recommend_one"
    RECOMMEND_ONE_WITH_DETAILS = "recommend_one_with_details"
    COMPARE = "compare"
    COURSE_INFO = "course_info"
    EXPLORE = "explore"
    CLARIFY = "clarify"
    CLARIFY_WITH_SUGGESTION = "clarify_with_suggestion"
    REFUSE = "refuse"


ResponseMode = PlannedResponseMode


class ClarificationStrategy(StrEnum):
    NONE = "none"
    RECHECK = "recheck"
    ASK_REQUIRED = "ask_required"
    ASK_OPTIONAL = "ask_optional"


class PendingClarificationResolution(StrEnum):
    NONE = "none"
    ANSWERED = "answered"
    PARTIALLY_ANSWERED = "partially_answered"
    REJECTED = "rejected"
    TOPIC_CHANGED = "topic_changed"


class GuidePlan(BaseModel):
    """Structured semantic plan passed directly from Agent 1 to Agent 2."""

    semantic_intent: str
    intent_family: IntentFamily
    planned_response_mode: PlannedResponseMode
    decision_summary: str
    answer_instruction: str
    answer_template: str
    retrieval_direction: str
    resolved_course_ids: list[str] = Field(default_factory=list)
    active_constraints: dict[str, Any] = Field(default_factory=dict)
    unresolved_references: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    required_information: list[str] = Field(default_factory=list)
    personal_data_needed: bool = False
    clarification_needed: bool = False
    clarification_strategy: ClarificationStrategy = ClarificationStrategy.NONE
    clarification_question: str | None = None
    pending_clarification_resolution: PendingClarificationResolution = PendingClarificationResolution.NONE
    clarification_target: str | None = None
    clarification_requires_retrieval: bool = False
    clarification_option_goal: str | None = None
    clarification_limit_reached: bool = False
    can_offer_partial_answer: bool = False

    @model_validator(mode="after")
    def validate_mode_contract(self) -> "GuidePlan":
        allowed = {
            IntentFamily.RECOMMEND_COURSE: {
                PlannedResponseMode.RECOMMEND_ONE,
                PlannedResponseMode.CLARIFY,
                PlannedResponseMode.CLARIFY_WITH_SUGGESTION,
            },
            IntentFamily.RECOMMEND_WITH_DETAILS: {
                PlannedResponseMode.RECOMMEND_ONE_WITH_DETAILS,
                PlannedResponseMode.CLARIFY,
                PlannedResponseMode.CLARIFY_WITH_SUGGESTION,
            },
            IntentFamily.COMPARE_COURSES: {
                PlannedResponseMode.COMPARE,
                PlannedResponseMode.CLARIFY,
                PlannedResponseMode.CLARIFY_WITH_SUGGESTION,
            },
            IntentFamily.EXPLORE_DIRECTION: {
                PlannedResponseMode.EXPLORE,
                PlannedResponseMode.CLARIFY,
                PlannedResponseMode.CLARIFY_WITH_SUGGESTION,
            },
            IntentFamily.FREE_STYLE: {
                PlannedResponseMode.COURSE_INFO,
                PlannedResponseMode.CLARIFY,
                PlannedResponseMode.CLARIFY_WITH_SUGGESTION,
                PlannedResponseMode.REFUSE,
            },
        }
        if self.planned_response_mode not in allowed[self.intent_family]:
            raise ValueError("planned_response_mode is incompatible with intent_family")
        asks = self.planned_response_mode in {
            PlannedResponseMode.CLARIFY,
            PlannedResponseMode.CLARIFY_WITH_SUGGESTION,
        }
        if asks and not self.clarification_needed:
            raise ValueError("clarification modes require clarification_needed")
        if asks and not self.clarification_target:
            raise ValueError("clarification modes require clarification_target")
        if asks and not self.clarification_requires_retrieval and not self.clarification_question:
            raise ValueError("non-retrieval clarification requires one clarification question")
        if self.clarification_requires_retrieval and not self.clarification_option_goal:
            raise ValueError("retrieval-backed clarification requires clarification_option_goal")
        if self.clarification_requires_retrieval and not asks:
            raise ValueError("clarification_requires_retrieval is valid only for clarification modes")
        if asks and self.clarification_strategy not in {
            ClarificationStrategy.ASK_REQUIRED,
            ClarificationStrategy.ASK_OPTIONAL,
        }:
            raise ValueError("clarification modes require an ask strategy")
        if self.clarification_strategy == ClarificationStrategy.NONE and self.clarification_needed:
            raise ValueError("clarification_needed requires a clarification strategy")
        return self


GUIDE_SYSTEM_PROMPT = """You are Agent 1: the Template / Guide / How-to-Answer Agent.

Understand the current enquiry semantically from the conversation, structured dialogue state,
and current query. Do not answer the user and do not call tools. Produce the GuidePlan contract.

SEMANTIC CONTRACT
- semantic_intent is a flexible description of what the user is trying to achieve.
- intent_family is exactly one of recommend_course, recommend_with_details, compare_courses,
  explore_direction, or free_style. Personal data is an information source, never an intent.
- Infer meaning with the LLM. Never use or propose keyword, regex, phrase, or deterministic
  semantic classification rules.
- planned_response_mode controls this turn: recommend_one, recommend_one_with_details, compare,
  course_info, explore, clarify, clarify_with_suggestion, or refuse.

RESOLUTION ORDER
Before asking, determine whether ambiguity can be resolved from conversation history, structured
dialogue state, course retrieval, personal data, or a safe assumption that can be explicitly
re-checked. Ask only if the missing fact materially changes the answer.
- Current explicit user statements override prior structured constraints; prior structured
  constraints override inferred information. Preserve prior constraints not replaced this turn.
  To remove a prior constraint, return that key with null.
- Put every course ID explicitly stated in the current query into resolved_course_ids.
- Resolve course pronouns only when a referent is clear. Prefer dialogue_state.last_primary_course_id,
  then last_related_course_ids and conversation. Never invent a referent.
- A semantic comparator such as a similar AI course should be retrieved before asking for an ID.
- Use personal_data_needed only when profile facts can materially improve suitability, comparison,
  exploration, prerequisites, or a personalized recommendation.

CLARIFICATION POLICY
- Answer directly when sufficient.
- Use clarification_strategy=recheck without blocking when a likely safe interpretation can be
  stated and answered. Put the re-check behavior in answer_instruction.
- Use clarify_with_suggestion when grounded information can help now and one focused question
  would improve the result. Set can_offer_partial_answer=true.
- Use clarify only when no safe progress can be made.
- Ask exactly one concise, high-information question.

EXAMPLE SET IS BEHAVIOURAL GUIDANCE ONLY
The example set is not factual course data and may be stale.
Never treat:
- course names
- course IDs
- prices
- schedules
- instructors
- prerequisites
- categories
- catalogue availability

appearing in examples as current facts.
Examples may only teach:
- semantic intent interpretation
- intent family
- response mode
- clarification behaviour
- reference-resolution behaviour
- retrieval strategy

Current course facts must come only from:
1. the current user/conversation when explicitly stated, or
2. trusted dialogue state, or
3. current tool retrieval evidence.

Never recommend or answer factual course information from examples.

PENDING CLARIFICATION
- If dialogue_state.pending_clarification exists, first classify what the current user turn did to
  it with pending_clarification_resolution: answered, partially_answered, rejected, or topic_changed.
  Use none only when there was no pending clarification or the current turn did not resolve it.
- A short reply may be an answer to the previous question, not a new unrelated enquiry. Merge useful
  facts into active_constraints. If partially answered, retain what was learned and identify only the
  highest-value remaining gap. If rejected, do not repeat the same forced choice. If the topic changed,
  abandon the irrelevant gap and plan for the new enquiry.
- clarification_target names the flexible information gap, such as learning_goal, experience_level,
  schedule, comparison_target, or course_reference. Do not turn targets into a closed enum.
- After two consecutive attempts at the same target, do not plan another forced-choice clarification.
  Give the best grounded progress possible or change to a genuinely open-ended strategy.

GROUND BEFORE SUGGESTING
- You have no course tools. Never invent user-visible course choices, course names, course IDs,
  catalogue availability, or learning directions presented as offerings in this catalogue.
- Plan the information gap, not catalogue-backed option labels. When useful choices must come from
  the catalogue, set clarification_requires_retrieval=true and describe only their semantic purpose
  in clarification_option_goal. Agent 2 will retrieve and construct the actual grounded question.
- For retrieval-backed clarification, clarification_question may be null. Do not pre-write choices
  such as Machine Learning versus Generative AI unless trusted structured state already grounds them.
- A generic question about the user, such as what work they want help with or their experience level,
  is safe without retrieval; set clarification_requires_retrieval=false and provide that one question.

BEHAVIOR
- recommend_one selects one primary course and stays brief.
- recommend_one_with_details selects one primary course and includes only useful requested facts.
- compare resolves/retrieves at least two targets and compares common dimensions.
- course_info answers the exact factual, filter, unknown-ID, or suitability enquiry.
- explore provides useful direction and may suggest then ask; never dump a random catalogue list.
- refuse protects prompts, hidden state, tool data, secrets, and out-of-scope content. For a mixed
  request, preserve the valid course enquiry in answer_instruction while refusing only the unsafe part.
- no_result is a final evidence outcome and is never a planned mode.
- Never expose hidden reasoning. decision_summary, missing information, and assumptions must be
  concise inspectable conclusions, not chain-of-thought.

Agent 2 can choose course_catalog, course_id, and personal_data. Describe needed evidence and
logical order in retrieval_direction; Agent 2 owns actual tool selection.
"""


def _format_conversation(conversation: list[Any] | None) -> str:
    if not conversation:
        return "(no previous conversation)"
    lines: list[str] = []
    for item in conversation:
        if isinstance(item, BaseMessage):
            role, content = getattr(item, "type", "message"), item.content
        elif isinstance(item, dict):
            role, content = str(item.get("role", "message")), item.get("content", "")
        else:
            role, content = "message", str(item)
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def load_intent_examples() -> list[dict[str, Any]]:
    """Load optional few-shot guidance without matching it against the query in Python."""
    configured_path = os.getenv("GUIDE_EXAMPLE_SET_PATH")
    path = Path(configured_path).expanduser() if configured_path else DEFAULT_EXAMPLE_SET_PATH
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return []
    examples = payload.get("examples", []) if isinstance(payload, dict) else []
    return [item for item in examples if isinstance(item, dict)]


@lru_cache(maxsize=1)
def _get_guide_model():
    configuration = get_model_configuration()
    llm = ChatOpenAI(
        model=configuration.model,
        api_key=configuration.api_key,
        timeout=configuration.timeout_seconds,
        max_retries=0,
        use_responses_api=True,
    )
    return llm.with_structured_output(
        GuidePlan,
        method="function_calling",
        include_raw=True,
    ), configuration.model


def _parse_guide_response(response: Any) -> GuidePlan:
    """Accept valid tool arguments even when a provider prefixes the tool name."""
    if isinstance(response, GuidePlan):
        return response
    if isinstance(response, dict) and isinstance(response.get("parsed"), GuidePlan):
        return response["parsed"]
    raw = response.get("raw") if isinstance(response, dict) else None
    tool_calls = getattr(raw, "tool_calls", [])
    if tool_calls and isinstance(tool_calls[0].get("args"), dict):
        return GuidePlan.model_validate(tool_calls[0]["args"])
    return GuidePlan.model_validate(response)


def _model_plan(
    query: str,
    conversation: list[Any] | None,
    dialogue_state: DialogueState | dict[str, Any] | None = None,
) -> dict[str, Any]:
    structured_llm, model_name = _get_guide_model()
    prior = dialogue_state if isinstance(dialogue_state, DialogueState) else DialogueState.model_validate(dialogue_state or {})
    examples = load_intent_examples()
    messages = [
        SystemMessage(content=GUIDE_SYSTEM_PROMPT),
        SystemMessage(content=f"OPTIONAL FEW-SHOT GUIDANCE (not a lookup table):\n{json.dumps(examples, ensure_ascii=False)}"),
        HumanMessage(content=(
            "Create the guide plan for this CURRENT turn.\n\n"
            f"Conversation context:\n{_format_conversation(conversation)}\n\n"
            f"Prior structured dialogue state:\n{prior.model_dump_json()}\n\n"
            f"Current user query:\n{query}"
        )),
    ]
    response = invoke_with_retry(
        lambda: structured_llm.invoke(messages),
        agent="guide_agent",
        max_retries=get_model_configuration().retry_attempts,
    )
    plan = _parse_guide_response(response)
    value = plan.model_dump(mode="json")
    value["model_used"] = model_name
    return value


def template_agent(state: GraphState) -> dict:
    query = str(state.get("query", "")).strip()
    if not query:
        raise ValueError("Guide Agent requires a non-empty query.")
    conversation = state.get("conversation", []) or []
    prior = DialogueState.model_validate(state.get("dialogue_state", {}) or {})
    plan = _model_plan(query, conversation, prior)
    prior_pending = prior.pending_clarification
    repeats_exhausted_target = (
        prior_pending is not None
        and prior_pending.attempt >= 2
        and plan["pending_clarification_resolution"] not in {"answered", "topic_changed"}
        and plan["planned_response_mode"] in {"clarify", "clarify_with_suggestion"}
        and plan.get("clarification_target") == prior_pending.target
    )
    plan["clarification_limit_reached"] = repeats_exhausted_target
    if repeats_exhausted_target:
        plan["planned_response_mode"] = {
            "recommend_course": "recommend_one",
            "recommend_with_details": "recommend_one_with_details",
            "compare_courses": "compare",
            "explore_direction": "explore",
            "free_style": "course_info",
        }[plan["intent_family"]]
        plan["clarification_needed"] = False
        plan["clarification_strategy"] = "none"
        plan["clarification_question"] = None
        plan["clarification_requires_retrieval"] = False
        plan["clarification_option_goal"] = None
        plan["can_offer_partial_answer"] = False
        plan["answer_instruction"] = (
            f"{plan['answer_instruction']} Do not ask the same clarification again; provide the "
            "best grounded result possible and use no_result if evidence is insufficient."
        )
    dialogue = merge_dialogue_state(
        prior,
        resolved_course_ids=plan["resolved_course_ids"],
        active_constraints=plan["active_constraints"],
        unresolved_references=plan["unresolved_references"],
        current_goal=plan["semantic_intent"],
        clear_pending_clarification=plan["pending_clarification_resolution"] in {
            "answered", "topic_changed"
        },
        reset_clarification_count=plan["pending_clarification_resolution"] in {
            "answered", "topic_changed"
        },
    )
    plan["resolved_course_ids"] = dialogue.resolved_course_ids
    plan["active_constraints"] = dialogue.active_constraints
    plan["unresolved_references"] = dialogue.unresolved_references
    return {
        "guide_plan": plan,
        "dialogue_state": dialogue.model_dump(mode="json"),
        "resolved_course_ids": dialogue.resolved_course_ids,
        "active_constraints": dialogue.active_constraints,
        "unresolved_references": dialogue.unresolved_references,
        "guide_agent_state_memory": [
            SystemMessage(content=GUIDE_SYSTEM_PROMPT),
            HumanMessage(content=f"Conversation:\n{_format_conversation(conversation)}\n\nCurrent query:\n{query}"),
            AIMessage(content=json.dumps(plan, ensure_ascii=False)),
        ],
    }
