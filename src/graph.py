"""Existing main graph and retrieval subgraph for the course-enquiry POC."""

import json
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from src.agents.data_retriever import search_agent
from src.agents.template_design import template_agent
from src.state import GraphState
from src.tools.course_id import find_course_by_id
from src.tools.course_catalog import load_course_catalog
from src.tools.personal_data import load_personal_data


def should_continue(state: GraphState, memory_key: str = "search_agent_state_memory") -> str:
    """Route the Search Agent's latest tool call to its existing tool node."""
    messages = state.get(memory_key, [])
    if not messages:
        return END
    tool_calls = getattr(messages[-1], "tool_calls", None)
    if not tool_calls:
        return END
    if state.get("tool_call_count", 0) >= state.get("max_tool_calls", 5):
        return "tool_call_limit_error"
    return {
        "course_catalog": "course_catalog_tool",
        "course_id": "course_id_tool",
        "personal_data": "personal_data_tool",
    }.get(tool_calls[0]["name"], END)


def _pending_tool_call(state: GraphState, tool_name: str) -> dict[str, Any] | None:
    for message in reversed(state.get("search_agent_state_memory", [])):
        if isinstance(message, AIMessage):
            return next((call for call in message.tool_calls if call["name"] == tool_name), None)
    return None


def _tool_result(state: GraphState, tool_name: str, artifact: Any, content: str) -> dict:
    tool_call = _pending_tool_call(state, tool_name)
    if tool_call is None:
        return {"search_agent_state_memory": []}
    call_count = state.get("tool_call_count", 0) + 1
    call_artifact = {
        "sequence": call_count,
        "tool_name": tool_name,
        "tool_input": tool_call.get("args", {}),
        "executed": True,
        "max_calls": state.get("max_tool_calls", 5),
    }
    return {
        "search_agent_state_memory": [ToolMessage(
            content=content,
            tool_call_id=tool_call["id"],
            name=tool_name,
            artifact=artifact,
        )],
        "search_attempts": state.get("search_attempts", 0) + 1,
        "tool_call_count": call_count,
        "tool_call_artifacts": [call_artifact],
    }


def tool_call_limit_error_node(state: GraphState) -> dict:
    """Record the rejected sixth tool call so callers can inspect the limit failure."""
    messages = state.get("search_agent_state_memory", [])
    tool_calls = getattr(messages[-1], "tool_calls", []) if messages else []
    attempted_call = tool_calls[0] if tool_calls else {}
    count = state.get("tool_call_count", 0)
    maximum = state.get("max_tool_calls", 5)
    error = f"Tool-call limit exceeded: attempted call {count + 1}; maximum is {maximum}."
    artifact = {
        "sequence": count + 1,
        "tool_name": attempted_call.get("name"),
        "tool_input": attempted_call.get("args", {}),
        "executed": False,
        "max_calls": maximum,
        "limit_exceeded": True,
        "error": error,
    }
    return {
        "tool_call_artifacts": [artifact],
        "tool_call_limit_error": error,
    }


def course_id_node(state: GraphState) -> dict:
    tool_call = _pending_tool_call(state, "course_id")
    course_id = str((tool_call or {}).get("args", {}).get("course_id", ""))
    try:
        course = find_course_by_id(course_id)
        if course is None:
            artifact = {"tool_name": "course_id", "course_id": course_id.upper(), "found": False}
            return _tool_result(state, "course_id", artifact, f"Course ID {course_id.upper()} was not found.")
        rank = sum(1 for item in state.get("retrieved_context_raw", []) if isinstance(item, dict) and item.get("course")) + 1
        artifact = {"tool_name": "course_id", "course_id": course_id.upper(), "course": course, "rank": rank}
        return _tool_result(state, "course_id", artifact, json.dumps(course, ensure_ascii=False))
    except (OSError, ValueError, json.JSONDecodeError):
        return _tool_result(state, "course_id", {"tool_name": "course_id", "error": "course lookup unavailable"}, "Course lookup is unavailable.")


def course_catalog_node(state: GraphState) -> dict:
    try:
        courses = load_course_catalog()
        artifact = {"tool_name": "course_catalog", "courses": courses}
        response = _tool_result(state, "course_catalog", artifact, json.dumps(courses, ensure_ascii=False))
        response["retrieved_context_raw"] = [*state.get("retrieved_context_raw", []), artifact]
        return response
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        artifact = {"tool_name": "course_catalog", "error": "course catalogue unavailable"}
        response = _tool_result(state, "course_catalog", artifact, "Course catalogue is unavailable.")
        response["retrieved_context_raw"] = [*state.get("retrieved_context_raw", []), artifact]
        return response


def personal_data_node(state: GraphState) -> dict:
    try:
        profile = load_personal_data()
        artifact = {"tool_name": "personal_data", "profile": profile}
        response = _tool_result(state, "personal_data", artifact, json.dumps(profile, ensure_ascii=False))
        response["retrieved_context_raw"] = [*state.get("retrieved_context_raw", []), artifact]
        return response
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        artifact = {"tool_name": "personal_data", "error": "personal data unavailable"}
        response = _tool_result(state, "personal_data", artifact, "Personal data is unavailable.")
        response["retrieved_context_raw"] = [*state.get("retrieved_context_raw", []), artifact]
        return response


def _capture_exact_course_artifact(state: GraphState, result: dict) -> dict:
    """Add exact lookup evidence, including not-found results, to shared context."""
    messages = result.get("search_agent_state_memory", [])
    if messages and isinstance(messages[0], ToolMessage) and isinstance(messages[0].artifact, dict):
        artifact = messages[0].artifact
        result["retrieved_context_raw"] = [*state.get("retrieved_context_raw", []), artifact]
    return result


def course_id_node_with_artifact(state: GraphState) -> dict:
    return _capture_exact_course_artifact(state, course_id_node(state))


def build_retrieval_graph():
    builder = StateGraph(GraphState)
    builder.add_node("search_agent", search_agent)
    builder.add_node("course_catalog_tool", course_catalog_node)
    builder.add_node("course_id_tool", course_id_node_with_artifact)
    builder.add_node("personal_data_tool", personal_data_node)
    builder.add_node("tool_call_limit_error", tool_call_limit_error_node)
    builder.add_edge(START, "search_agent")
    builder.add_conditional_edges("search_agent", should_continue, {
        "course_catalog_tool": "course_catalog_tool",
        "course_id_tool": "course_id_tool",
        "personal_data_tool": "personal_data_tool",
        "tool_call_limit_error": "tool_call_limit_error",
        END: END,
    })
    builder.add_edge("course_catalog_tool", "search_agent")
    builder.add_edge("course_id_tool", "search_agent")
    builder.add_edge("personal_data_tool", "search_agent")
    builder.add_edge("tool_call_limit_error", END)
    return builder.compile()


def build_main_graph():
    builder = StateGraph(GraphState)
    builder.add_node("template_agent", template_agent)
    builder.add_node("search_agent", build_retrieval_graph())
    builder.add_edge(START, "template_agent")
    builder.add_edge("template_agent", "search_agent")
    builder.add_edge("search_agent", END)
    return builder.compile()


graph = build_main_graph()
