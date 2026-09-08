"""Existing main graph and retrieval subgraph for the course-enquiry POC."""

import json
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from src.agents.data_retriever import search_agent
from src.agents.template_design import template_agent
from src.state import GraphState
from src.tools.course_id import find_course_by_id
from src.tools.keyword_search import multiple_keyword_search
from src.tools.personal_data import load_personal_data


def should_continue(state: GraphState, memory_key: str = "search_agent_state_memory") -> str:
    """Route the Search Agent's latest tool call to its existing tool node."""
    messages = state.get(memory_key, [])
    if not messages or state.get("search_attempts", 0) >= state.get("max_search_attempts", 4):
        return END
    tool_calls = getattr(messages[-1], "tool_calls", None)
    if not tool_calls:
        return END
    return {
        "personal_data": "personal_data_tool",
        "keyword_search": "keyword_search_tool",
        "course_id": "course_id_tool",
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
    return {
        "search_agent_state_memory": [ToolMessage(
            content=content,
            tool_call_id=tool_call["id"],
            name=tool_name,
            artifact=artifact,
        )],
        "search_attempts": state.get("search_attempts", 0) + 1,
    }


def personal_data_node(state: GraphState) -> dict:
    try:
        profile = load_personal_data()
        return _tool_result(state, "personal_data", {"tool_name": "personal_data", "profile": profile}, json.dumps(profile, ensure_ascii=False))
    except (OSError, ValueError, json.JSONDecodeError):
        return _tool_result(state, "personal_data", {"tool_name": "personal_data", "error": "personal data unavailable"}, "Personal data is unavailable.")


def course_id_node(state: GraphState) -> dict:
    tool_call = _pending_tool_call(state, "course_id")
    course_id = str((tool_call or {}).get("args", {}).get("course_id", ""))
    try:
        course = find_course_by_id(course_id)
        if course is None:
            artifact = {"tool_name": "course_id", "course_id": course_id.upper(), "found": False}
            return _tool_result(state, "course_id", artifact, f"Course ID {course_id.upper()} was not found.")
        rank = sum(1 for item in state.get("retrieved_context_raw", []) if isinstance(item, dict) and item.get("course")) + 1
        artifact = {"tool_name": "course_id", "course_id": course_id.upper(), "course": course, "rank": rank, "matched_keywords": [course_id.upper()]}
        return _tool_result(state, "course_id", artifact, json.dumps(course, ensure_ascii=False))
    except (OSError, ValueError, json.JSONDecodeError):
        return _tool_result(state, "course_id", {"tool_name": "course_id", "error": "course lookup unavailable"}, "Course lookup is unavailable.")


def keyword_search_node(state: GraphState) -> dict:
    tool_call = _pending_tool_call(state, "keyword_search")
    args = (tool_call or {}).get("args", {})
    try:
        results = multiple_keyword_search(args.get("keywords", []), top_k=int(args.get("top_k", 5)))
        artifacts = [{"tool_name": "keyword_search", **result} for result in results]
        response = _tool_result(state, "keyword_search", artifacts, json.dumps(results, ensure_ascii=False))
        response["retrieved_context_raw"] = [*state.get("retrieved_context_raw", []), *artifacts]
        return response
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        response = _tool_result(state, "keyword_search", {"tool_name": "keyword_search", "error": "course search unavailable"}, "Course search is unavailable.")
        response["retrieved_context_raw"] = state.get("retrieved_context_raw", [])
        return response


def _capture_exact_course_artifact(state: GraphState, result: dict) -> dict:
    """Add exact course lookup artifacts to the shared raw retrieval context."""
    messages = result.get("search_agent_state_memory", [])
    if messages and isinstance(messages[0], ToolMessage) and isinstance(messages[0].artifact, dict):
        artifact = messages[0].artifact
        if artifact.get("course"):
            result["retrieved_context_raw"] = [*state.get("retrieved_context_raw", []), artifact]
    return result


def course_id_node_with_artifact(state: GraphState) -> dict:
    return _capture_exact_course_artifact(state, course_id_node(state))


def personal_data_node_with_artifact(state: GraphState) -> dict:
    result = personal_data_node(state)
    messages = result.get("search_agent_state_memory", [])
    if messages and isinstance(messages[0], ToolMessage) and isinstance(messages[0].artifact, dict):
        result["retrieved_context_raw"] = [*state.get("retrieved_context_raw", []), messages[0].artifact]
    return result


def build_retrieval_graph():
    builder = StateGraph(GraphState)
    builder.add_node("search_agent", search_agent)
    builder.add_node("personal_data_tool", personal_data_node_with_artifact)
    builder.add_node("keyword_search_tool", keyword_search_node)
    builder.add_node("course_id_tool", course_id_node_with_artifact)
    builder.add_edge(START, "search_agent")
    builder.add_conditional_edges("search_agent", should_continue, {
        "personal_data_tool": "personal_data_tool",
        "keyword_search_tool": "keyword_search_tool",
        "course_id_tool": "course_id_tool",
        END: END,
    })
    builder.add_edge("personal_data_tool", "search_agent")
    builder.add_edge("keyword_search_tool", "search_agent")
    builder.add_edge("course_id_tool", "search_agent")
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
