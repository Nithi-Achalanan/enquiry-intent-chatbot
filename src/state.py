from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from src.agents.template_agent import template_agent
from agents.data_retriever import search_agent
from src.state import GraphState

from src.tools.personal_data import personal_data_tool
from tools.keyword_search import keyword_search_tool
from src.tools.course_id_search import course_id_tool


def should_continue(state: GraphState,memory_key: str) -> str:

    messages = state.get(memory_key, [])

    if not messages:
        return END

    last_message = messages[-1]
    tool_calls = getattr(last_message, "tool_calls", None)

    if not tool_calls:
        return END

    tool_name = tool_calls[0]["name"]

    if tool_name == "personal_data":
        return "personal_data_tool"

    if tool_name == "keyword_search":
        return "keyword_search_tool"

    if tool_name == "course_id":
        return "course_id_tool"

    return END


def build_retrieval_graph():
    builder = StateGraph(GraphState)

    builder.add_node("search_agent",search_agent)
    builder.add_node("personal_data_tool",ToolNode([personal_data_tool]))
    builder.add_node("keyword_search_tool",ToolNode([keyword_search_tool]))
    builder.add_node("course_id_tool",ToolNode([course_id_tool]))

    builder.add_edge(START,"search_agent")
    builder.add_conditional_edges("search_agent",
        lambda state: should_continue(
            state,
            "search_agent_state_memory",
        ),
        {
            "personal_data_tool": "personal_data_tool",
            "keyword_search_tool": "keyword_search_tool",
            "course_id_tool": "course_id_tool",
            END: END,
        },
    )
    builder.add_edge("personal_data_tool","search_agent")
    builder.add_edge("keyword_search_tool","search_agent")
    builder.add_edge("course_id_tool","search_agent")

    return builder.compile()


def build_main_graph():
    builder = StateGraph(GraphState)

    builder.add_node("template_agent",template_agent)
    builder.add_node("search_agent", build_retrieval_graph())

    builder.add_edge(START,"template_agent")
    builder.add_edge("template_agent","search_agent")
    builder.add_edge("search_agent",END)

    return builder.compile()

graph = build_main_graph()