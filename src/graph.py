
from typing import TypedDict
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from tools.keyword_search import SearchResult


class GraphState(TypedDict, total=False):
    conversation: list[str]
    query: str

# first agent artifacts
    guide_agent_state_memory: list[HumanMessage | SystemMessage | ToolMessage | AIMessage]

# second agent artifacts
    search_agent_state_memory: list[HumanMessage | SystemMessage | ToolMessage | AIMessage]
    retrieved_context_raw: list[SearchResult]
    search_attempts: int
    max_search_attempts: int
    final_answer: str

