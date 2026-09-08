from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage


AgentMessage = HumanMessage | SystemMessage | ToolMessage | AIMessage


class GraphState(TypedDict, total=False):
    """State shared by the guide agent and retrieval subgraph."""

    conversation: list[str]
    query: str
    guide_agent_state_memory: Annotated[list[AgentMessage], operator.add]
    search_agent_state_memory: Annotated[list[AgentMessage], operator.add]
    retrieved_context_raw: list[dict[str, Any]]
    search_attempts: int
    max_search_attempts: int
    final_answer: str
