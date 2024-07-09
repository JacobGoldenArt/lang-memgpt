from __future__ import annotations

from typing import List

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from typing_extensions import Annotated, TypedDict


class GraphConfig(TypedDict):
    model: str | None
    thread_id: str
    user_id: str
    delay: float | None

class State(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    core_memories: List[str]
    recall_memories: List[str]


__all__ = [
    "State",
    "GraphConfig",
]
