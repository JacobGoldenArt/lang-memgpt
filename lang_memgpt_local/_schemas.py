from __future__ import annotations

from typing import List, Optional

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from typing_extensions import Annotated, TypedDict


class GraphConfig(TypedDict, total=False):
    model: Optional[str]
    thread_id: str
    user_id: str
    delay: Optional[float]

# Define the schema for the state maintained throughout the conversation
class State(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    """The messages in the conversation."""
    core_memories: List[str]
    """The core memories associated with the user."""
    recall_memories: List[str]
    """The recall memories retrieved for the current context."""


__all__ = [
    "State",
    "GraphConfig",
]
