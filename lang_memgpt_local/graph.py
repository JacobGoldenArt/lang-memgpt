from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from typing import Union, Literal

from lang_memgpt_local._schemas import State, GraphConfig
from lang_memgpt_local.agent import agent
from lang_memgpt_local.memory import load_memories, query_memories
from lang_memgpt_local.tools import all_tools

def create_memory_graph() -> StateGraph:
    builder = StateGraph(State, GraphConfig)

    # Add nodes to the graph
    builder.add_node("load_memories", load_memories)
    builder.add_node("query_memories", query_memories)
    builder.add_node("agent", agent)
    builder.add_node("tools", ToolNode(all_tools))

    # Define edges
    builder.add_edge(START, "load_memories")
    builder.add_edge("load_memories", "query_memories")
    builder.add_edge("query_memories", "agent")
    builder.add_conditional_edges("agent", route_tools)
    builder.add_edge("tools", "query_memories")

    return builder.compile()

def route_tools(state: State) -> Union[Literal["tools"], Literal["__end__"]]:
    """Determine whether to use tools or end the conversation based on the last message."""
    if state["messages"] and state["messages"][-1].tool_calls:
        return "tools"
    return END

memgraph = create_memory_graph()