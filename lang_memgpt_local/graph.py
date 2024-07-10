from __future__ import annotations
import tiktoken
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional, Tuple, List, Dict, Any
import pytest
import langsmith
from langchain.chat_models import init_chat_model
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages.utils import get_buffer_string
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.config import (
    RunnableConfig,
    ensure_config,
    get_executor_for_config,
)
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from typing_extensions import Literal

from lang_memgpt_local import _constants as constants
from lang_memgpt_local import _schemas as schemas
from lang_memgpt_local import _utils as utils


# Set up logging
logger = logging.getLogger("memory")
logger.setLevel(logging.DEBUG)

# Initialize the search tool for external information retrieval
search_tool = TavilySearchResults(max_results=1)
tools = [search_tool]

# Initialize the database adapter
db_adapter = utils.get_vectordb_client()


@tool
async def save_recall_memory(memory: str) -> str:
    """Save a memory to the database for later semantic retrieval."""
    config = ensure_config()
    configurable = utils.ensure_configurable(config)
    embeddings = utils.get_embeddings()
    vector = await embeddings.aembed_query(memory)

    current_time = datetime.now(tz=timezone.utc)
    event_id = str(uuid.uuid4())
    path = constants.INSERT_PATH.format(
        user_id=configurable["user_id"],
        event_id=event_id,
    )

    metadata = {
        constants.PAYLOAD_KEY: memory,
        constants.PATH_KEY: path,
        constants.TIMESTAMP_KEY: current_time.isoformat(),
        constants.TYPE_KEY: "recall",
        "user_id": configurable["user_id"],
    }

    db_adapter.add_memory(event_id, vector, metadata, memory)
    return memory

@tool
def search_memory(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Search for memories in the database based on semantic similarity."""
    try:
        config = utils.ensure_config()
        configurable = utils.ensure_configurable(config)
        embeddings = utils.get_embeddings()
        vector = embeddings.embed_query(query)

        where_clause = {
            "$and": [
                {"user_id": {"$eq": configurable["user_id"]}},
                {constants.TYPE_KEY: {"$eq": "recall"}}
            ]
        }

        results = db_adapter.query_memories(vector, where_clause, top_k)

        # Return the full metadata instead of just the payload
        return results

    except Exception as e:
        logger.error(f"Error in search_memory: {str(e)}")
        return []


@langsmith.traceable
def fetch_core_memories(user_id: str) -> Tuple[str, list[str]]:
    """Fetch core memories for a specific user."""
    path = constants.PATCH_PATH.format(user_id=user_id)
    collection = db_adapter.get_collection("core_memories")
    results = collection.get(ids=[path], include=["metadatas"])

    memories = []
    if results and results['metadatas']:
        payload = results['metadatas'][0][constants.PAYLOAD_KEY]
        memories = json.loads(payload)["memories"]
    return path, memories


@tool
def store_core_memory(memory: str, index: Optional[int] = None) -> str:
    """Store a core memory in the database."""
    config = ensure_config()
    configurable = utils.ensure_configurable(config)
    path, existing_memories = fetch_core_memories(configurable["user_id"])

    if index is not None:
        if index < 0 or index >= len(existing_memories):
            return "Error: Index out of bounds."
        existing_memories[index] = memory
    else:
        if memory not in existing_memories:
            existing_memories.insert(0, memory)

    db_adapter.upsert(
        "core_memories",
        [path],
        [{
            constants.PAYLOAD_KEY: json.dumps({"memories": existing_memories}),
            constants.PATH_KEY: path,
            constants.TIMESTAMP_KEY: datetime.now(tz=timezone.utc).isoformat(),
            constants.TYPE_KEY: "core",
            "user_id": configurable["user_id"],
        }],
        [json.dumps({"memories": existing_memories})]
    )
    return "Memory stored."


# Combine all tools including the tavily search tool
all_tools = tools + [save_recall_memory, search_memory, store_core_memory]

# Define the prompt template for the agent
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant with advanced long-term memory"
            " capabilities. Utilize the available memory tools to store and retrieve"
            " important details that will help you better attend to the user's"
            " needs and understand their context.\n\n"
            "## Core Memories\n"
            "Core memories are fundamental to understanding the user and are"
            " always available:\n{core_memories}\n\n"
            "## Recall Memories\n"
            "Recall memories are contextually retrieved based on the current"
            " conversation:\n{recall_memories}\n\n"
            "## Instructions\n"
            "Engage with the user naturally, as a trusted colleague or friend."
            " There's no need to explicitly mention your memory capabilities."
            " Instead, seamlessly incorporate your understanding of the user"
            " into your responses. Be attentive to subtle cues and underlying"
            " emotions. Adapt your communication style to match the user's"
            " preferences and current emotional state. Use tools to persist"
            " information you want to retain in the next conversation.\n\n"
            "Current system time: {current_time}\n\n",
        ),
        ("placeholder", "{messages}"),
    ]
)

# Main agent function
async def agent(state: schemas.State, config: RunnableConfig) -> schemas.State:
    """Process the current state and generate a response using the LLM."""
    configurable = utils.ensure_configurable(config)
    llm = init_chat_model(configurable["model"])
    bound = prompt | llm.bind_tools(all_tools)
    core_str = (
        "<core_memory>\n" + "\n".join(state["core_memories"]) + "\n</core_memory>"
    )
    recall_str = (
        "<recall_memory>\n" + "\n".join(state["recall_memories"]) + "\n</recall_memory>"
    )
    logger.debug(f"Core memories: {core_str}")
    logger.debug(f"Recall memories: {recall_str}")
    prediction = await bound.ainvoke(
        {
            "messages": state["messages"],
            "core_memories": core_str,
            "recall_memories": recall_str,
            "current_time": datetime.now(tz=timezone.utc).isoformat(),
        }
    )
    return {
        "messages": prediction,
        "core_memories": state["core_memories"],
        "recall_memories": state["recall_memories"],
    }

# Function to load memories for the current conversation
def load_memories(state: schemas.State, config: RunnableConfig) -> schemas.State:
    """Load core and recall memories for the current conversation."""
    configurable = utils.ensure_configurable(config)
    user_id = configurable["user_id"]
    tokenizer = tiktoken.encoding_for_model("gpt-4o")
    convo_str = get_buffer_string(state["messages"])
    convo_str = tokenizer.decode(tokenizer.encode(convo_str)[:2048])

    with get_executor_for_config(config) as executor:
        futures = [
            executor.submit(fetch_core_memories, user_id),
            executor.submit(search_memory.invoke, convo_str),
        ]
        _, core_memories = futures[0].result()
        recall_memories = futures[1].result()
    return {
        "core_memories": core_memories,
        "recall_memories": recall_memories,
        "messages": state["messages"],
    }

async def query_memories(state: schemas.State, config: RunnableConfig) -> schemas.State:
    """Query the user's memories."""
    configurable = utils.ensure_configurable(config)
    user_id = configurable["user_id"]
    embeddings = utils.get_embeddings()

    # Get the last few messages to use as a query
    last_messages = state["messages"][-5:]  # Adjust this number as needed
    query = " ".join([str(m.content) for m in last_messages if m.type == "human"])
    logger.debug(f"Querying memories with: {query}")
    
    vec = await embeddings.aembed_query(query)
    chroma_client = utils.get_vectordb_client()
    collection = chroma_client.get_or_create_collection("memories")
    
    # Correct the where clause format
    where_clause = {
        "$and": [
            {"user_id": {"$eq": str(user_id)}},
            {constants.TYPE_KEY: {"$eq": "recall"}}
        ]
    }
    
    logger.debug(f"Searching for memories with where clause: {where_clause}")
    
    results = collection.query(
        query_embeddings=[vec],
        where=where_clause,
        n_results=10,
    )
    
    # Correct handling of ChromaDB query results
    memories = []
    if results['metadatas']:
        for metadata in results['metadatas']:
            if isinstance(metadata, list):
                memories.extend([m.get(constants.PAYLOAD_KEY) for m in metadata if constants.PAYLOAD_KEY in m])
            elif isinstance(metadata, dict):
                if constants.PAYLOAD_KEY in metadata:
                    memories.append(metadata[constants.PAYLOAD_KEY])
    
    logger.debug(f"Retrieved memories: {memories}")
    
    return {
        "recall_memories": memories,
        "messages": state["messages"],
        "core_memories": state["core_memories"],
    }



# Function to determine the next step in the graph
def route_tools(state: schemas.State) -> Literal["tools", "__end__"]:
    """Determine whether to use tools or end the conversation based on the last message."""
    msg = state["messages"][-1]
    if msg.tool_calls:
        return "tools"
    return END

# Create the LangGraph StateGraph
builder = StateGraph(schemas.State, schemas.GraphConfig)

# Add nodes to the graph
builder.add_node("load_memories", load_memories)
builder.add_node("query_memories", query_memories)
builder.add_node("agent", agent)
builder.add_node("tools", ToolNode(all_tools))

# Update the edges to include query_memories
builder.add_edge(START, "load_memories")
builder.add_edge("load_memories", "query_memories")
builder.add_edge("query_memories", "agent")
builder.add_conditional_edges("agent", route_tools)
builder.add_edge("tools", "query_memories")

# Compile the graph into an executable LangGraph
memgraph = builder.compile()

__all__ = ["memgraph"]
