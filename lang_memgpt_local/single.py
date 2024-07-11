from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import List, Optional, Union, Dict, Any
from typing import Tuple

import chromadb
import tiktoken
from langchain.chat_models import init_chat_model
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import AnyMessage
from langchain_core.messages.utils import get_buffer_string
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.config import (
    RunnableConfig,
    ensure_config,
    get_executor_for_config,
)
from langchain_core.tools import tool
from langchain_fireworks import FireworksEmbeddings
from langgraph.checkpoint import MemorySaver
from langgraph.graph import END
from langgraph.graph import START, StateGraph, add_messages
from langgraph.prebuilt import ToolNode
from pydantic.v1 import BaseModel, Field
from pydantic_settings import BaseSettings
from typing_extensions import Annotated, TypedDict
from typing_extensions import Literal

# Set up logging
logger = logging.getLogger("memory")
logger.setLevel(logging.ERROR)


# Constants

class Constants(BaseSettings):
    PAYLOAD_KEY: str = "content"
    PATH_KEY: str = "path"
    PATCH_PATH: str = "user/{user_id}/core"
    INSERT_PATH: str = "user/{user_id}/recall/{event_id}"
    TIMESTAMP_KEY: str = "timestamp"
    TYPE_KEY: str = "type"


constants = Constants()

# Schemas
class GraphConfig(TypedDict):
    model: str | None
    """The model to use for the memory assistant."""
    thread_id: str
    """The thread ID of the conversation."""
    user_id: str
    """The ID of the user to remember in the conversation."""


# Define the schema for the state maintained throughout the conversation
class State(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    """The messages in the conversation."""
    core_memories: List[str]
    """The core memories associated with the user."""
    recall_memories: List[str]
    """The recall memories retrieved for the current context."""


# Settings

class Settings(BaseSettings):
    chroma_persist_directory: str = "./chroma_db"
    model: str = "claude-3-5-sonnet-20240620"


settings = Settings()

# Utils


_DEFAULT_DELAY: int = 60  # seconds
chroma_persist_directory: str = "./chroma_db"
model: str = "claude-3-5-sonnet-20240620"


def get_chroma_client() -> chromadb.Client:
    return chromadb.PersistentClient(path=settings.chroma_persist_directory)


def ensure_configurable(config: RunnableConfig) -> GraphConfig:
    """Merge the user-provided config with default values."""
    configurable = config.get("configurable", {})
    return {
        **configurable,
        **GraphConfig(
            delay=configurable.get("delay", _DEFAULT_DELAY),
            model=configurable.get("model", settings.model),
            thread_id=configurable["thread_id"],
            user_id=configurable["user_id"],
        ),
    }

def get_embeddings():
    return FireworksEmbeddings(model="nomic-ai/nomic-embed-text-v1.5")



# Graph

"""Lang-MemGPT: A Long-Term Memory Agent.

This module implements an agent with long-term memory capabilities using LangGraph.
The agent can store, retrieve, and use memories to enhance its interactions with users.

Key Components:
1. Memory Types: Core (always available) and Recall (contextual/semantic)
2. Tools: For saving and retrieving memories + performing other tasks.
3. Vector Database: for recall memory. Uses Chroma for local storage.

"""



# Initialize the search tool for external information retrieval
search_tool = TavilySearchResults(max_results=1)
tools = [search_tool]


# LangGraph tool for saving recall memories
@tool
async def save_recall_memory(memory: str) -> str:
    """Save a memory to the database for later semantic retrieval."""
    # Ensure proper configuration
    config = ensure_config()
    configurable = ensure_configurable(config)

    # Generate embedding for the memory
    embeddings = get_embeddings()
    vector = await embeddings.aembed_query(memory)

    current_time = datetime.now(tz=timezone.utc)
    event_id = str(uuid.uuid4())
    path = constants.INSERT_PATH.format(
        thread_id=configurable["thread_id"],
        user_id=configurable["user_id"],
        event_id=event_id
    )

    # Store memory in ChromaDB
    chroma_client = get_chroma_client()
    collection = chroma_client.get_or_create_collection("memories")
    collection.add(
        ids=[event_id],
        embeddings=[vector],
        metadatas=[{
            constants.PAYLOAD_KEY: memory,
            constants.PATH_KEY: path,
            constants.TIMESTAMP_KEY: current_time.isoformat(),
            constants.TYPE_KEY: "recall",
            "user_id": configurable["user_id"],
        }],
        documents=[memory]
    )
    return memory


# LangGraph tool for searching memories
@tool
def search_memory(query: str, top_k: int = 5) -> List[str]:
    """Search for memories in the database based on semantic similarity."""
    try:
        config = ensure_config()
        configurable = ensure_configurable(config)
        embeddings = get_embeddings()
        vector = embeddings.embed_query(query)

        chroma_client = get_chroma_client()
        collection = chroma_client.get_or_create_collection("memories")

        where_clause = {
            "$and": [
                {"user_id": {"$eq": configurable["user_id"]}},
                {constants.TYPE_KEY: {"$eq": "recall"}}
            ]
        }

        logger.debug(f"Searching memories with query: {query}")
        logger.debug(f"Where clause: {where_clause}")

        results = collection.query(
            query_embeddings=[vector],
            where=where_clause,
            n_results=top_k,
        )

        memories = []
        if 'metadatas' in results and results['metadatas']:
            for metadata_list in results['metadatas']:
                for metadata in metadata_list:
                    if constants.PAYLOAD_KEY in metadata:
                        memories.append(metadata[constants.PAYLOAD_KEY])

        logger.debug(f"Retrieved {len(memories)} memories")
        return memories

    except Exception as e:
        logger.error(f"Error in search_memory: {str(e)}")
        return []


# Function to fetch core memories for a user
def _fetch_core_memories(user_id: str) -> Tuple[str, list[str]]:
    """Fetch core memories for a specific user."""
    path = constants.PATCH_PATH.format(user_id=user_id)
    chroma_client = get_chroma_client()
    collection = chroma_client.get_or_create_collection("core_memories")
    results = collection.get(ids=[path], include=["metadatas"])

    memories = []
    if results and results['metadatas']:
        payload = results['metadatas'][0][constants.PAYLOAD_KEY]
        memories = json.loads(payload)["memories"]
    return path, memories


# LangGraph tool for storing core memories
@tool
def store_core_memory(memory: str, index: Optional[int] = None) -> str:
    """Store a core memory in the database."""
    config = ensure_config()
    configurable = ensure_configurable(config)
    path, existing_memories = _fetch_core_memories(configurable["user_id"])

    if index is not None:
        if index < 0 or index >= len(existing_memories):
            return "Error: Index out of bounds."
        existing_memories[index] = memory
    else:
        # Check if the memory already exists to avoid duplicates
        if memory not in existing_memories:
            existing_memories.insert(0, memory)

    chroma_client = get_chroma_client()
    collection = chroma_client.get_or_create_collection("core_memories")
    collection.upsert(
        ids=[path],
        metadatas=[{
            constants.PAYLOAD_KEY: json.dumps({"memories": existing_memories}),
            constants.PATH_KEY: path,
            constants.TIMESTAMP_KEY: datetime.now(tz=timezone.utc).isoformat(),
            constants.TYPE_KEY: "core",
            "user_id": configurable["user_id"],
        }],
        documents=[json.dumps({"memories": existing_memories})]
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
async def agent(state: State, config: RunnableConfig) -> State:
    """Process the current state and generate a response using the LLM."""
    configurable = ensure_configurable(config)
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
    }


# Function to load memories for the current conversation
def load_memories(state: State, config: RunnableConfig) -> State:
    """Load core and recall memories for the current conversation."""
    configurable = ensure_configurable(config)
    user_id = configurable["user_id"]
    tokenizer = tiktoken.encoding_for_model("gpt-4o")
    convo_str = get_buffer_string(state["messages"])
    convo_str = tokenizer.decode(tokenizer.encode(convo_str)[:2048])

    with get_executor_for_config(config) as executor:
        futures = [
            executor.submit(_fetch_core_memories, user_id),
            executor.submit(search_memory.invoke, convo_str),
        ]
        _, core_memories = futures[0].result()
        recall_memories = futures[1].result()
    return {
        "core_memories": core_memories,
        "recall_memories": recall_memories,
    }


async def query_memories(state: State, config: RunnableConfig) -> State:
    """Query the user's memories."""
    configurable = ensure_configurable(config)
    user_id = configurable["user_id"]
    embeddings = get_embeddings()

    # Get the last few messages to use as a query
    last_messages = state["messages"][-5:]  # Adjust this number as needed
    query = " ".join([str(m.content) for m in last_messages if m.type == "human"])
    logger.debug(f"Querying memories with: {query}")

    vec = await embeddings.aembed_query(query)
    chroma_client = get_chroma_client()
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
    }


# Function to determine the next step in the graph
def route_tools(state: State) -> Literal["tools", "__end__"]:
    """Determine whether to use tools or end the conversation based on the last message."""
    msg = state["messages"][-1]
    if msg.tool_calls:
        return "tools"
    return END


# Create the LangGraph StateGraph
builder = StateGraph(State, GraphConfig)

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


########## The rest of code sets up a chatbot using the memory agent code above ##########


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class ChatState(TypedDict):
    """The state of the chatbot."""
    messages: Annotated[List[AnyMessage], add_messages]
    user_memories: List[dict]


class ChatConfigurable(TypedDict):
    """The configurable fields for the chatbot."""
    user_id: str
    thread_id: str
    model: str
    delay: Optional[float]


def _ensure_configurable(config: RunnableConfig) -> ChatConfigurable:
    """Ensure the configuration is valid."""
    return ChatConfigurable(
        user_id=config["configurable"]["user_id"],
        thread_id=config["configurable"]["thread_id"],
        model=config["configurable"].get(
            "model", "accounts/fireworks/models/firefunction-v2"
        ),
        delay=config["configurable"].get("delay", 60),
    )


PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful and friendly chatbot. Get to know the user!"
            " Ask questions! Be spontaneous!"
            "{user_info}\n\nSystem Time: {time}",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(
    time=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
)


def format_query(messages: List[AnyMessage]) -> str:
    """Format the query for the user's memories."""
    return " ".join([str(m.content) for m in messages if m.type == "human"][-5:])


async def query_memories(state: ChatState, config: RunnableConfig) -> ChatState:
    """Query the user's memories."""
    configurable: ChatConfigurable = config["configurable"]
    user_id = configurable["user_id"]
    embeddings = get_embeddings()

    query = format_query(state["messages"])
    vec = await embeddings.aembed_query(query)
    chroma_client = get_chroma_client()
    collection = chroma_client.get_or_create_collection("memories")

    results = collection.query(
        query_embeddings=[vec],
        where={
            "$and": [
                {"user_id": {"$eq": str(user_id)}},
                {constants.TYPE_KEY: {"$eq": "recall"}}
            ]
        },
        n_results=10,
    )

    memories = [m[constants.PAYLOAD_KEY] for m in results['metadatas'][0]]
    return {
        "user_memories": memories,
    }


def format_memories(memories: List[dict]) -> str:
    """Format the user's memories."""
    if not memories:
        return ""
    memories = "\n".join(str(m) for m in memories)
    return f"""

## Memories

You have noted the following memorable events from previous interactions with the user.
<memories>
{memories}
</memories>
"""


async def bot(state: ChatState, config: RunnableConfig) -> ChatState:
    """Prompt the bot to respond to the user, incorporating memories (if provided)."""
    configurable = _ensure_configurable(config)
    model = init_chat_model(configurable["model"])
    chain = PROMPT | model
    memories = format_memories(state["user_memories"])
    m = await chain.ainvoke(
        {
            "messages": state["messages"],
            "user_info": memories,
        },
        config,
    )

    return {
        "messages": [m],
    }


class MemorableEvent(BaseModel):
    """A memorable event."""
    description: str
    participants: List[str] = Field(
        description="Names of participants in the event and their relationship to the user."
    )


async def post_messages(state: ChatState, config: RunnableConfig) -> ChatState:
    """Process messages and store memories."""
    configurable = _ensure_configurable(config)
    thread_id = config["configurable"]["thread_id"]
    memory_thread_id = uuid.uuid5(uuid.NAMESPACE_URL, f"memory_{thread_id}")

    # Here you would implement the logic to process messages and store memories
    # For example:
    # memories = extract_memories(state["messages"])
    # for memory in memories:
    #     await save_recall_memory(memory)

    return {
        "messages": [],
    }


builder = StateGraph(ChatState, ChatConfigurable)
builder.add_node(query_memories)
builder.add_node(bot)
builder.add_node(post_messages)
builder.add_edge(START, "query_memories")
builder.add_edge("query_memories", "bot")
builder.add_edge("bot", "post_messages")

chat_graph = builder.compile(checkpointer=MemorySaver())


# Example usage
async def main():
    user_id = str(uuid.uuid4())
    thread_id = str(uuid.uuid4())

    chat = Chat(user_id, thread_id)

    response = await chat("Hi there")
    print("Bot:", response)

    response = await chat("I've been planning a surprise party for my friend Steve.")
    print("Bot:", response)

    response = await chat("Steve really likes crocheting. Maybe I can do something with that?")
    print("Bot:", response)

    response = await chat("He's also into capoeira...")
    print("Bot:", response)

    # Wait for a minute to simulate time passing
    print("Waiting for a minute to simulate time passing...")
    await asyncio.sleep(60)

    # Start a new conversation
    thread_id_2 = str(uuid.uuid4())
    chat2 = Chat(user_id, thread_id_2)

    response = await chat2("Remember me?")
    print("Bot:", response)

    response = await chat2("What do you remember about Steve?")
    print("Bot:", response)


class Chat:
    def __init__(self, user_id: str, thread_id: str):
        self.thread_id = thread_id
        self.user_id = user_id

    async def __call__(self, query: str) -> str:
        logger.debug(f"Chat called with query: {query}")
        logger.debug(f"User ID: {self.user_id}, Thread ID: {self.thread_id}")

        chunks = memgraph.astream_events(
            input={
                "messages": [("human", query)],
            },
            config={
                "configurable": {
                    "user_id": self.user_id,
                    "thread_id": self.thread_id,
                    "model": settings.model,
                    "delay": 4,
                }
            },
            version="v1",
        )
        res = []
        try:
            async for event in chunks:
                if event.get("event") == "on_chat_model_stream":
                    tok = event["data"]["chunk"].content
                    self.process_token(tok, res)
                elif event.get("event") == "on_tool_start":
                    logger.debug(f"Tool started: {event.get('name')}")
                elif event.get("event") == "on_tool_end":
                    logger.debug(f"Tool ended: {event.get('name')}")
                    logger.debug(f"Tool output: {event.get('data', {}).get('output')}")
        except Exception as e:
            logger.error(f"Error during chat streaming: {str(e)}")

        print()  # New line after all output
        full_response = "".join(res)
        logger.debug(f"Full response: {full_response}")
        return full_response

    def process_token(self, tok: Union[str, list, Dict[str, Any]], res: list):
        if isinstance(tok, str):
            print(tok, end="", flush=True)
            res.append(tok)
        elif isinstance(tok, list):
            for item in tok:
                self.process_token(item, res)
        elif isinstance(tok, dict):
            if 'text' in tok:
                self.process_token(tok['text'], res)
            else:
                logger.warning(f"Received dict without 'text' key: {tok}")
        else:
            logger.warning(f"Unexpected token type: {type(tok)}")


if __name__ == "__main__":
    asyncio.run(main())



