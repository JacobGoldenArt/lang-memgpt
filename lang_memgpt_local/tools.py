from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from typing import Optional, List
import json

from lang_memgpt_local import _utils as utils
from lang_memgpt_local import _constants as constants
from lang_memgpt_local.memory import fetch_core_memories, search_memory

search_tool = TavilySearchResults(max_results=1)


@tool
async def save_recall_memory(memory: str) -> str:
    """Save a memory to the database for later semantic retrieval."""
    config = utils.ensure_config()
    configurable = utils.ensure_configurable(config)
    vector = await utils.get_embedding(memory)

    event_id = utils.generate_uuid()
    path = constants.INSERT_PATH.format(
        user_id=configurable["user_id"],
        event_id=event_id,
    )

    metadata = {
        constants.PAYLOAD_KEY: memory,
        constants.PATH_KEY: path,
        constants.TIMESTAMP_KEY: utils.get_current_time(),
        constants.TYPE_KEY: "recall",
        "user_id": configurable["user_id"],
        "thread_id": configurable["thread_id"],
    }

    utils.db_adapter.add_memory(event_id, vector, metadata, memory)
    return memory


# Update other tool functions similarly


@tool
async def search_memory_tool(query: str, top_k: int = 5) -> List[str]:
    """Search for memories in the database based on semantic similarity."""
    config = utils.ensure_config()
    return await search_memory(query, config, top_k)





@tool
async def store_core_memory(memory: str, index: Optional[int] = None) -> str:
    """Store a core memory in the database."""
    config = utils.ensure_config()
    configurable = utils.ensure_configurable(config)
    existing_memories = await fetch_core_memories(configurable["user_id"])

    if index is not None:
        if index < 0 or index >= len(existing_memories):
            return "Error: Index out of bounds."
        existing_memories[index] = memory
    else:
        if memory not in existing_memories:
            existing_memories.insert(0, memory)

    path = constants.PATCH_PATH.format(user_id=configurable["user_id"])
    utils.db_adapter.upsert(
        "core_memories",
        [path],
        [{
            constants.PAYLOAD_KEY: json.dumps({"memories": existing_memories}),
            constants.PATH_KEY: path,
            constants.TIMESTAMP_KEY: utils.get_current_time(),
            constants.TYPE_KEY: "core",
            "user_id": configurable["user_id"],
            "thread_id": configurable["thread_id"],  # Add this line
        }],
        [json.dumps({"memories": existing_memories})]
    )
    return "Memory stored."


all_tools = [search_tool, save_recall_memory, search_memory_tool, store_core_memory]