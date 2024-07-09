import json
from typing import List
from langchain_core.runnables import RunnableConfig

from lang_memgpt_local import _constants as constants
from lang_memgpt_local import _utils as utils
from lang_memgpt_local._schemas import State


async def load_memories(state: State, config: RunnableConfig) -> State:
    """Load core and recall memories for the current conversation."""
    configurable = utils.ensure_configurable(config)
    user_id = configurable["user_id"]

    core_memories = await fetch_core_memories(user_id)
    recall_memories = await search_memory(utils.get_conversation_summary(state["messages"]), config)

    return {
        "messages": state["messages"],
        "core_memories": core_memories,
        "recall_memories": recall_memories,
    }


async def query_memories(state: State, config: RunnableConfig) -> State:
    """Query the user's memories."""
    configurable = utils.ensure_configurable(config)
    user_id = configurable["user_id"]
    query = utils.get_recent_messages_summary(state["messages"])

    where_clause = {
        "$and": [
            {"user_id": {"$eq": str(user_id)}},
            {"thread_id": {"$eq": configurable["thread_id"]}},
            {constants.TYPE_KEY: {"$eq": "recall"}}
        ]
    }

    vector = await utils.get_embedding(query)
    results = utils.db_adapter.query_memories(
        vector=vector,
        where=where_clause,
        n_results=10,
    )

    return {
        "messages": state["messages"],
        "core_memories": [m[constants.PAYLOAD_KEY] for m in results if constants.PAYLOAD_KEY in m],
        "recall_memories": [m[constants.PAYLOAD_KEY] for m in results if constants.PAYLOAD_KEY in m],
    }


async def fetch_core_memories(user_id: str) -> List[str]:
    """Fetch core memories for a specific user."""
    path = constants.PATCH_PATH.format(user_id=user_id)
    collection = utils.db_adapter.get_collection("core_memories")
    results = collection.get(ids=[path], include=["metadatas"])

    memories = []
    if results and results['metadatas']:
        payload = results['metadatas'][0][constants.PAYLOAD_KEY]
        memories = json.loads(payload)["memories"]
    return memories


async def search_memory(query: str, config: RunnableConfig, top_k: int = 5) -> List[str]:
    """Search for memories in the database based on semantic similarity."""
    try:
        configurable = utils.ensure_configurable(config)
        vector = await utils.get_embedding(query)

        where_clause = {
            "$and": [
                {"user_id": {"$eq": configurable["user_id"]}},
                {"thread_id": {"$eq": configurable["thread_id"]}},
                {constants.TYPE_KEY: {"$eq": "recall"}}
            ]
        }

        results = utils.db_adapter.query_memories(vector, where_clause, top_k)

        return [m[constants.PAYLOAD_KEY] for m in results if constants.PAYLOAD_KEY in m]

    except Exception as e:
        utils.logger.error(f"Error in search_memory: {str(e)}")
        return []
