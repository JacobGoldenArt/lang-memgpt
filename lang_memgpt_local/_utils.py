import uuid
from datetime import datetime, timezone
from functools import lru_cache
from importlib import import_module
from typing import Any, Dict

from langchain_core.runnables import RunnableConfig
from langchain_core.messages.utils import get_buffer_string
from langchain_fireworks import FireworksEmbeddings
import tiktoken

from lang_memgpt_local import _settings as settings
from lang_memgpt_local._schemas import GraphConfig

import logging

logger = logging.getLogger(__name__)

_DEFAULT_DELAY = 60  # seconds


@lru_cache
def get_vectordb_client():
    module_name, class_name = settings.SETTINGS.vectordb_class.rsplit('.', 1)
    module = import_module(module_name)
    VectorDBClass = getattr(module, class_name)
    return VectorDBClass(**settings.SETTINGS.vectordb_config)

db_adapter = get_vectordb_client()


def ensure_config() -> RunnableConfig:
    """Ensure a config is present."""
    return {}



def ensure_configurable(config: RunnableConfig) -> GraphConfig:
    """Merge the user-provided config with default values."""
    configurable = config.get("configurable", {})
    return GraphConfig(
        delay=configurable.get("delay", 60),
        model=configurable.get("model", "claude-3-haiku-20240307"),
        thread_id=configurable["thread_id"],
        user_id=configurable["user_id"],
    )



@lru_cache
def get_embeddings():
    return FireworksEmbeddings(model="nomic-ai/nomic-embed-text-v1.5")


async def get_embedding(text: str):
    embeddings = get_embeddings()
    return await embeddings.aembed_query(text)


def get_current_time():
    return datetime.now(tz=timezone.utc).isoformat()


def generate_uuid():
    return str(uuid.uuid4())


def get_conversation_summary(messages: list) -> str:
    tokenizer = tiktoken.encoding_for_model("gpt-4")
    convo_str = get_buffer_string(messages)
    return tokenizer.decode(tokenizer.encode(convo_str)[:2048])


def get_recent_messages_summary(messages: list, num_messages: int = 5) -> str:
    recent_messages = messages[-num_messages:]
    return " ".join([str(m.content) for m in recent_messages if m.type == "human"])


def get_executor_for_config(config: Dict[str, Any]):
    # This is a placeholder. You might want to implement actual executor logic here.
    class DummyExecutor:
        def submit(self, func, *args, **kwargs):
            return DummyFuture(func(*args, **kwargs))

    class DummyFuture:
        def __init__(self, result):
            self._result = result

        def result(self):
            return self._result

    return DummyExecutor()

__all__ = ["ensure_configurable", "get_vectordb_client", "db_adapter", "get_embeddings"]