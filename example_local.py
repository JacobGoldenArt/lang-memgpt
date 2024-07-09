import asyncio
import uuid
import logging
from typing import List, Optional, Union, Dict, Any
from langchain_core.messages import AnyMessage, BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, add_messages
from typing_extensions import Annotated, TypedDict

from lang_memgpt_local import _settings as settings
from lang_memgpt_local import _utils as utils
from lang_memgpt_local.graph import memgraph

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


class ChatState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    user_memories: List[dict]


class ChatConfigurable(TypedDict):
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


class Chat:
    def __init__(self, user_id: str, thread_id: str = None):
        self.thread_id = thread_id or str(uuid.uuid4())
        self.user_id = user_id
        self.state = ChatState(messages=[], user_memories=[])

    async def __call__(self, query: str) -> str:
        logger.debug(f"Chat called with query: {query}")
        logger.debug(f"User ID: {self.user_id}, Thread ID: {self.thread_id}")

        self.state["messages"].append(("human", query))

        chunks = memgraph.astream_events(
            self.state,
            config={
                "configurable": {
                    "user_id": self.user_id,
                    "thread_id": self.thread_id,
                    "model": settings.SETTINGS.model,
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
        self.state["messages"].append(("ai", full_response))
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


if __name__ == "__main__":
    asyncio.run(main())