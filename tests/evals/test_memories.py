import pytest
from unittest.mock import MagicMock, patch
import json
from typing import List
from langchain_core.messages import AIMessage

from lang_memgpt_local import _constants as constants
from lang_memgpt_local import _utils as utils
from lang_memgpt_local.graph import memgraph
from lang_memgpt_local._schemas import GraphConfig


@pytest.fixture(scope="function")
def mock_db_adapter():
    with patch("lang_memgpt_local._utils.db_adapter") as mock_adapter:
        mock_collection = MagicMock()
        mock_adapter.get_collection.return_value = mock_collection
        yield mock_adapter


@pytest.fixture(scope="function")
def mock_chat_model():
    with patch("lang_memgpt_local.agent.init_chat_model") as mock_init:
        mock_model = MagicMock()
        mock_model.invoke.return_value = AIMessage(content="This is a test response")
        mock_init.return_value = mock_model
        yield mock_init


@pytest.mark.parametrize(
    "messages, existing, num_mems_expected",
    [
        ([("user", "hi")], {}, 0),
        (
                [
                    (
                            "user",
                            "When I was young, I had a dog named spot. He was my favorite pup. It's really one of my core memories.",
                    )
                ],
                {},
                1,
        ),
        (
                [
                    (
                            "user",
                            "When I was young, I had a dog named spot. It's really one of my core memories.",
                    )
                ],
                {"memories": ["I am afraid of spiders."]},
                2,
        ),
    ],
)
async def test_patch_memory(
        messages: List[tuple],
        num_mems_expected: int,
        existing: dict,
        mock_db_adapter,
        mock_chat_model,
):
    user_id = "4fddb3ef-fcc9-4ef7-91b6-89e4a3efd112"
    thread_id = "e1d0b7f7-0a8b-4c5f-8c4b-8a6c9f6e5c7a"

    # Set up existing memories
    if existing:
        mock_db_adapter.get_collection.return_value.get.return_value = {
            "metadatas": [{
                constants.PAYLOAD_KEY: json.dumps(existing)
            }]
        }
    else:
        mock_db_adapter.get_collection.return_value.get.return_value = {"metadatas": []}

    # When the memories are patched
    await memgraph.ainvoke(
        {
            "messages": messages,
        },
        {
            "configurable": GraphConfig(
                delay=0.1,
                user_id=user_id,
                thread_id=thread_id,
                model="test-model"
            ),
        },
    )

    if num_mems_expected:
        # Check if upsert was called
        assert mock_db_adapter.upsert.call_count >= 1

        # Get the last call arguments
        last_call_args = mock_db_adapter.upsert.call_args

        # Check the content of the last upsert
        upserted_metadata = last_call_args[0][2][0]  # args[2] is metadatas, [0] is the first metadata dict
        upserted_content = json.loads(upserted_metadata[constants.PAYLOAD_KEY])
        assert len(upserted_content['memories']) == num_mems_expected

        # Check if the new memory is in the upserted content
        new_memory = any("spot" in mem.lower() for mem in upserted_content['memories'])
        assert new_memory, "New memory about Spot should be in the upserted content"

        # If there was an existing memory, check if it's still there
        if existing:
            existing_memory = any("spiders" in mem.lower() for mem in upserted_content['memories'])
            assert existing_memory, "Existing memory about spiders should still be in the upserted content"
    else:
        # If no memories are expected, ensure upsert wasn't called
        mock_db_adapter.upsert.assert_not_called()


@pytest.mark.parametrize(
    "messages, num_events_expected",
    [
        ([("user", "hi")], 0),
        (
                [
                    ("user", "I went to the beach with my friends today."),
                    ("assistant", "That sounds like a fun day."),
                    ("user", "You speak the truth."),
                ],
                1,
        ),
        (
                [
                    ("user", "I went to the beach with my friends."),
                    ("assistant", "That sounds like a fun day."),
                    ("user", "I also went to the park with my family - I like the park."),
                ],
                1,
        ),
    ],
)
async def test_insert_memory(
        messages: List[tuple],
        num_events_expected: int,
        mock_db_adapter,
        mock_chat_model,
):
    user_id = "4fddb3ef-fcc9-4ef7-91b6-89e4a3efd112"
    thread_id = "e1d0b7f7-0a8b-4c5f-8c4b-8a6c9f6e5c7a"

    mock_db_adapter.get_collection.return_value.get.return_value = {"metadatas": []}

    # When the events are inserted
    await memgraph.ainvoke(
        {
            "messages": messages,
        },
        {
            "configurable": GraphConfig(
                delay=0.1,
                user_id=user_id,
                thread_id=thread_id,
                model="test-model"
            ),
        },
    )

    if num_events_expected:
        # Check if add_memory was called at least num_events_expected times
        assert mock_db_adapter.add_memory.call_count >= num_events_expected
    else:
        # If no events are expected, ensure add_memory wasn't called
        mock_db_adapter.add_memory.assert_not_called()