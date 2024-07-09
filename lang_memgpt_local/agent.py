import langsmith
from langchain.chat_models import ChatAnthropic
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessage, BaseMessage

from lang_memgpt_local import _utils as utils
from lang_memgpt_local.prompts import get_agent_prompt
from lang_memgpt_local.tools import all_tools
from lang_memgpt_local._schemas import State


def init_chat_model(model_name: str):
    return ChatAnthropic(model=model_name)


@langsmith.traceable
async def agent(state: State, config: RunnableConfig) -> State:
    """Process the current state and generate a response using the LLM."""
    configurable = utils.ensure_configurable(config)
    llm = init_chat_model(configurable["model"])
    bound = get_agent_prompt() | llm.bind_tools(all_tools)

    context = {
        "messages": state["messages"],
        "core_memories": state.get("core_memories", []),
        "recall_memories": state.get("recall_memories", []),
        "current_time": utils.get_current_time(),
        "thread_id": configurable["thread_id"],
        "user_id": configurable["user_id"],
    }

    prediction = await bound.ainvoke(context, config=config)

    # Ensure prediction is an AIMessage
    if not isinstance(prediction, AIMessage):
        prediction = AIMessage(content=str(prediction))

    # Update the state with the new message
    updated_messages = state["messages"] + [prediction]

    return {
        "messages": updated_messages,
        "core_memories": state.get("core_memories", []),
        "recall_memories": state.get("recall_memories", []),
    }