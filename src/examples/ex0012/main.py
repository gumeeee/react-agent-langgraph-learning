import asyncio

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import RunnableConfig
from rich import print
from rich.markdown import Markdown
from rich.prompt import Prompt

from examples.ex0012.checkpointer import build_checkpointer_psql
from examples.ex0012.constants import DB_DSN
from examples.ex0012.context import Context
from examples.ex0012.graph import build_graph
from examples.ex0012.prompts import SYSTEM_PROMPT
from examples.ex0012.utils import async_lifespan


async def run_graph(checkpointer: BaseCheckpointSaver) -> None:
    graph = build_graph(checkpointer)

    context = Context(user_type="plus")
    config = RunnableConfig(
        configurable={"thread_id": 1},
    )
    all_messages: list[BaseMessage] = []

    prompt = Prompt()
    Prompt.prompt_suffix = ""

    while True:
        user_input = prompt.ask("[bold cyan]Você: \n")
        print(Markdown("\n\n  ---  \n\n"))

        if user_input.lower() in ["q", "quit"]:
            break

        human_message = HumanMessage(user_input)
        current_loop_messages = [human_message]

        if len(all_messages) == 0:
            current_loop_messages = [SystemMessage(SYSTEM_PROMPT), human_message]

        result = await graph.ainvoke(
            {"messages": current_loop_messages}, config=config, context=context
        )

        model_name = ""
        last_message = result["messages"][-1]

        if isinstance(last_message, AIMessage):
            model_name = last_message.response_metadata.get("model", "")

        print(f"[bold cyan]RESPOSTA ({model_name}): \n")
        print(Markdown(last_message.text))
        print(last_message)
        print(Markdown("\n\n  ---  \n\n"))

        all_messages = result["messages"]

    print(await graph.aget_state(config=config))


async def main() -> None:
    async with (
        async_lifespan() as value,
        build_checkpointer_psql(DB_DSN) as checkpointer,
    ):
        await run_graph(checkpointer)


if __name__ == "__main__":
    asyncio.run(main())
