from typing import Literal

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tracers.stdout import FunctionCallbackHandler
from langgraph.graph.state import RunnableConfig
from rich import print
from rich.markdown import Markdown
from rich.prompt import Prompt

from examples.ex007.graph import build_graph
from examples.ex007.prompts import SYSTEM_PROMPT


def main() -> None:
    graph = build_graph()

    fn_handler_db = FunctionCallbackHandler(function=print)

    user_type: Literal["plus", "enterprise"] = "plus"
    config = RunnableConfig(
        run_name="meu_grafo",
        tags=["enterprise"],
        configurable={"thread_id": 1, "user_type": user_type},
        max_concurrency=4,
        recursion_limit=25,
        callbacks=[fn_handler_db],
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

        result = graph.invoke({"messages": current_loop_messages}, config=config)

        print("[bold cyan]RESPOSTA: \n")
        print(Markdown(result["messages"][-1].content))
        print(Markdown("\n\n  ---  \n\n"))

        all_messages = result["messages"]

    print(all_messages)


if __name__ == "__main__":
    main()
