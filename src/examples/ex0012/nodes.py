from langgraph.prebuilt.tool_node import ToolNode
from langgraph.runtime import Runtime
from rich import print

from examples.ex0012.context import Context
from examples.ex0012.state import State
from examples.ex0012.tools import TOOLS
from examples.ex0012.utils import load_llm

tool_node = ToolNode(tools=TOOLS)


def call_llm(state: State, runtime: Runtime[Context]) -> State:
    print("> call llm")
    ctx = runtime.context
    user_type = ctx.user_type

    model_provider = "google_genai" if user_type == "plus" else "ollama"
    model = "gemini-2.5-flash" if user_type == "plus" else "qwen3-coder:30b"

    llm_with_tools = load_llm().bind_tools(TOOLS)
    llm_with_config = llm_with_tools.with_config(
        config={
            "configurable": {
                "model": model,
                "model_provider": model_provider,
            }
        }
    )

    result = llm_with_config.invoke(
        state["messages"],
    )

    return {"messages": [result]}
