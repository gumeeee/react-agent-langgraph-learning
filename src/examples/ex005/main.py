from langchain.chat_models import init_chat_model
from langchain.tools import BaseTool, tool
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from pydantic import ValidationError
from rich import print


@tool
def multiply(a: float, b: float) -> float:
    """Multiply a * b and returns the result

    Args:
        a: float multiplicand
        b: float multiplier

    Returns:
        the resulting float of the equation a * b
    """
    return a * b


llm = init_chat_model("google_genai:gemini-2.5-flash")


system_message = SystemMessage(
    "You are a helpful assistent. You have access to tools. When the user asks "
    "for something, first look if you have a tool that solves that problem."
)

human_message = HumanMessage(
    "Oi, sou Guilherme Moura. Pode me falar quanto é 1.13 vezes 2.31?"
)

messages: list[BaseMessage] = [
    system_message,
    human_message,
]


tools: list[BaseTool] = [multiply]
tools_by_name = {tool.name: tool for tool in tools}
llm_with_tools = llm.bind_tools(tools)

llm_response = llm_with_tools.invoke(messages)
messages.append(llm_response)

if isinstance(llm_response, AIMessage) and getattr(llm_response, "tool_calls", None):
    call = llm_response.tool_calls[-1]
    name, args, id_ = call["name"], call["args"], call["id"]

    try:
        content = tools_by_name[name].invoke(args)
        status = "success"
    except (KeyError, IndexError, TypeError, ValidationError, ValueError) as error:
        content = f"Please, fix your mistakes: {error}"
        status = "error"

    tool_message = ToolMessage(content=content, tool_call_id=id_, status=status)
    messages.append(tool_message)

    llm_response = llm_with_tools.invoke(messages)
    messages.append(llm_response)

print(messages)
