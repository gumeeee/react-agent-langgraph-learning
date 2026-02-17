from langchain.tools import BaseTool, tool
from langgraph.prebuilt.tool_node import ToolRuntime


@tool
def multiply(a: float, b: float, runtime: ToolRuntime) -> float:
    """Multiply a * b and returns the result

    Args:
        a: float multiplicand
        b: float multiplier

    Returns:
        the resulting float of the equation a * b
    """
    return a * b


TOOLS: list[BaseTool] = [multiply]
TOOLS_BY_NAME: dict[str, BaseTool] = {tool.name: tool for tool in TOOLS}
