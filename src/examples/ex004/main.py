from collections.abc import Sequence
from typing import Annotated, TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph, add_messages
from rich import print
from rich.markdown import Markdown

llm = init_chat_model("google_genai:gemini-2.5-flash")


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


def call_llm(state: AgentState) -> AgentState:
    llm_result = llm.invoke(state["messages"])
    return {"messages": [llm_result]}


builder = StateGraph(
    AgentState, context_schema=None, input_schema=AgentState, output_schema=AgentState
)

builder.add_node("call_llm", call_llm)
builder.add_edge(START, "call_llm")
builder.add_edge("call_llm", END)

graph = builder.compile()

if __name__ == "__main__":
    current_messages: Sequence[BaseMessage] = []

    while True:
        user_input = input("Digite sua message: ")
        print(Markdown("---"))

        if user_input.lower() in ["q", "quit", "exit"]:
            print("Bye 👋")
            print(Markdown("---"))
            break

        human_message = HumanMessage(user_input)
        current_messages = [*current_messages, human_message]
        result = graph.invoke({"messages": current_messages})
        current_messages = result["messages"]

        print(Markdown(str(result["messages"][-1].content)))
        print(Markdown("---"))
