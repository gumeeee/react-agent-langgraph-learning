# ruff: noqa: S101
from typing import cast

from langchain.chat_models import BaseChatModel, init_chat_model


def load_llm() -> BaseChatModel:
    model = cast(
        "BaseChatModel",
        init_chat_model(
            "google_genai:gemini-2.5-flash",
            configurable_fields="any",
        ),  # pyright: ignore[reportCallIssue]
    )

    assert hasattr(model, "bind_tools")
    assert hasattr(model, "invoke")
    assert hasattr(model, "with_config")

    return model
