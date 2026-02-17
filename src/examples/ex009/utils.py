# ruff: noqa: S101
from typing import cast

from langchain.chat_models import BaseChatModel, init_chat_model


def load_llm() -> BaseChatModel:
    model = cast(
        "BaseChatModel",
        init_chat_model(
            model="google_genai:gemini-2.5-flash",
            temperature=0.2,
            configurable_fields="any",
        ),
    )

    assert hasattr(model, "bind_tools")
    assert hasattr(model, "invoke")
    assert hasattr(model, "with_config")

    return model
