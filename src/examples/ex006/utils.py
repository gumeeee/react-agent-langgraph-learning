from langchain.chat_models import BaseChatModel, init_chat_model


def load_llm() -> BaseChatModel:
    return init_chat_model("google_genai:gemini-2.5-flash")
