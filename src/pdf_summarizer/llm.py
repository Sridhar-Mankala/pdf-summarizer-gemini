"""Model factory - keeps LangChain wiring in one place."""

from __future__ import annotations

from langchain_google_genai import ChatGoogleGenerativeAI

from .config import Settings


def build_chat_model(settings: Settings, *, streaming: bool = False) -> ChatGoogleGenerativeAI:
    """Gemini chat model with retries for transient 429/5xx responses."""
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.api_key,
        temperature=settings.temperature,
        max_retries=3,
        streaming=streaming,
        convert_system_message_to_human=False,
    )
