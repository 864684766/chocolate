from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI

from ..config import Settings


class GoogleProvider:
    """Google GenAI 提供商适配器。

    负责基于 Settings 构建 LangChain ChatGoogleGenerativeAI 实例。
    """

    def __init__(self, settings: Settings):
        self.settings = settings

    def build_chat_model(self) -> Any:
        return ChatGoogleGenerativeAI(
            model=self.settings.model,
            google_api_key=self.settings.google_api_key,
            temperature=self.settings.temperature,
        )