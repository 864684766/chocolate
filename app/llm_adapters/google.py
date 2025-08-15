from typing import Any

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except Exception as e:  # 延迟在 build 时抛出更友好的错误
    ChatGoogleGenerativeAI = None  # type: ignore
    _import_error = e

from ..config import Settings


class GoogleProvider:
    """Google GenAI 提供商适配器。

    负责基于 Settings 构建 LangChain ChatGoogleGenerativeAI 实例。
    """

    def __init__(self, settings: Settings):
        self.settings = settings

    def build_chat_model(self) -> Any:
        if ChatGoogleGenerativeAI is None:
            raise RuntimeError(
                "缺少依赖: langchain-google-genai。请执行 'pip install langchain-google-genai' 或在 README 的安装指引中选择安装对应 Provider 的依赖。"
            ) from _import_error
        return ChatGoogleGenerativeAI(
            model=self.settings.model,
            google_api_key=self.settings.google_api_key,
            temperature=self.settings.temperature,
        )