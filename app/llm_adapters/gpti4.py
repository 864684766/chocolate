from typing import Any

try:
    from langchain_openai import ChatOpenAI
except Exception as e:  # 延迟到构建时抛出更友好的错误
    ChatOpenAI = None  # type: ignore
    _import_error = e

from ..config import Settings


class Gpti4Provider:
    """OpenAI GPT-4 家族提供商适配器（注册名：gpti4）。

    通过 Settings 构建 LangChain 的 ChatOpenAI 实例。
    - 依赖包：langchain-openai（按需安装）
    - 使用的密钥：Settings.api_key（从环境变量 API_KEY 读取）
    """

    def __init__(self, settings: Settings):
        self.settings = settings

    def build_chat_model(self) -> Any:
        if ChatOpenAI is None:
            raise RuntimeError(
                "缺少依赖: langchain-openai。请执行 'pip install langchain-openai'，并在 .env 中设置 API_KEY=你的OpenAIKey"
            ) from _import_error
        if not self.settings.api_key:
            raise RuntimeError(
                "缺少 API_KEY，请在环境或 .env 中设置 API_KEY=你的OpenAIKey（当 PROVIDER=gpti4 时必需）"
            )
        return ChatOpenAI(
            model=self.settings.model,
            api_key=self.settings.api_key,
            temperature=self.settings.temperature,
        )