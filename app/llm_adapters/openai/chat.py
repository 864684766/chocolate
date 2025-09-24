from typing import Any

from pydantic import SecretStr

try:
    from langchain_openai import ChatOpenAI
except Exception as e:  # 延迟到构建时抛出更友好的错误
    ChatOpenAI = None  # type: ignore
    _import_error = e

from ...config import Settings
from ..base import ChatProviderBase, ChatModel
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage  # type: ignore


class OpenAIProvider(ChatProviderBase):
    """
    OpenAI 提供商适配器（包结构）。

    职责：
    - 基于 Settings 构建 ChatOpenAI（官方 SDK）实例。
    """

    def __init__(self, settings: Settings):
        self.settings = settings

    def build_chat_model(self) -> Any:
        """
        构建 OpenAI Chat 模型。

        返回：
        - ChatOpenAI：可调用 generate 的聊天模型实例。
        """
        if ChatOpenAI is None:
            raise RuntimeError(
                "缺少依赖: langchain-openai。请执行 'pip install langchain-openai'，并在 .env 中设置 API_KEY=你的OpenAIKey"
            ) from _import_error
        if not self.settings.api_key:
            raise RuntimeError(
                "缺少 API_KEY，请在环境或 .env 中设置 API_KEY=你的OpenAIKey（当 PROVIDER=openai 时必需）"
            )
        lc_model = ChatOpenAI(
            model=self.settings.model,
            api_key=SecretStr(self.settings.api_key),
            temperature=self.settings.temperature,
        )
        return _LCChatWrapper(lc_model)


class _LCChatWrapper(ChatModel):
    def __init__(self, lc_model: Any) -> None:
        self.lc_model = lc_model

    def _to_lc_messages(self, messages: list[dict]) -> list[BaseMessage]:
        out: list[BaseMessage] = []
        for m in messages:
            role = str(m.get("role", "user")).lower()
            content = str(m.get("content", ""))
            if role == "system":
                out.append(SystemMessage(content=content))
            elif role == "assistant":
                out.append(AIMessage(content=content))
            else:
                out.append(HumanMessage(content=content))
        return out

    def generate(self, messages: list[dict], **kwargs: Any) -> dict:
        lc_messages = self._to_lc_messages(messages)
        result: BaseMessage = self.lc_model.invoke(lc_messages)
        text = getattr(result, "content", str(result))
        return {"content": text, "raw_text": text}


