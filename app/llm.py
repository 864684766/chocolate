from typing import Any

from .llm_adapters.factory import LLMProviderFactory


def get_chat_model() -> Any:
    """根据配置返回对应提供商的聊天模型实例。
    通过 providers 工厂集中管理，便于扩展 OpenAI/Azure/Anthropic 等。
    """
    try:
        return LLMProviderFactory.get_chat_model()
    except NotImplementedError as e:
        # 提供更友好的错误提示
        raise RuntimeError(
            "未找到可用的 LLM Provider，请检查 PROVIDER 环境变量，或安装对应依赖（如 provider=google 需安装 langchain-google-genai）。"
        ) from e


