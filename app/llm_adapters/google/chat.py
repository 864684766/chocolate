from typing import Any

try:
    from langchain_google_genai import (
        ChatGoogleGenerativeAI,
        HarmBlockThreshold,
        HarmCategory,
    )
except Exception as e:  # 延迟在 build 时抛出更友好的错误
    ChatGoogleGenerativeAI = None  # type: ignore
    HarmBlockThreshold = None  # type: ignore
    HarmCategory = None  # type: ignore
    _import_error = e

from ...config import Settings, get_config_manager
from ..base import ChatProviderBase, ChatModel
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage  # type: ignore
from typing import Dict


class GoogleProvider(ChatProviderBase):
    """
    Google GenAI 提供商适配器（包结构）。

    职责：
    - 读取 providers.google.models.<model> 的 generation_config/safety。
    - 基于 Settings 构建 ChatGoogleGenerativeAI 实例。

    说明（重要）：
    - generation_config：用于设置输出相关参数，例如 `response_mime_type`、`max_output_tokens`。
      由于部分 SDK 不接受直接传入一个 generation_config 字段，这里将常见子项拆解到显式参数中，
      以避免出现 "Unexpected argument 'generation_config'" 之类的告警。
    - safety：安全策略映射。配置文件中使用字符串阈值（例如："BLOCK_NONE"、"BLOCK_ONLY_HIGH"），
      在此转换为 LangChain/SDK 暴露的枚举（HarmBlockThreshold / HarmCategory），实现强类型校验与 IDE 提示。
    """

    def __init__(self, settings: Settings):
        self.settings = settings

    def build_chat_model(self) -> Any:
        """
        构建 Google Chat 模型。

        返回：
        - ChatGoogleGenerativeAI：可调用 generate 的聊天模型实例。
        """
        if ChatGoogleGenerativeAI is None:
            raise RuntimeError(
                "缺少依赖: langchain-google-genai。请执行 'pip install langchain-google-genai' 或在 README 的安装指引中选择安装对应 Provider 的依赖。"
            ) from _import_error
        # 读取配置中的 generation_config / safety（仅 google provider 有效）
        config = get_config_manager().get_model_config("google", self.settings.model)
        generation_config: Dict = config.get("generation_config", {})
        # 拆分常见字段，避免直接传入嵌套字典导致的参数不识别
        max_output_tokens = generation_config.get("max_output_tokens")
        model_kwargs: Dict = {}
        if generation_config.get("response_mime_type"):
            model_kwargs["response_mime_type"] = generation_config["response_mime_type"]

        safety_settings = None
        safety_cfg = config.get("safety", {})
        if safety_cfg and HarmBlockThreshold is not None and HarmCategory is not None:
            def _enum_th(name: str, default: str = "BLOCK_NONE"):
                """
                将配置文件中的字符串阈值（例如："BLOCK_NONE"、"BLOCK_ONLY_HIGH"）
                映射为 LangChain 暴露的 HarmBlockThreshold 枚举。

                参数:
                - name (str): 配置项的键名（hate_speech/harassment/sex/danger）
                - default (str): 当配置未提供该键时使用的默认阈值

                返回:
                - 对应的枚举值，如 HarmBlockThreshold.BLOCK_NONE；如果找不到则返回 None
                """
                label = str(safety_cfg.get(name, default))
                return getattr(HarmBlockThreshold, label, None)

            # safety_settings 传给 ChatGoogleGenerativeAI，用于控制有害内容的拦截级别。
            # 键（key）为“危害类别”枚举，值（value）为“拦截阈值”枚举：
            #
            # 1) 危害类别 HarmCategory（内容类型）：
            #    - HARM_CATEGORY_HATE_SPEECH         仇恨言论/歧视
            #    - HARM_CATEGORY_HARASSMENT          骚扰/霸凌
            #    - HARM_CATEGORY_SEXUALLY_EXPLICIT   性相关内容
            #    - HARM_CATEGORY_DANGEROUS_CONTENT   危险内容（违法/自伤/武器等）
            #
            # 2) 拦截阈值 HarmBlockThreshold（越“高”越严格）：
            #    - BLOCK_NONE                 不拦截（最宽松）
            #    - BLOCK_ONLY_HIGH            仅拦截高风险
            #    - BLOCK_MEDIUM_AND_ABOVE     拦截中/高风险
            #    - BLOCK_LOW_AND_ABOVE        拦截低/中/高风险（最严格）
            #
            # 建议：一般先从较宽松开始（BLOCK_ONLY_HIGH 或 BLOCK_NONE），如遇不当输出再逐步收紧；
            # 不同模型（如 pro/flash）可单独配置以适配业务需求。
            safety_settings_raw = {
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: _enum_th("hate_speech"),
                HarmCategory.HARM_CATEGORY_HARASSMENT: _enum_th("harassment"),
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: _enum_th("sex"),
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: _enum_th("danger"),
            }
            safety_settings = {k: v for k, v in safety_settings_raw.items() if v is not None} or None

        kwargs: Dict = {
            "model": self.settings.model,
            "google_api_key": self.settings.api_key,
            "temperature": self.settings.temperature,
        }
        if isinstance(max_output_tokens, int):
            kwargs["max_output_tokens"] = max_output_tokens
        if model_kwargs:
            kwargs["model_kwargs"] = model_kwargs
        if safety_settings:
            kwargs["safety_settings"] = safety_settings

        lc_model = ChatGoogleGenerativeAI(**kwargs)
        return _LCChatWrapper(lc_model)


class _LCChatWrapper(ChatModel):
    """
    LangChain ChatModel 包装器：
    - 适配本项目统一的 ChatModel 接口（messages 为 dict 列表；返回 dict）。
    - 内部将 messages 转为 LangChain BaseMessage，再调用 lc_model.invoke。
    """

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
        # 对于大多数 LangChain ChatModel，优先使用 invoke 简化批处理
        result: BaseMessage = self.lc_model.invoke(lc_messages)
        text = getattr(result, "content", str(result))
        return {"content": text, "raw_text": text}


