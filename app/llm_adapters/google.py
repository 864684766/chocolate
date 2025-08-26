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

from ..config import Settings, get_config_manager
from typing import Dict, List


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
        # 读取配置中的 generation_config / safety（仅 google provider 有效）
        config = get_config_manager().get_model_config("google", self.settings.model)
        generation_config: Dict = config.get("generation_config", {})

        # 依据官方文档，将配置映射为 LangChain 提供的枚举类型
        safety_settings = None
        safety_cfg = config.get("safety", {})
        if safety_cfg and HarmBlockThreshold is not None and HarmCategory is not None:
            def _enum_th(name: str, default: str = "BLOCK_ONLY_HIGH"):
                """
                将配置文件中的字符串阈值（例如："BLOCK_NONE"、"BLOCK_ONLY_HIGH"）
                映射为 LangChain 暴露的 HarmBlockThreshold 枚举。

                参数:
                - name: 配置项的键名（hate_speech/harassment/sex/danger）
                - default: 当配置未提供该键时使用的默认阈值

                返回:
                - 对应的枚举值，如 HarmBlockThreshold.BLOCK_NONE；如果找不到则返回 None
                """
                label = str(safety_cfg.get(name, default))  # 读取字符串阈值
                return getattr(HarmBlockThreshold, label, None)  # 反射到枚举

            # safety_settings 是传给 ChatGoogleGenerativeAI 的安全设置字典。
            # 键（key）是“危害类别”的枚举，值（value）是“拦截阈值”的枚举。
            #
            # 1) 危害类别 HarmCategory（内容类型）：
            #    - HARM_CATEGORY_HATE_SPEECH         仇恨言论/歧视
            #    - HARM_CATEGORY_HARASSMENT          骚扰/霸凌
            #    - HARM_CATEGORY_SEXUALLY_EXPLICIT   性相关内容
            #    - HARM_CATEGORY_DANGEROUS_CONTENT   危险内容（违法/自伤/武器等）
            #
            # 2) 拦截阈值 HarmBlockThreshold（安全阀值，数值越“高”越严格）：
            #    - BLOCK_NONE                 不拦截（最宽松）
            #    - BLOCK_ONLY_HIGH            仅拦截高风险
            #    - BLOCK_MEDIUM_AND_ABOVE     拦截中/高风险
            #    - BLOCK_LOW_AND_ABOVE        拦截低/中/高风险（最严格）
            #
            # 提示：一般先从较宽松开始（BLOCK_ONLY_HIGH 或 BLOCK_NONE），
            # 如遇到不当输出再逐步收紧；不同模型（如 pro/flash）可单独配置以适配业务需求。
            # 这里我们把 config/app_config.json 里 google.models.<model>.safety 的字符串配置
            # 按类别逐项映射成对应的枚举值，便于根据不同模型/环境差异化配置安全策略。
            safety_settings = {
                # 仇恨言论类别的安全阈值
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: _enum_th("hate_speech"),
                # 骚扰/霸凌类别的安全阈值
                HarmCategory.HARM_CATEGORY_HARASSMENT: _enum_th("harassment"),
                # 性相关内容类别的安全阈值
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: _enum_th("sex"),
                # 危险内容（违法/自伤/武器等）类别的安全阈值
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: _enum_th("danger"),
            }

        return ChatGoogleGenerativeAI(
            model=self.settings.model,
            google_api_key=self.settings.api_key,
            temperature=self.settings.temperature,
            generation_config=generation_config or None,
            safety_settings=safety_settings,
        )