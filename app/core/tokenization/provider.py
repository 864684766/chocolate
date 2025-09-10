from __future__ import annotations

"""
分词计数提供器（TokenCounter）。

目标：
- 复用应用层的模型配置（ai_type/provider → model_name），统一从配置中心读取；
- 对 OpenAI 等闭源在线模型使用 tiktoken 计数；
- 对 HuggingFace/本地模型使用 transformers 的 AutoTokenizer 计数；
- 供检索层 ContextBuilder 做“真实 token 预算”时使用。
"""

from typing import Optional
from functools import lru_cache

from app.config import get_config_manager
from app.infra.logging import get_logger

# 使用 tiktoken 计数的提供商集合（可按需扩展，比如 Azure OpenAI 等）
PROVIDERS_USING_TIKTOKEN = {"openai", "azure_openai"}


class TokenCounter:
    """通用分词计数器接口。"""

    def __init__(self, tokenizer_impl):
        self._impl = tokenizer_impl

    def count_tokens(self, text: str) -> int:
        return self._impl(text)


@lru_cache(maxsize=32)
def get_token_counter(ai_type: Optional[str] = None, provider: Optional[str] = None) -> TokenCounter:
    """根据应用层选择的 ai_type/provider 返回分词计数器。

    规则：
    - 从配置中心读取当前会用到的 provider 与 model 名称；
    - provider == "openai" → 使用 tiktoken.encoding_for_model(model)；
    - 否则 → 使用 transformers.AutoTokenizer.from_pretrained(model)；
    - 若初始化失败，降级为字符长度近似计数。
    """
    cfg = get_config_manager()
    settings = cfg.get_settings(ai_type, provider)  # 包含 provider、model 等
    provider_name = (settings.provider or "").lower()
    model_name = settings.model

    # 使用 tiktoken 的提供商：按集合判断，避免写死单一字符串
    if provider_name in PROVIDERS_USING_TIKTOKEN:
        try:
            import tiktoken  # type: ignore

            enc = tiktoken.encoding_for_model(model_name)

            def _fn(text: str) -> int:
                return len(enc.encode(text or ""))

            return TokenCounter(_fn)
        except ImportError:
            # 无 tiktoken 包，降级
            get_logger(__name__).warning("tiktoken 未安装，分词计数降级为字符长度")
            return TokenCounter(lambda s: len(s or ""))
        except (KeyError, ValueError):
            # 模型名不被 tiktoken 识别
            get_logger(__name__).warning("tiktoken 不识别模型 %s，分词计数降级为字符长度", model_name)
            return TokenCounter(lambda s: len(s or ""))

    # 其他：统一走 HuggingFace 分词器
    try:
        from transformers import AutoTokenizer  # type: ignore

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        def _hf(text: str) -> int:
            return len(tokenizer.encode(text or "", add_special_tokens=False))

        return TokenCounter(_hf)
    except ImportError:
        get_logger(__name__).warning("transformers 未安装，分词计数降级为字符长度")
        return TokenCounter(lambda s: len(s or ""))
    except OSError:
        get_logger(__name__).warning("无法加载分词器 %s，分词计数降级为字符长度", model_name)
        return TokenCounter(lambda s: len(s or ""))


