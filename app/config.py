import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv


@dataclass
class Settings:
    provider: str = "google"
    model: str = "gemini-2.5-flash"
    temperature: float = 0.7
    request_timeout: int = 30
    # 大模型的key
    api_key: Optional[str] = None


_cached_settings: Optional[Settings] = None


def get_settings(ai_type:Optional[str]=None,provider:Optional[str]=None) -> Settings:
    """加载并缓存应用配置。支持按 provider 选择性校验密钥。"""
    global _cached_settings
    if _cached_settings is not None:
        return _cached_settings

    load_dotenv(dotenv_path='.env.local')

    cur_provider = os.environ.get("DEFAULT_PROVIDER").strip().lower() if provider.lower() is None else provider.lower()

    # 模型与参数

    temperature = float(os.environ.get("TEMPERATURE", "0.7"))
    request_timeout = int(os.environ.get("REQUEST_TIMEOUT", "30"))
    api_key = os.environ.get("DEFAULT_API_KEY")
    model = ai_type.lower().replace('_','-')
    if ai_type is not None:
        print(f"{ai_type.upper()}_API_KEY")
        api_key = os.environ.get(f"{ai_type.upper()}_API_KEY")

    _cached_settings = Settings(
        provider=cur_provider,
        model=model,
        temperature=temperature,
        request_timeout=request_timeout,
        api_key=api_key,
    )
    return _cached_settings


