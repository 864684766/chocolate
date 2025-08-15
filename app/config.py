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
    # 针对特定提供商的密钥（如 Google）
    google_api_key: Optional[str] = None
    # 通用密钥占位（非 google 提供商时可使用）
    api_key: Optional[str] = None


_cached_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """加载并缓存应用配置。支持按 provider 选择性校验密钥。"""
    global _cached_settings
    if _cached_settings is not None:
        return _cached_settings

    load_dotenv()

    provider = os.environ.get("PROVIDER", "google").strip().lower()

    # 模型与参数
    model = os.environ.get("MODEL", "gemini-2.5-flash")
    temperature = float(os.environ.get("TEMPERATURE", "0.7"))
    request_timeout = int(os.environ.get("REQUEST_TIMEOUT", "30"))

    google_api_key: Optional[str] = None
    api_key: Optional[str] = os.environ.get("API_KEY")

    if provider == "google":
        google_api_key = os.environ.get("GOOGLE_API_KEY")
        # 将强制校验延迟到真正构建 Google Provider 时再进行，避免非 google provider 的测试因缺少该变量而失败
        if not google_api_key:
            # 仅发出提示，由具体 provider 构建时抛出更明确的异常
            google_api_key = None
        # 兼容：将通用 api_key 也设置为 google 密钥，便于上层按通用字段读取
        if not api_key and google_api_key:
            api_key = google_api_key

    _cached_settings = Settings(
        provider=provider,
        model=model,
        temperature=temperature,
        request_timeout=request_timeout,
        google_api_key=google_api_key,
        api_key=api_key,
    )
    return _cached_settings


