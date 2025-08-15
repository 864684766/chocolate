from typing import Any, Dict, Type

from ..config import get_settings, Settings


class LLMProviderFactory:
    """LLM 提供商工厂，基于注册表映射选择 Provider。"""

    _registry: Dict[str, Type] = {}
    _bootstrapped: bool = False

    @classmethod
    def register(cls, name: str, provider_cls: Type) -> None:
        cls._registry[name.lower()] = provider_cls

    @classmethod
    def unregister(cls, name: str) -> None:
        cls._registry.pop(name.lower(), None)

    @classmethod
    def _bootstrap_defaults(cls) -> None:
        if cls._bootstrapped:
            return
        # 懒加载注册，避免导入时触发第三方依赖
        try:
            from .google import GoogleProvider  # noqa: F401
            cls.register("google", GoogleProvider)
        except Exception:
            # 允许没有 google 依赖的环境，仅在真正请求该 provider 时报错
            pass
        cls._bootstrapped = True

    @classmethod
    def get_chat_model(cls) -> Any:
        settings: Settings = get_settings()
        provider = (settings.provider or "google").lower()

        # 确保默认 Provider 已尝试注册
        cls._bootstrap_defaults()

        provider_cls = cls._registry.get(provider)
        if not provider_cls:
            raise NotImplementedError(
                f"Provider '{provider}' 尚未注册，请添加实现或安装相应依赖。"
            )
        return provider_cls(settings).build_chat_model()