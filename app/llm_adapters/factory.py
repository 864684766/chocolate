from typing import Any, Dict, Type, Optional, Tuple
from ..config import get_settings, Settings, get_config_manager
from ..core.dict_helper import touch_cache_key, pop_lru_item


class LLMProviderFactory:
    """LLM 提供商工厂，基于注册表映射选择 Provider，支持模型实例缓存。"""

    _registry: Dict[str, Type] = {}
    _model_cache: dict = dict()  # LRU缓存实现
    _max_cache_size: int = None  # 最大缓存数量，从配置文件读取
    _bootstrapped: bool = False

    @classmethod
    def register(cls, name: str, provider_cls: Type) -> None:
        """注册提供商类"""
        cls._registry[name.lower()] = provider_cls

    @classmethod
    def unregister(cls, name: str) -> None:
        """注销提供商类"""
        cls._registry.pop(name.lower(), None)

    @classmethod
    def _bootstrap_defaults(cls) -> None:
        """懒加载注册默认提供商"""
        if cls._bootstrapped:
            return
        
        # 注册 Google 提供商
        try:
            from .google import GoogleProvider
            cls.register("google", GoogleProvider)
        except Exception:
            pass
        
        # 注册 OpenAI 提供商
        try:
            from .openai import OpenaiProvider
            cls.register("openai", OpenaiProvider)
        except Exception:
            pass
        
        cls._bootstrapped = True

    @classmethod
    def get_chat_model(cls, ai_type: Optional[str] = None, provider: Optional[str] = None) -> Any:
        """获取聊天模型实例，支持缓存"""
        settings: Settings = get_settings(ai_type, provider)

        if settings.api_key is None:
            raise ValueError('api_key必须配置')
        
        provider_name = (settings.provider or "google").lower()
        
        # 创建缓存键
        cache_key = (ai_type or "default", provider_name)
        
        # 检查缓存（LRU实现）
        if cache_key in cls._model_cache:
            # 移动到末尾（最近使用）
            touch_cache_key(cls._model_cache,cache_key)
            return cls._model_cache[cache_key]

        # 确保默认 Provider 已尝试注册
        cls._bootstrap_defaults()

        provider_cls = cls._registry.get(provider_name)
        if not provider_cls:
            raise NotImplementedError(
                f"Provider '{provider_name}' 尚未注册，请添加实现或安装相应依赖。"
            )
        
        # 创建模型实例并缓存
        model_instance = provider_cls(settings).build_chat_model()
        
        # LRU缓存管理
        cls._model_cache[cache_key] = model_instance
        touch_cache_key(cls._model_cache, cache_key)  # 移动到末尾
        
        # 获取缓存配置
        if cls._max_cache_size is None:
            cache_config = get_config_manager().get_cache_config()
            cls._max_cache_size = cache_config.get("max_cache_size", 10)
        
        # 如果缓存超过最大大小，删除最旧的
        if len(cls._model_cache) > cls._max_cache_size:
            pop_lru_item(cls._model_cache)# 删除最旧的
        
        return model_instance

    @classmethod
    def clear_cache(cls) -> None:
        """清除模型缓存"""
        cls._model_cache.clear()

    @classmethod
    def get_cached_models(cls) -> Dict[Tuple[str, str], Any]:
        """获取当前缓存的所有模型实例"""
        return cls._model_cache.copy()

    @classmethod
    def get_registered_providers(cls) -> list:
        """获取已注册的提供商列表"""
        cls._bootstrap_defaults()
        return list(cls._registry.keys())