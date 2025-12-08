"""
通用模型加载器核心实现

提供统一的模型加载接口和缓存机制。
"""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, Optional, Tuple, Type

from ....config import get_config_manager

logger = logging.getLogger(__name__)


class ModelType(str, Enum):
    """模型类型枚举"""
    SENTENCE_TRANSFORMER = "sentence_transformer"
    TRANSFORMERS = "transformers"
    WHISPER = "whisper"
    CLIP = "clip"
    CROSS_ENCODER = "cross_encoder"
    CUSTOM = "custom"


class LoaderConfig:
    """模型加载器配置
    
    用处：封装模型加载所需的配置参数，包括模型路径、设备、其他参数等。
    
    Attributes:
        model_name (str): 模型名称或路径
        device (str, optional): 设备类型，默认为 "auto"
        model_type (ModelType, optional): 模型类型，默认为 ModelType.CUSTOM
        **kwargs: 其他模型特定的参数
    """
    
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        model_type: ModelType = ModelType.CUSTOM,
        **kwargs: Any
    ):
        self.model_name = model_name
        self.device = device
        self.model_type = model_type
        self.kwargs = kwargs
    
    def get_cache_key(self) -> Tuple[str, ...]:
        """生成缓存键
        
        Returns:
            Tuple[str, ...]: 缓存键元组，包含模型名称、设备、模型类型等
        """
        key_parts = [self.model_type.value, self.model_name, self.device]
        # 添加其他影响模型加载的参数
        if self.kwargs:
            sorted_kwargs = tuple(sorted(self.kwargs.items()))
            key_parts.append(str(sorted_kwargs))
        return tuple(key_parts)


class ModelLoaderBase(ABC):
    """模型加载器基类
    
    用处：定义模型加载器的统一接口，所有具体的模型加载器都需要继承此类。
    """
    
    @abstractmethod
    def load(self, config: LoaderConfig) -> Any:
        """加载模型
        
        Args:
            config: 加载器配置对象
            
        Returns:
            Any: 加载的模型实例
            
        Raises:
            RuntimeError: 模型加载失败时抛出
        """
        pass
    
    @abstractmethod
    def get_supported_type(self) -> ModelType:
        """获取支持的模型类型
        
        Returns:
            ModelType: 支持的模型类型
        """
        pass


class ModelLoader:
    """通用模型加载器
    
    用处：提供统一的模型加载接口，自动管理缓存，支持多种模型类型。
    通过注册不同的加载器实现，可以支持各种模型框架。
    
    特性：
    - 模块级缓存，避免重复加载
    - 线程安全
    - 支持LRU缓存策略
    - 可扩展的加载器注册机制
    """
    
    # 模块级缓存字典，键为缓存键元组，值为模型实例
    _model_cache: Dict[Tuple[str, ...], Any] = {}
    
    # 加载器注册表，键为模型类型，值为加载器类
    _loaders: Dict[ModelType, Type[ModelLoaderBase]] = {}
    
    # 线程锁，保证线程安全
    _lock = threading.Lock()
    
    # 最大缓存数量
    _max_cache_size: Optional[int] = None
    
    @classmethod
    def register_loader(cls, model_type: ModelType, loader_class: Type[ModelLoaderBase]) -> None:
        """注册模型加载器
        
        用处：注册特定类型的模型加载器，使 ModelLoader 能够加载该类型的模型。
        
        Args:
            model_type: 模型类型
            loader_class: 加载器类，必须继承 ModelLoaderBase
            
        Raises:
            TypeError: 如果 loader_class 不是类或不继承 ModelLoaderBase
        """
        # 检查是否为类（而不是实例）
        if not isinstance(loader_class, type):
            raise TypeError(f"loader_class 必须是类，而不是实例: {type(loader_class)}")
        
        # 检查是否继承 ModelLoaderBase
        if not issubclass(loader_class, ModelLoaderBase):
            raise TypeError(f"加载器类必须继承 ModelLoaderBase: {loader_class}")
        
        cls._loaders[model_type] = loader_class
        logger.debug(f"已注册模型加载器: {model_type.value} -> {loader_class.__name__}")
    
    @classmethod
    def unregister_loader(cls, model_type: ModelType) -> None:
        """注销模型加载器
        
        Args:
            model_type: 模型类型
        """
        cls._loaders.pop(model_type, None)
        logger.debug(f"已注销模型加载器: {model_type.value}")
    
    @classmethod
    def load_model(cls, config: LoaderConfig) -> Any:
        """加载模型（带缓存）
        
        用处：根据配置加载模型，如果缓存中存在则直接返回，否则加载并缓存。
        
        Args:
            config: 加载器配置对象
            
        Returns:
            Any: 加载的模型实例
            
        Raises:
            RuntimeError: 模型加载失败或未找到对应的加载器时抛出
        """
        cache_key = config.get_cache_key()
        
        # 检查缓存
        with cls._lock:
            if cache_key in cls._model_cache:
                logger.debug(f"从缓存加载模型: {config.model_name} (类型: {config.model_type.value})")
                return cls._model_cache[cache_key]
        
        # 获取加载器
        loader_class = cls._loaders.get(config.model_type)
        if not loader_class:
            raise RuntimeError(
                f"未找到模型类型 '{config.model_type.value}' 的加载器，"
                f"请先注册: ModelLoader.register_loader({config.model_type}, YourLoaderClass)"
            )
        
        # 加载模型
        logger.info(f"正在加载模型: {config.model_name} (类型: {config.model_type.value}, 设备: {config.device})")
        loader = loader_class()
        model = loader.load(config)
        
        # 缓存模型
        with cls._lock:
            cls._model_cache[cache_key] = model
            cls._touch_cache_key(cache_key)
            cls._enforce_cache_limit()
        
        logger.info(f"模型加载成功并已缓存: {config.model_name}")
        return model
    
    @classmethod
    def _touch_cache_key(cls, key: Tuple[str, ...]) -> None:
        """将缓存键移动到末尾（LRU策略）
        
        Args:
            key: 缓存键
        """
        if key in cls._model_cache:
            value = cls._model_cache.pop(key)
            cls._model_cache[key] = value
    
    @classmethod
    def _enforce_cache_limit(cls) -> None:
        """强制执行缓存大小限制
        
        用处：当缓存超过最大大小时，删除最旧的缓存项（LRU策略）。
        """
        if cls._max_cache_size is None:
            cache_config = get_config_manager().get_cache_config()
            cls._max_cache_size = cache_config.get("max_cache_size", 10)
        
        while len(cls._model_cache) > cls._max_cache_size:
            # 删除最旧的项（第一个）
            oldest_key = next(iter(cls._model_cache))
            cls._model_cache.pop(oldest_key)
            logger.debug(f"缓存已满，删除最旧模型: {oldest_key}")
    
    @classmethod
    def clear_cache(cls) -> None:
        """清除所有模型缓存
        
        用处：释放内存，通常在需要重新加载模型或内存不足时调用。
        """
        with cls._lock:
            count = len(cls._model_cache)
            cls._model_cache.clear()
            logger.info(f"已清除 {count} 个模型缓存")
    
    @classmethod
    def remove_from_cache(cls, config: LoaderConfig) -> bool:
        """从缓存中移除指定模型
        
        Args:
            config: 加载器配置对象
            
        Returns:
            bool: 是否成功移除
        """
        cache_key = config.get_cache_key()
        with cls._lock:
            if cache_key in cls._model_cache:
                cls._model_cache.pop(cache_key)
                logger.debug(f"已从缓存移除模型: {config.model_name}")
                return True
        return False
    
    @classmethod
    def get_cache_info(cls) -> Dict[str, Any]:
        """获取缓存信息
        
        Returns:
            Dict[str, Any]: 包含缓存大小、最大大小等信息的字典
        """
        if cls._max_cache_size is None:
            cache_config = get_config_manager().get_cache_config()
            cls._max_cache_size = cache_config.get("max_cache_size", 10)
        
        return {
            "cache_size": len(cls._model_cache),
            "max_cache_size": cls._max_cache_size,
            "cached_models": list(cls._model_cache.keys())
        }

