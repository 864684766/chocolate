"""
通用模型加载器模块

提供统一的模型加载和缓存机制，支持多种模型类型。
"""

from .loaders import (
    ModelLoader,
    ModelType,
    LoaderConfig,
    ModelLoaderBase,
    SentenceTransformerLoader,
    TransformersLoader,
    WhisperLoader,
    CLIPLoader,
    CrossEncoderLoader,
)
from . import registry  # 触发自动注册

__all__ = [
    "ModelLoader",
    "ModelType",
    "LoaderConfig",
    "ModelLoaderBase",
    "SentenceTransformerLoader",
    "TransformersLoader",
    "WhisperLoader",
    "CLIPLoader",
    "CrossEncoderLoader",
]

