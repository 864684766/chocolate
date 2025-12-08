"""
模型加载器模块

每个模型类型都有独立的加载器文件，便于维护和扩展。
"""

from .base import ModelLoader, ModelType, LoaderConfig, ModelLoaderBase
from .sentence_transformer import SentenceTransformerLoader
from .transformers import TransformersLoader
from .whisper import WhisperLoader
from .clip import CLIPLoader
from .cross_encoder import CrossEncoderLoader

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

