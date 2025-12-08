"""
模型加载器注册表

自动注册所有内置的模型加载器。
"""

from .loaders import (
    ModelLoader,
    ModelType,
    CLIPLoader,
    CrossEncoderLoader,
    SentenceTransformerLoader,
    TransformersLoader,
    WhisperLoader,
)


def register_default_loaders() -> None:
    """注册所有默认的模型加载器
    
    用处：在模块导入时自动注册所有内置的模型加载器，使 ModelLoader 能够加载各种类型的模型。
    """
    ModelLoader.register_loader(ModelType.SENTENCE_TRANSFORMER, SentenceTransformerLoader)
    ModelLoader.register_loader(ModelType.TRANSFORMERS, TransformersLoader)
    ModelLoader.register_loader(ModelType.WHISPER, WhisperLoader)
    ModelLoader.register_loader(ModelType.CLIP, CLIPLoader)
    ModelLoader.register_loader(ModelType.CROSS_ENCODER, CrossEncoderLoader)


# 自动注册默认加载器
register_default_loaders()

