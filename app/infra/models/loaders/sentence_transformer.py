"""
SentenceTransformer 模型加载器

用处：加载 sentence-transformers 库的模型，用于文本向量化和相似度计算。
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from .base import LoaderConfig, ModelLoaderBase, ModelType

logger = logging.getLogger(__name__)


class SentenceTransformerLoader(ModelLoaderBase):
    """SentenceTransformer 模型加载器
    
    用处：加载 sentence-transformers 库的模型，用于文本向量化和相似度计算。
    """
    
    def load(self, config: LoaderConfig) -> Any:
        """加载 SentenceTransformer 模型
        
        Args:
            config: 加载器配置，model_name 为模型路径或名称
            
        Returns:
            Any: SentenceTransformer 模型实例
        """
        try:
            from sentence_transformers import SentenceTransformer
            
            device = config.device
            if device == "auto":
                device = self._get_auto_device()
            
            model = SentenceTransformer(
                config.model_name,
                device=device,
                cache_folder=config.kwargs.get("cache_folder", None)
            )
            
            logger.debug(f"SentenceTransformer 模型加载成功: {config.model_name}, 设备: {device}")
            return model
            
        except ImportError:
            raise RuntimeError("sentence-transformers 库未安装，请安装: pip install sentence-transformers")
        except Exception as e:
            raise RuntimeError(f"加载 SentenceTransformer 模型失败: {e}")
    
    def get_supported_type(self) -> ModelType:
        return ModelType.SENTENCE_TRANSFORMER
    
    @staticmethod
    def _get_auto_device() -> str:
        """自动检测设备
        
        Returns:
            str: 设备名称 ('cpu', 'cuda', 'mps' 等)
        """
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

