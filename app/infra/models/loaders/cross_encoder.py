"""
CrossEncoder 模型加载器

用处：加载 sentence-transformers 的 CrossEncoder 模型，用于检索结果重排。
"""

from __future__ import annotations

import logging
from typing import Any

from .base import LoaderConfig, ModelLoaderBase, ModelType

logger = logging.getLogger(__name__)


class CrossEncoderLoader(ModelLoaderBase):
    """CrossEncoder 模型加载器
    
    用处：加载 sentence-transformers 的 CrossEncoder 模型，用于检索结果重排。
    """
    
    def load(self, config: LoaderConfig) -> Any:
        """加载 CrossEncoder 模型
        
        Args:
            config: 加载器配置，model_name 为 CrossEncoder 模型名称或路径
            
        Returns:
            Any: CrossEncoder 模型实例
        """
        try:
            from sentence_transformers import CrossEncoder
            
            model = CrossEncoder(config.model_name)
            logger.debug(f"CrossEncoder 模型加载成功: {config.model_name}")
            return model
            
        except ImportError:
            raise RuntimeError("sentence-transformers 库未安装，请安装: pip install sentence-transformers")
        except Exception as e:
            raise RuntimeError(f"加载 CrossEncoder 模型失败: {e}")
    
    def get_supported_type(self) -> ModelType:
        return ModelType.CROSS_ENCODER

