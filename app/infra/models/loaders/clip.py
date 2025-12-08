"""
CLIP 模型加载器

用处：加载 CLIP 模型和处理器，用于图像-文本相似度计算。
"""

from __future__ import annotations

import logging
from typing import Any

from .base import LoaderConfig, ModelLoaderBase, ModelType

logger = logging.getLogger(__name__)


class CLIPLoader(ModelLoaderBase):
    """CLIP 模型加载器
    
    用处：加载 CLIP 模型和处理器，用于图像-文本相似度计算。
    """
    
    def load(self, config: LoaderConfig) -> Any:
        """加载 CLIP 模型和处理器
        
        Args:
            config: 加载器配置，model_name 为 CLIP 模型名称
            
        Returns:
            Tuple[Any, Any]: (CLIPModel, CLIPProcessor) 元组
        """
        try:
            from transformers import CLIPModel, CLIPProcessor
            
            clip_model = CLIPModel.from_pretrained(config.model_name)
            processor = CLIPProcessor.from_pretrained(config.model_name)
            
            logger.debug(f"CLIP 模型和处理器加载成功: {config.model_name}")
            return clip_model, processor
            
        except ImportError:
            raise RuntimeError("transformers 库未安装，请安装: pip install transformers")
        except Exception as e:
            raise RuntimeError(f"加载 CLIP 模型失败: {e}")
    
    def get_supported_type(self) -> ModelType:
        return ModelType.CLIP

