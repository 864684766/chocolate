"""
Whisper 模型加载器

用处：加载 OpenAI Whisper 模型，用于音频和视频的语音识别。
"""

from __future__ import annotations

import logging
from typing import Any

from .base import LoaderConfig, ModelLoaderBase, ModelType

logger = logging.getLogger(__name__)


class WhisperLoader(ModelLoaderBase):
    """Whisper 模型加载器
    
    用处：加载 OpenAI Whisper 模型，用于音频和视频的语音识别。
    """
    
    def load(self, config: LoaderConfig) -> Any:
        """加载 Whisper 模型
        
        Args:
            config: 加载器配置，model_name 为模型大小（如 "base", "small", "medium", "large"）
            
        Returns:
            Any: Whisper 模型实例
        """
        try:
            import whisper
            
            # 移除 "whisper-" 前缀（如果存在）
            model_name = config.model_name
            if model_name.startswith("whisper-"):
                model_name = model_name.replace("whisper-", "")
            
            model = whisper.load_model(model_name)
            logger.debug(f"Whisper 模型加载成功: {model_name}")
            return model
            
        except ImportError:
            raise RuntimeError("openai-whisper 库未安装，请安装: pip install openai-whisper")
        except Exception as e:
            raise RuntimeError(f"加载 Whisper 模型失败: {e}")
    
    def get_supported_type(self) -> ModelType:
        return ModelType.WHISPER

