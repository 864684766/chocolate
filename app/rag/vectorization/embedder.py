from typing import List, Dict, Any
import logging
import torch

from app.infra.models import ModelLoader, ModelType, LoaderConfig

logger = logging.getLogger(__name__)


class Embedder:
    """文本向量化器，基于 sentence-transformers 实现。

    功能：
    - 将文本列表转换为向量列表
    - 支持批量处理和并行编码
    - 自动处理设备选择（CPU/GPU）
    - 提供模型信息查询
    """

    def __init__(self, config):
        """
        初始化向量化器

        Args:
            config: VectorizationConfig 配置对象，包含模型路径、设备等参数
        
        说明：
            - 采用延迟加载策略，模型在首次调用 encode 时才加载
            - 避免在不需要向量化的场景下加载模型，提升性能
        """
        self.config = config
        self.model = None
        self._model_loaded = False

    def _load_model(self) -> None:
        """
        加载 sentence-transformers 模型

        说明：
        - 根据配置的 device 参数选择计算设备
        - 支持本地模型路径和 HuggingFace 模型名
        - 首次加载可能需要下载模型文件
        - 使用通用模型加载器，自动处理缓存
        """
        try:
            device = self._get_device()
            
            # 使用通用模型加载器加载模型（自动缓存）
            config = LoaderConfig(
                model_name=self.config.model,
                device=device,
                model_type=ModelType.SENTENCE_TRANSFORMER,
                cache_folder=None  # 使用默认缓存目录
            )
            self.model = ModelLoader.load_model(config)
            
            logger.info(f"模型加载成功，向量维度: {self.model.get_sentence_embedding_dimension()}")
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise RuntimeError(f"无法加载向量化模型 {self.config.model}: {e}")

    def _get_device(self) -> str:
        """
        确定计算设备

        Returns:
            str: 设备名称 ('cpu', 'cuda', 'mps' 等)
        """
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"  # Apple Silicon
            else:
                return "cpu"
        return self.config.device

    def _ensure_model_loaded(self) -> None:
        """
        确保模型已加载（延迟加载）
        
        用处：在首次需要向量化时才加载模型，避免不必要的模型加载开销。
        如果模型已加载，则直接返回。
        """
        if not self._model_loaded:
            self._load_model()
            self._model_loaded = True

    def encode(self, texts: List[str]) -> List[List[float]]:
        """
        将文本列表转换为向量列表

        Args:
            texts: 待编码的文本列表

        Returns:
            List[List[float]]: 对应的向量列表，每个向量是一个浮点数列表

        说明：
            - 空文本会被跳过
            - 超长文本会被截断到 max_sequence_length
            - 返回的向量已进行 L2 归一化
            - 采用延迟加载，首次调用时才会加载模型
        """
        if not texts:
            return []

        # 延迟加载模型（仅在需要向量化时才加载）
        self._ensure_model_loaded()

        # 过滤空文本
        valid_texts = [text for text in texts if text and text.strip()]
        if not valid_texts:
            logger.warning("所有输入文本都为空，返回空向量列表")
            return []

        try:
            # 批量编码
            embeddings = self.model.encode(
                valid_texts,
                batch_size=self.config.batch_size,
                normalize_embeddings=True,  # L2 归一化
                show_progress_bar=False,
                convert_to_tensor=False,  # 返回 numpy 数组
                device=self._get_device()
            )
            
            # 转换为列表格式
            result = [embedding.tolist() for embedding in embeddings]
            logger.debug(f"成功编码 {len(result)} 个文本，向量维度: {len(result[0]) if result else 0}")
            return result
            
        except Exception as e:
            logger.error(f"文本编码失败: {e}")
            raise RuntimeError(f"向量编码过程出错: {e}")

    def encode_parallel(self, texts: List[str]) -> List[List[float]]:
        """
        并行编码接口（与 encode 相同，保持接口兼容性）

        Args:
            texts: 待编码的文本列表

        Returns:
            List[List[float]]: 对应的向量列表

        说明：
        - 当前实现与 encode 相同
        - 未来可扩展为真正的并行处理（多进程/多线程）
        """
        return self.encode(texts)

    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息

        Returns:
            Dict[str, Any]: 包含模型名称、向量维度、设备等信息
        
        说明：
            - 如果模型未加载，返回状态为 "not_loaded"
            - 如果模型已加载，返回完整的模型信息
        """
        if not self._model_loaded or not self.model:
            return {
                "model": self.config.model,
                "status": "not_loaded",
                "device": self._get_device(),
                "batch_size": self.config.batch_size
            }
        
        return {
            "model": self.config.model,
            "embedding_dimension": self.model.get_sentence_embedding_dimension(),
            "device": self._get_device(),
            "batch_size": self.config.batch_size,
            "status": "loaded"
        }


