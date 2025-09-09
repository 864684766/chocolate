from typing import List, Optional
import time


class Embedder:
    """简单嵌入器占位实现。

    说明：此处仅占位，实际可集成 sentence-transformers。
    """

    def __init__(self, config):
        self.config = config

    @staticmethod
    def encode(texts: List[str]) -> List[Optional[List[float]]]:
        # 占位：返回 None 列表，待替换为真实向量
        return [[0.0, 0.0, 0.0] for _ in texts]

    def encode_parallel(self, texts: List[str]) -> List[Optional[List[float]]]:
        # 占位并模拟耗时
        time.sleep(0.01)
        return self.encode(texts)

    def get_model_info(self) -> dict:
        return {"model_name": self.config.model_name}


