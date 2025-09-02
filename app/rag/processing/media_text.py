from __future__ import annotations

from typing import Dict, Any
from .interfaces import RawSample, MediaExtractor


class PlainTextExtractor(MediaExtractor):
    """将 bytes 直接按 utf-8 解码为文本；解码失败则回退 latin-1。
    适用于 .txt/.md 等。
    """

    def extract(self, sample: RawSample) -> Dict[str, Any]:
        meta = dict(sample.meta)
        try:
            text = sample.bytes.decode("utf-8")
        except UnicodeDecodeError:
            # 国标编码优先于 latin-1，兼容中文本地文件
            try:
                text = sample.bytes.decode("gb18030")
            except UnicodeDecodeError:
                text = sample.bytes.decode("latin-1", errors="ignore")
        return {"text": text, "meta": meta}


