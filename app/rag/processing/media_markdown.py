from __future__ import annotations

from typing import Dict, Any
import re
from .interfaces import RawSample, MediaExtractor


class MarkdownExtractor(MediaExtractor):
    """最小 Markdown 提取：先按文本解码，再去掉少量标记（标题/代码围栏）。"""

    def extract(self, sample: RawSample) -> Dict[str, Any]:
        from .media_text import PlainTextExtractor
        base = PlainTextExtractor().extract(sample)
        text = base["text"]
        # 简单去除围栏与井号前缀（仅示意，保守处理）
        text = re.sub(r"```[\s\S]*?```", "\n", text)
        text = re.sub(r"^\s*#+\s*", "", text, flags=re.MULTILINE)
        base["text"] = text
        return base


