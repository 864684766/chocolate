"""通用解析工具函数

提供解析器共用的工具函数，如别名匹配、N-gram 处理等。
"""

from typing import Dict, Optional
import re


def parse_by_aliases(query: str, mapping: Dict[str, str]) -> Optional[str]:
    """在给定映射中匹配别名并返回标准值。

    策略：
    1) 优先进行"词边界/分词"匹配（英文/下划线单词、连续中文片段）；
    2) 其次对中文片段做 2~4 字 N-gram 命中；
    3) 最后仍未命中则返回 None。

    Args:
        query: 原始查询文本。
        mapping: 别名到标准值的映射字典（键应为小写或原样中文）。

    Returns:
        命中时返回标准值，未命中返回 None。
    """
    if not mapping:
        return None
    q = query.lower()
    # 1) 英文/下划线 token
    word_tokens = set(re.findall(r"[a-z_]+", q, flags=re.IGNORECASE))
    for alias, val in mapping.items():
        a = alias.lower()
        if a and a.isascii():
            if a in word_tokens:
                return val
    # 2) 中文连续片段与 N-gram
    cjk_spans = re.findall(r"[\u4e00-\u9fff]+", query)
    if cjk_spans:
        # 直接片段包含
        for span in cjk_spans:
            for alias, val in mapping.items():
                if any(0x4e00 <= ord(ch) <= 0x9fff for ch in alias):
                    if alias in span:
                        return val
        # N-gram 命中（2~4）
        grams = set()
        for span in cjk_spans:
            n = len(span)
            for g in range(2, min(4, n) + 1):
                for i in range(0, n - g + 1):
                    grams.add(span[i : i + g])
        for alias, val in mapping.items():
            if alias in grams:
                return val
    return None
