from __future__ import annotations

"""
查询侧轻量清洗工具（仅作用于 query，不修改库内文档）。

职责与原则：
- 统一规范：Unicode NFKC、去控制字符、空白归一化、常见标点与全角/半角统一。
- 单一职责：不做质量判定与过滤（黑名单/重复检测等不在此处）。
- 可配置启用：通过 app_config.json > retrieval.clean_query 开关控制（缺省启用）。

函数均保持方法体精简（≤20行），便于阅读与复用。
"""

from typing import Dict
import re
import unicodedata

def _normalize_unicode_nfkc(text: str) -> str:
    """将文本做 Unicode NFKC 归一化，统一全角/半角与兼容字符。

    Args:
        text: 原始查询字符串
    Returns:
        归一化后的字符串
    """
    return unicodedata.normalize("NFKC", text or "")


def _strip_control_chars(text: str) -> str:
    """移除不可见控制字符（保留换行与制表符可按需修改）。

    Args:
        text: 输入字符串
    Returns:
        去除控制字符后的字符串
    """
    return "".join(ch for ch in (text or "") if ch.isprintable())


def _normalize_whitespace(text: str) -> str:
    """空白归一化：将连续空白折叠为单空格，并去首尾空白。

    Args:
        text: 输入字符串
    Returns:
        空白规范后的字符串
    """
    s = re.sub(r"\s+", " ", text or "")
    return s.strip()


def _normalize_punctuation(text: str) -> str:
    """常见中英标点与全角符号统一为半角形式。

    说明：只做有限映射，保持轻量与可控；如需扩展可在映射表中补充。
    """
    mapping: Dict[int, int] = {
        ord("“"): ord('"'), ord("”"): ord('"'),
        ord("‘"): ord("'"), ord("’"): ord("'"),
        ord("—"): ord("-"), ord("–"): ord("-"), ord("－"): ord("-"),
        ord("（"): ord("("), ord("）"): ord(")"),
        ord("，"): ord(","), ord("。"): ord("."), ord("："): ord(":"),
        ord("；"): ord(";"), ord("！"): ord("!"), ord("？"): ord("?"),
        ord("【"): ord("["), ord("】"): ord("]"), ord("《"): ord("<"), ord("》"): ord(">"),
    }
    return (text or "").translate(mapping)


def clean_query_basic(text: str) -> str:
    """对查询做轻量规范化清洗（不做语义修改）。

    Args:
        text: 原始查询字符串
    Returns:
        规范化后的查询字符串
    """
    s = _normalize_unicode_nfkc(text)
    s = _strip_control_chars(s)
    s = _normalize_punctuation(s)
    s = _normalize_whitespace(s)
    return s


