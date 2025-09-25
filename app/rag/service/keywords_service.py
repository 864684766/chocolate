from __future__ import annotations

"""
关键词服务：为入库与检索共用的关键词抽取能力。

设计目标：
- 单一职责：只做关键词抽取与简单适配。
- 统一对外接口：extract_keyphrases(text, lang, topk, method)。
- 可配置：从 app_config.json 读取默认 method/topk/lang。

注意：
- 中量默认实现采用 TextRank（基于 jieba.analyse.textrank）。
- 轻量/重量策略留作扩展位，方法体不超过 20 行。
"""

from typing import List
import re

from app.config import get_config_manager


def _get_defaults() -> tuple[str, int, str]:
    """读取配置默认值（method/topk/lang）。

    返回:
    - (method, topk, lang)
    """
    cfg = get_config_manager().get_config("metadata") or {}
    meta_cfg = cfg.get("keywords") or {}
    method = str(meta_cfg.get("method", "textrank"))
    topk = int(meta_cfg.get("topk", 10))
    lang = str(meta_cfg.get("lang", "auto"))
    return method, topk, lang


def _load_stopwords(lang: str) -> set[str]:
    """加载停用词集合。

    优先级：配置文件路径 > 内置（use_builtin=true）> 空集合。
    """
    cfg = get_config_manager().get_config("metadata") or {}
    kw_cfg = cfg.get("keywords") or {}
    sw_cfg = (kw_cfg.get("stopwords") or {})
    path = str(sw_cfg.get("path" or ""))
    if path:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return {ln.strip() for ln in f if ln.strip() and not ln.startswith("#")}
        except (OSError, UnicodeError):
            pass
    return set()


def extract_keyphrases(text: str,
                       lang: str | None = None,
                       topk: int | None = None,
                       method: str | None = None) -> List[str]:
    """抽取关键词（对外统一入口）。

    参数：
    - text (str): 输入文本。
    - lang (str|None): 语言，auto/zh/en 等；None 则读取配置默认。
    - topk (int|None): 抽取数量上限；None 则读取配置默认。
    - method (str|None): 抽取方法（textrank|light|keybert|none）；None 读取默认。

    返回：
    - List[str]: 关键词列表（去重、去空）。
    """
    if not isinstance(text, str) or not text.strip():
        return []
    def_method, def_topk, def_lang = _get_defaults()
    method = (method or def_method).lower()
    topk = int(topk or def_topk)
    lang = (lang or def_lang).lower()
    if method == "none" or topk <= 0:
        return []
    # 非中文时，优先使用轻量方案，避免中文特化算法误判
    if method == "textrank":
        if lang.startswith("zh"):
            # TextRank 走原算法；可选前置轻量过滤
            return _extract_textrank(text, topk)
        return _extract_light(text, topk, lang)
    if method == "light":
        return _extract_light(text, topk, lang)
    if method == "keybert":
        # 预留重量方案占位，避免引入重依赖
        return _extract_light(text, topk, lang)
    # 默认回退
    return _extract_light(text, topk, lang)


def _extract_textrank(text: str, topk: int) -> List[str]:
    """基于 TextRank 的中量抽取实现（中文友好）。

    参数：
    - text (str): 输入文本。
    - topk (int): 抽取数量上限。

    返回：
    - List[str]: 关键词列表。
    """
    try:
        from jieba.analyse import textrank  # type: ignore
    except ImportError:
        return _extract_light(text, topk)
    words = textrank(text, topK=max(1, topk)) or []
    return [w.strip() for w in words if isinstance(w, str) and w.strip()]


def _extract_light(text: str, topk: int, lang: str | None = None) -> List[str]:
    """轻量方案：简单分词+去停用词+频次排序（近似）。

    参数：
    - text (str): 输入文本。
    - topk (int): 抽取数量上限。
    - lang (str|None): 语言提示（en 走英文分词；其他默认中文分词）。

    返回：
    - List[str]: 关键词列表。
    """
    stop = _load_stopwords((lang or "").lower())
    if (lang or "").startswith("en"):
        # 英文：基于非字母数字切分，转小写，过滤短词
        tokens = [t for t in re.split(r"\W+", text.lower()) if len(t) > 2 and t not in stop]
    else:
        try:
            import jieba  # type: ignore
        except ImportError:
            tokens = [t for t in text.split() if len(t) > 1 and t not in stop]
        else:
            tokens = [t for t in jieba.cut(text) if len(t.strip()) > 1 and t.strip() not in stop]
    # 简单频次统计
    freq: dict[str, int] = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    ranked = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [w for w, _ in ranked[: max(1, topk)]]


