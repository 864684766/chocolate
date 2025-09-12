from __future__ import annotations

from typing import Any, Dict, Union, Iterable, List
import re
import unicodedata
from app.config import get_config_manager

Primitive = Union[str, int, float, bool, None]


def _is_primitive(value: Any) -> bool:
    """判断是否为基础类型。

    用处：ChromaDB 的 metadata 值必须是基础类型。

    Args:
        value (Any): 任意值。

    Returns:
        bool: 是否为 str/int/float/bool/None。
    """
    return isinstance(value, (str, int, float, bool)) or value is None


def build_metadata_from_meta(meta: Dict[str, Any]) -> Dict[str, Primitive]:
    """根据配置白名单构造写库用 metadatas（仅基础类型，扁平）。

    兼容两种白名单写法：
    - 旧版：List[str]
    - 新版：List[{field: str, type: str}]
    """
    cfg = get_config_manager().get_config("vectorization")
    wl_raw = cfg.get("metadata_whitelist", [])
    types_map: Dict[str, str] = {}
    if wl_raw and isinstance(wl_raw[0], dict):
        whitelist: Iterable[str] = [str(x.get("field")) for x in wl_raw if x.get("field")]
        types_map = {str(x.get("field")): str(x.get("type")) for x in wl_raw if x.get("field")}
    else:
        whitelist = wl_raw

    out: Dict[str, Primitive] = {}
    for k in whitelist:
        if k not in meta:
            continue
        v = meta.get(k)
        # 跳过 None/空串，Chroma 不接受 null 值
        # 注意：字段缺失是正常的，ChromaDB 支持稀疏元数据
        if v is None or v == "":
            continue
        t = types_map.get(k)
        if t == "number":
            # 仅接受 int/float，尝试安全转换
            if isinstance(v, (int, float)):
                out[k] = v  # type: ignore[assignment]
            else:
                try:
                    num = float(str(v).strip())
                    # 尽量以 int 写入
                    out[k] = int(num) if num.is_integer() else num  # type: ignore[assignment]
                except Exception:
                    continue
        elif t == "boolean":
            if isinstance(v, bool):
                out[k] = v  # type: ignore[assignment]
            else:
                sv = str(v).strip().lower()
                if sv in ("true", "1", "yes"): out[k] = True  # type: ignore[assignment]
                elif sv in ("false", "0", "no"): out[k] = False  # type: ignore[assignment]
        elif t == "string":
            sv = str(v)
            if sv != "":
                out[k] = sv  # type: ignore[assignment]
        else:
            # 无类型声明时，接受基础类型且非空
            if _is_primitive(v):
                if isinstance(v, str) and v == "":
                    continue
                out[k] = v  # type: ignore[assignment]
    return out


# ---- 新增：统一规范化入口函数 ----

def normalize_text_for_vector(text: str) -> str:
    """将文本规范化用于向量写入与稳定ID生成（不截断）。

    处理策略：
    - Unicode 规范化 NFKC，统一全角/半角与兼容字符
    - 去除控制字符（除换行/制表外），随后将所有空白折叠为单个空格
    - 去首尾空白
    - 不做长度截断，确保写库文本完整
    """
    if not isinstance(text, str):
        return ""
    # Unicode 规范化
    t = unicodedata.normalize("NFKC", text)
    # 去控制字符（保留常规空白，再统一折叠）
    t = "".join(ch for ch in t if (ch.isprintable() or ch in ("\n", "\t", " ")))
    # 空白折叠：所有空白 → 单空格
    t = re.sub(r"\s+", " ", t).strip()
    return t


def normalize_meta_for_vector(meta: Dict[str, Any]) -> Dict[str, Primitive]:
    """按白名单归一化元数据（单次遍历，类型化默认值）。

    规则：
    - 仅保留白名单键；
    - 若键缺失：按类型补默认值（str→""，int/float→0，bool→False）；
      已知应为数值的键：chunk_index/page_number/start_pos/end_pos/region_index/min_x/max_x/min_y/max_y；
    - `media_type` 若缺失，默认 "text"；
    - 复用展开逻辑，将图片/边界信息拉平。
    """
    cfg = get_config_manager().get_config("vectorization")
    # 支持两种写法：旧版 List[str] 与新版 List[{field,type}]
    wl_raw = cfg.get("metadata_whitelist", [])
    if wl_raw and isinstance(wl_raw[0], dict):
        whitelist: List[str] = [str(x.get("field")) for x in wl_raw if x.get("field")]
        types_map: Dict[str, str] = {str(x.get("field")): str(x.get("type")) for x in wl_raw if x.get("field")}
    else:
        whitelist = list(wl_raw)
        types_map = {}
    base: Dict[str, Primitive] = {}
    # 数值字段：完全由 types 映射推断（不再硬编码默认集合）
    numeric_from_types = {k for k, t in types_map.items() if t == "number"}
    numeric_keys = numeric_from_types

    for k in whitelist:
        v = meta.get(k)
        if _is_primitive(v):
            base[k] = v  # type: ignore[assignment]
        else:
            if k in numeric_keys:
                base[k] = 0
            elif k == "media_type":
                base[k] = "text"
            else:
                base[k] = ""

    # 统一约定：meta 为扁平结构，不做展开

    return base


