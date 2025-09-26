"""解析器模块

包含各种字段解析器的实现，每个解析器负责将自然语言查询转换为对应的 where 条件。

模块结构：
- common.py: 通用解析工具函数
- lang_parser.py: 语言解析器
- media_parser.py: 媒体类型解析器  
- quality_parser.py: 质量分数解析器
- time_parser.py: 时间范围解析器
- tags_parser.py: 标签/关键词解析器
"""

# 延迟导入，避免循环导入问题
def _import_parsers():
    from .common import parse_by_aliases
    from .lang_parser import infer_lang, lang_parser
    from .media_parser import media_type_parser
    from .quality_parser import quality_score_parser
    from .time_parser import created_at_parser
    from .tags_parser import tags_contains_parser
    
    return {
        "parse_by_aliases": parse_by_aliases,
        "infer_lang": infer_lang,
        "lang_parser": lang_parser,
        "media_type_parser": media_type_parser,
        "quality_score_parser": quality_score_parser,
        "created_at_parser": created_at_parser,
        "tags_contains_parser": tags_contains_parser,
    }

# 使用 __getattr__ 实现延迟导入
def __getattr__(name: str):
    parsers = _import_parsers()
    if name in parsers:
        return parsers[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    "parse_by_aliases",
    "infer_lang", 
    "lang_parser",
    "media_type_parser",
    "quality_score_parser", 
    "created_at_parser",
    "tags_contains_parser",
]
