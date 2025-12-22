"""ChromaDB where 条件转换为 Meilisearch filter 的工具函数。

职责：
- 将 ChromaDB 的 where 语法（如 {"field": {"$eq": "value"}}）转换为 Meilisearch 的 filter 语法
- 支持基本操作符：$eq, $ne, $gt, $gte, $lt, $lte, $in
- 支持逻辑组合：$and, $or

说明：
- Meilisearch filter 使用字符串表达式，如 "field = value"
- $in 操作符转换为 Meilisearch 的 IN 操作，如 "field IN [value1, value2]"
- 保留 $contains 的转换逻辑作为兼容性处理（向后兼容）
"""

from typing import Any, Dict, List, Optional


def convert_where_to_filter(where: Optional[Dict[str, Any]]) -> Optional[List[str]]:
    """将 ChromaDB where 条件转换为 Meilisearch filter 列表。

    Args:
        where: ChromaDB 的 where 条件字典，如 {"field": {"$eq": "value"}} 或 {"$and": [...]}

    Returns:
        Optional[List[str]]: Meilisearch filter 表达式列表；无条件时返回 None。
        每个元素是一个 filter 字符串，如 ["field = value", "other_field >= 10"]
    """
    if not where:
        return None

    filters: List[str] = []
    _convert_node(where, filters)
    return filters if filters else None


def _convert_node(node: Dict[str, Any], filters: List[str]) -> None:
    """递归转换 where 节点。

    Args:
        node: where 条件节点（可能是操作符或字段条件）
        filters: 累积的 filter 表达式列表
    """
    # 处理逻辑组合操作符
    if "$and" in node:
        # Meilisearch 中，多个 filter 字符串会被自动 AND 连接
        # 所以 $and 中的每个子句都作为独立的 filter 添加
        for clause in node["$and"]:
            _convert_node(clause, filters)
        return

    if "$or" in node:
        # Meilisearch 的 OR 需要使用数组语法：["field1 = value1", "field2 = value2"]
        # 但更复杂的 OR 需要嵌套数组，这里简化处理
        or_filters: List[str] = []
        for clause in node["$or"]:
            clause_filters: List[str] = []
            _convert_node(clause, clause_filters)
            if clause_filters:
                # 对于 OR，将每个子句的 filter 用 OR 连接
                # 注意：Meilisearch 的 filter 数组默认是 AND，OR 需要特殊处理
                # 这里简化：将 OR 子句合并为一个字符串表达式
                if len(clause_filters) == 1:
                    or_filters.append(clause_filters[0])
                else:
                    or_filters.append(f"({' AND '.join(clause_filters)})")
        if or_filters:
            # Meilisearch 不支持直接的 OR 语法，这里使用括号和 OR 关键字
            # 注意：这可能需要 Meilisearch 的特定版本支持
            filters.append(f"({' OR '.join(or_filters)})")
        return

    # 处理字段条件
    for field, condition in node.items():
        if isinstance(condition, dict):
            _convert_field_condition(field, condition, filters)
        else:
            # 简写形式：{"field": "value"} 等同于 {"field": {"$eq": "value"}}
            filters.append(_format_filter(field, "=", condition))


def _convert_field_condition(field: str, condition: Dict[str, Any], filters: List[str]) -> None:
    """转换单个字段的条件。

    Args:
        field: 字段名
        condition: 条件字典，如 {"$eq": "value"}
        filters: 累积的 filter 表达式列表
    """
    for op, value in condition.items():
        if op == "$eq":
            filters.append(_format_filter(field, "=", value))
        elif op == "$ne":
            filters.append(_format_filter(field, "!=", value))
        elif op == "$gt":
            filters.append(_format_filter(field, ">", value))
        elif op == "$gte":
            filters.append(_format_filter(field, ">=", value))
        elif op == "$lt":
            filters.append(_format_filter(field, "<", value))
        elif op == "$lte":
            filters.append(_format_filter(field, "<=", value))
        elif op == "$in":
            # Meilisearch 的 IN 操作
            if isinstance(value, list) and value:
                values_str = ", ".join(_format_value(v) for v in value)
                filters.append(f"{field} IN [{values_str}]")
        elif op == "$contains":
            # 数组字段的包含检查，转换为 Meilisearch 的 IN 操作
            filters.append(f"{field} IN [{_format_value(value)}]")
        # 忽略不支持的操作符，记录日志（可选）


def _format_filter(field: str, operator: str, value: Any) -> str:
    """格式化单个 filter 表达式。

    Args:
        field: 字段名
        operator: 操作符（=, !=, >, >=, <, <=）
        value: 值

    Returns:
        str: 格式化后的 filter 表达式，如 "field = 'value'"
    """
    formatted_value = _format_value(value)
    return f"{field} {operator} {formatted_value}"


def _format_value(value: Any) -> str:
    """格式化值为 Meilisearch filter 中的字符串表示。

    Args:
        value: 要格式化的值（字符串、数字、布尔值等）

    Returns:
        str: 格式化后的值字符串
    """
    if isinstance(value, str):
        # 转义单引号并包裹在引号中
        escaped = value.replace("'", "\\'")
        return f"'{escaped}'"
    elif isinstance(value, bool):
        return "true" if value else "false"
    elif isinstance(value, (int, float)):
        return str(value)
    else:
        # 其他类型转为字符串
        escaped = str(value).replace("'", "\\'")
        return f"'{escaped}'"

