from __future__ import annotations

"""
命令行工具：清空 Neo4j 数据库中所有节点与关系（开发/测试使用）。

用法：
    python -m tests.cleanup_neo4j
"""

import argparse

from app.infra.database.neo4j.db_helper import Neo4jDBHelper
from app.infra.logging import get_logger


logger = get_logger(__name__)  # 工具日志记录器


def _get_helper() -> Neo4jDBHelper:
    """
    获取 Neo4j 助手实例。

    Returns:
        Neo4jDBHelper: 已加载配置的 Neo4j 助手。
    """
    return Neo4jDBHelper()


def _clear_all(helper: Neo4jDBHelper) -> int:
    """
    执行清空操作：删除所有节点与关系。

    Args:
        helper (Neo4jDBHelper): 已配置的 Neo4j 助手。

    Returns:
        int: 受影响的记录数估计值（执行成功返回 0，失败抛出异常）。
    """
    query = "MATCH (n) DETACH DELETE n"
    helper.run_write(query, {})
    return 0


def _parse_args() -> argparse.Namespace:
    """
    解析命令行参数。

    Returns:
        argparse.Namespace: 解析结果。
    """
    parser = argparse.ArgumentParser(description="清空 Neo4j 数据库（开发/测试用途）")
    parser.add_argument(
        "--force",
        action="store_true",
        help="跳过交互确认，直接清空所有节点与关系",
    )
    return parser.parse_args()


def _confirm(force: bool) -> bool:
    """本工具仅用于开发/测试，去除交互直接执行。"""
    return True


def main() -> None:
    """
    主函数：解析参数、确认并执行清空。

    Returns:
        None
    """
    args = _parse_args()
    if not _confirm(args.force):
        print("已取消操作")
        return
    helper = _get_helper()
    if not helper.has_config():
        raise RuntimeError("Neo4j 未配置，无法执行清空")
    _clear_all(helper)
    logger.info("Neo4j 数据库已清空（所有节点与关系已删除）")
    print("Neo4j 清空完成")


if __name__ == "__main__":
    main()

