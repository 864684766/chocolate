"""
命令行工具：清空配置集合下的 ChromaDB 数据。

用法：
  - 直接使用配置集合名：
        python -m tests.cleanup_chromadb

  - 指定集合名：
        python -m tests.cleanup_chromadb --collection documents

说明：
  - 仅用于本地/测试环境，请勿在生产误用。
"""
from __future__ import annotations

import argparse
from typing import List

from app.infra.database.chroma.db_helper import ChromaDBHelper
from app.rag.vectorization.config import VectorizationConfig


def delete_all(collection_name: str) -> int:
    """删除指定集合中的所有数据。

    Args:
        collection_name: 集合名

    Returns:
        int: 删除的记录数（最佳努力）
    """
    helper = ChromaDBHelper()
    try:
        col = helper.get_collection(collection_name, create_if_not_exists=False)
    except Exception:
        return 0

    total_deleted = 0
    while True:
        peek = col.peek(limit=10_000)
        ids: List[str] = peek.get("ids", [])
        if not ids:
            break
        helper.delete(collection_name=collection_name, ids=ids)
        total_deleted += len(ids)
    return total_deleted


def main() -> None:
    parser = argparse.ArgumentParser(description="清空 ChromaDB 指定集合数据")
    parser.add_argument(
        "--collection",
        dest="collection",
        default=None,
        help="集合名；不传则读取配置 vectorization.database.collection_name",
    )
    args = parser.parse_args()

    cfg = VectorizationConfig.from_config_manager()
    collection = args.collection or cfg.collection_name
    deleted = delete_all(collection)
    print(f"Deleted {deleted} records from collection '{collection}'.")


if __name__ == "__main__":
    main()


