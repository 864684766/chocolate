"""
命令行工具：清空配置索引下的 Meilisearch 数据。

用法：
  - 直接使用配置索引名：
        python -m tests.test_cleanup_meilisearch

  - 指定索引名：
        python -m tests.test_cleanup_meilisearch --index documents

  - 根据条件删除：
        python -m tests.test_cleanup_meilisearch --filter "status = 'test'"

说明：
  - 仅用于本地/测试环境，请勿在生产误用。
  - 删除操作是异步的，会自动等待任务完成。
"""
from __future__ import annotations

import argparse
import time
from typing import Optional

from app.config import get_config_manager
from app.infra.logging import get_logger

# 任务状态常量
TASK_STATUS_SUCCEEDED = "succeeded"
TASK_STATUS_FAILED = "failed"
TASK_STATUS_ENQUEUED = "enqueued"
TASK_STATUS_PROCESSING = "processing"

# 任务等待配置
TASK_WAIT_TIMEOUT_MS = 30000  # 30秒超时
TASK_POLL_INTERVAL_MS = 500  # 每500毫秒轮询一次


def _get_meilisearch_client():
    """获取 Meilisearch 客户端实例。

    Returns:
        meilisearch.Client: Meilisearch 客户端实例

    Raises:
        ImportError: meilisearch 模块未安装
        RuntimeError: 客户端初始化失败或未配置
    """
    try:
        import meilisearch
    except ImportError:
        raise ImportError(
            "meilisearch 模块未安装，请运行: poetry add meilisearch"
        )

    cfg = get_config_manager()
    meili_cfg = cfg.get_meilisearch_database_config() or {}
    host = str(meili_cfg.get("host", "")).strip()
    if not host:
        raise RuntimeError("Meilisearch host 未配置")

    api_key = str(meili_cfg.get("api_key", "")) or None

    try:
        client = meilisearch.Client(url=host, api_key=api_key)
        return client
    except Exception as e:
        raise RuntimeError(f"初始化 Meilisearch 客户端失败: {e}") from e


def _try_wait_for_task_sdk(index, task_uid: str, timeout_ms: int) -> Optional[bool]:
    """尝试使用 SDK 的 wait_for_task 方法等待任务完成。

    Args:
        index: Meilisearch 索引对象
        task_uid: 任务 UID
        timeout_ms: 超时时间（毫秒）

    Returns:
        Optional[bool]: 如果成功返回任务状态（True/False），如果方法不存在返回 None
    """
    try:
        task = index.wait_for_task(task_uid, timeout_in_ms=timeout_ms)
        return task.status == TASK_STATUS_SUCCEEDED
    except AttributeError:
        return None
    except Exception:
        return None


def _poll_task_status(index, task_uid: str, timeout_seconds: float) -> bool:
    """手动轮询任务状态直到完成或超时。

    Args:
        index: Meilisearch 索引对象
        task_uid: 任务 UID
        timeout_seconds: 超时时间（秒）

    Returns:
        bool: 如果任务成功完成返回 True，否则返回 False
    """
    start_time = time.time()
    poll_interval = TASK_POLL_INTERVAL_MS / 1000.0

    while True:
        # 检查超时
        elapsed = time.time() - start_time
        if elapsed > timeout_seconds:
            return False

        try:
            task = index.get_task(task_uid)
            status = task.status

            # 任务成功完成
            if status == TASK_STATUS_SUCCEEDED:
                return True
            
            # 任务失败
            if status == TASK_STATUS_FAILED:
                return False
            
            # 任务进行中，继续等待
            if status in (TASK_STATUS_ENQUEUED, TASK_STATUS_PROCESSING):
                time.sleep(poll_interval)
                continue
            
            # 其他未知状态，继续等待
            time.sleep(poll_interval)
            continue
        except Exception:
            # 获取任务状态出错，继续等待
            time.sleep(poll_interval)
            continue
    
    # 理论上不会到达这里，但为了类型检查器添加明确的返回
    return False


def _wait_for_task(index, task_uid: str, timeout_ms: int = TASK_WAIT_TIMEOUT_MS) -> bool:
    """等待 Meilisearch 任务完成。

    Args:
        index: Meilisearch 索引对象
        task_uid: 任务 UID
        timeout_ms: 超时时间（毫秒）

    Returns:
        bool: 如果任务成功完成返回 True，否则返回 False

    Notes:
        - 优先使用 SDK 的 wait_for_task 方法（如果可用）
        - 如果 SDK 不支持，则回退到手动轮询任务状态
    """
    result = _try_wait_for_task_sdk(index, task_uid, timeout_ms)
    if result is not None:
        return result

    timeout_seconds = timeout_ms / 1000.0
    return _poll_task_status(index, task_uid, timeout_seconds)


def _execute_delete_all(index, index_name: str) -> bool:
    """执行删除所有文档的操作并等待完成。

    Args:
        index: Meilisearch 索引对象
        index_name: 索引名称

    Returns:
        bool: 如果删除成功返回 True，否则返回 False
    """
    logger = get_logger(__name__)
    
    try:
        # HTTP API: DELETE /indexes/{indexUid}/documents
        # Python SDK: index.delete_all_documents()
        result = index.delete_all_documents()
        success = _wait_for_task(index, result.task_uid)
        
        if success:
            logger.info(f"成功清空 Meilisearch 索引 '{index_name}' 中的所有文档")
        else:
            logger.error(f"清空 Meilisearch 索引 '{index_name}' 失败")
        
        return success
    except AttributeError as e:
        logger.error(
            f"Meilisearch Python SDK 可能不支持 delete_all_documents() 方法: {e}",
            exc_info=True
        )
        return False


def delete_all(index_name: str) -> bool:
    """删除指定索引中的所有文档。

    Args:
        index_name: 索引名称

    Returns:
        bool: 如果删除成功返回 True，否则返回 False

    Notes:
        - 此操作会删除索引中的所有文档，但保留索引本身
        - 删除操作是异步的，会自动等待任务完成
        - 使用 Meilisearch Python SDK 的 delete_all_documents() 方法
    """
    logger = get_logger(__name__)
    
    try:
        client = _get_meilisearch_client()
        index = client.index(index_name)
        return _execute_delete_all(index, index_name)
    except Exception as e:
        logger.error(f"删除 Meilisearch 索引 '{index_name}' 的所有文档时出错: {e}", exc_info=True)
        return False


def _execute_delete_by_filter(index, index_name: str, filter_condition: str) -> bool:
    """执行根据条件删除文档的操作并等待完成。

    Args:
        index: Meilisearch 索引对象
        index_name: 索引名称
        filter_condition: 过滤条件字符串

    Returns:
        bool: 如果删除成功返回 True，否则返回 False

    Notes:
        - 使用 Meilisearch Python SDK 的 delete_documents 方法
        - HTTP API: POST /indexes/{indexUid}/documents/delete
        - 请求体: {"filter": "..."}
    """
    logger = get_logger(__name__)
    
    # HTTP API: POST /indexes/{indexUid}/documents/delete
    # 请求体: {"filter": "..."}
    # Python SDK: index.delete_documents({"filter": filter_condition})
    result = index.delete_documents({"filter": filter_condition})
    
    success = _wait_for_task(index, result.task_uid)
    
    if success:
        logger.info(
            f"成功删除 Meilisearch 索引 '{index_name}' 中符合条件 '{filter_condition}' 的文档"
        )
    else:
        logger.error(
            f"删除 Meilisearch 索引 '{index_name}' 中符合条件 '{filter_condition}' 的文档失败"
        )
    
    return success


def delete_by_filter(index_name: str, filter_condition: str) -> bool:
    """根据条件删除指定索引中的文档。

    Args:
        index_name: 索引名称
        filter_condition: 过滤条件字符串，例如 "status = 'test'" 或 "genre = Horror"

    Returns:
        bool: 如果删除成功返回 True，否则返回 False

    Notes:
        - 过滤条件必须使用 Meilisearch 的过滤语法
        - 用于过滤的字段必须已设置为可过滤属性
        - 删除操作是异步的，会自动等待任务完成
        - 使用 Meilisearch Python SDK 的 delete_documents() 方法，传入 {"filter": "..."}
        - HTTP API: POST /indexes/{indexUid}/documents/delete，请求体: {"filter": "..."}

    Examples:
        delete_by_filter("documents", "status = 'test'")
        delete_by_filter("documents", "genre = Horror OR genre = Comedy")
    """
    logger = get_logger(__name__)
    
    try:
        client = _get_meilisearch_client()
        index = client.index(index_name)
        return _execute_delete_by_filter(index, index_name, filter_condition)
    except Exception as e:
        logger.error(
            f"根据条件删除 Meilisearch 索引 '{index_name}' 的文档时出错: {e}",
            exc_info=True
        )
        return False


def _parse_arguments():
    """解析命令行参数。

    Returns:
        argparse.Namespace: 解析后的命令行参数
    """
    parser = argparse.ArgumentParser(description="清空 Meilisearch 指定索引数据")
    parser.add_argument(
        "--index",
        dest="index",
        default=None,
        help="索引名；不传则读取配置 retrieval.meilisearch.index",
    )
    parser.add_argument(
        "--filter",
        dest="filter",
        default=None,
        help="过滤条件（可选），例如：status = 'test'；如果提供则只删除符合条件的文档",
    )
    return parser.parse_args()


def _get_index_name(args_index: Optional[str]) -> str:
    """从参数或配置中获取索引名称。

    Args:
        args_index: 命令行参数中的索引名（可选）

    Returns:
        str: 索引名称
    """
    if args_index:
        return args_index
    
    cfg = get_config_manager()
    meili_cfg = cfg.get_meilisearch_database_config() or {}
    return str(meili_cfg.get("index", "documents"))


def _handle_delete_result(success: bool, index_name: str, filter_condition: Optional[str]) -> None:
    """处理删除操作的结果并输出信息。

    Args:
        success: 删除操作是否成功
        index_name: 索引名称
        filter_condition: 过滤条件（可选）

    Returns:
        None
    """
    if filter_condition:
        if success:
            print(f"成功删除索引 '{index_name}' 中符合条件 '{filter_condition}' 的文档")
        else:
            print(f"删除索引 '{index_name}' 中符合条件 '{filter_condition}' 的文档失败")
            exit(1)
    else:
        if success:
            print(f"成功清空索引 '{index_name}' 中的所有文档")
        else:
            print(f"清空索引 '{index_name}' 中的所有文档失败")
            exit(1)


def main() -> None:
    """主函数：解析命令行参数并执行删除操作。

    Returns:
        None
    """
    args = _parse_arguments()
    index_name = _get_index_name(args.index)

    if args.filter:
        success = delete_by_filter(index_name, args.filter)
        _handle_delete_result(success, index_name, args.filter)
    else:
        success = delete_all(index_name)
        _handle_delete_result(success, index_name, None)


if __name__ == "__main__":
    main()

