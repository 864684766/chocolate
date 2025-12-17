from __future__ import annotations

"""
Neo4j 数据库助手：集中管理驱动初始化与会话执行。
"""

from typing import Any, Dict, List, Optional

from app.config import get_config_manager
from app.infra.logging import get_logger


class Neo4jDBHelper:
    """Neo4j 助手，负责驱动创建与读写会话。"""

    def __init__(self, config_manager=None) -> None:
        """初始化助手，读取 databases.neo4j 配置。"""
        cfg_mgr = config_manager or get_config_manager()
        cfg = cfg_mgr.get_neo4j_config() or {}
        self.url: str = str(cfg.get("url", "")).strip()  # 连接地址
        self.user: str = str(cfg.get("user", "")).strip()  # 用户名
        self.password: str = str(cfg.get("password", "")).strip()  # 密码
        self._driver = None  # 延迟创建的驱动实例
        self.logger = get_logger(__name__)  # 日志记录器

    def has_config(self) -> bool:
        """判断是否配置了 Neo4j。"""
        return bool(self.url and self.user)

    def get_driver(self):
        """获取或创建 Neo4j 驱动。"""
        if self._driver:
            return self._driver
        if not self.has_config():
            raise RuntimeError("Neo4j 未配置，缺少 url 或 user")
        try:
            from neo4j import GraphDatabase
        except ImportError as exc:
            raise ImportError("neo4j 模块未安装，请运行: poetry add neo4j") from exc
        self._driver = GraphDatabase.driver(self.url, auth=(self.user, self.password))
        return self._driver

    def close(self) -> None:
        """关闭驱动连接。"""
        if self._driver:
            self._driver.close()
            self._driver = None

    def run_read(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """执行只读查询并返回结果列表。"""
        driver = self.get_driver()
        with driver.session(default_access_mode="READ") as session:
            result = session.run(query, params or {})
            return [record.data() for record in result]

    def run_write(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """执行写入查询并返回结果列表。"""
        driver = self.get_driver()
        with driver.session(default_access_mode="WRITE") as session:
            result = session.run(query, params or {})
            return [record.data() for record in result]

