"""
审计日志与操作记录

提供关键操作的审计日志记录和追踪功能
"""

import json
import time
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from pathlib import Path

logger = logging.getLogger(__name__)


class AuditAction(Enum):
    """审计操作类型"""
    UPLOAD = "upload"
    DELETE = "delete"
    QUERY = "query"
    UPDATE = "update"
    LOGIN = "login"
    LOGOUT = "logout"
    CONFIG_CHANGE = "config_change"
    PERMISSION_CHANGE = "permission_change"
    BACKUP = "backup"
    RESTORE = "restore"


class AuditLevel(Enum):
    """审计级别"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AuditRecord:
    """审计记录数据结构"""
    timestamp: float
    action: AuditAction
    level: AuditLevel
    user_id: Optional[str]
    session_id: Optional[str]
    resource_type: str
    resource_id: Optional[str]
    details: Dict[str, Any]
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    success: bool = True
    error_message: Optional[str] = None


class AuditLogger:
    """审计日志记录器"""
    
    def __init__(self, log_file: Optional[str] = None, max_records: int = 10000):
        """
        初始化审计日志记录器
        
        Args:
            log_file: 日志文件路径，如果为 None 则只记录到标准日志
            max_records: 内存中最大记录数
        """
        self.log_file = Path(log_file) if log_file else None
        self.max_records = max_records
        self._records: List[AuditRecord] = []
        self._lock = threading.Lock()
        
        # 设置审计专用日志器
        self.audit_logger = logging.getLogger("audit")
        if not self.audit_logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.audit_logger.addHandler(handler)
            self.audit_logger.setLevel(logging.INFO)
    
    def log_action(
        self,
        action: AuditAction,
        resource_type: str,
        resource_id: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        level: AuditLevel = AuditLevel.INFO,
        success: bool = True,
        error_message: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ):
        """
        记录审计操作
        
        Args:
            action: 操作类型
            resource_type: 资源类型
            resource_id: 资源ID
            user_id: 用户ID
            session_id: 会话ID
            details: 操作详情
            level: 审计级别
            success: 操作是否成功
            error_message: 错误信息
            ip_address: IP地址
            user_agent: 用户代理
        """
        record = AuditRecord(
            timestamp=time.time(),
            action=action,
            level=level,
            user_id=user_id,
            session_id=session_id,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details or {},
            ip_address=ip_address,
            user_agent=user_agent,
            success=success,
            error_message=error_message
        )
        
        with self._lock:
            self._records.append(record)
            
            # 保持记录数量在限制内
            if len(self._records) > self.max_records:
                self._records = self._records[-self.max_records:]
        
        # 记录到日志
        self._write_to_log(record)
        
        # 写入文件
        if self.log_file:
            self._write_to_file(record)
    
    def _write_to_log(self, record: AuditRecord):
        """写入标准日志"""
        log_message = (
            f"Action: {record.action.value}, "
            f"Resource: {record.resource_type}:{record.resource_id}, "
            f"User: {record.user_id}, "
            f"Success: {record.success}"
        )
        
        if record.error_message:
            log_message += f", Error: {record.error_message}"
        
        if record.details:
            log_message += f", Details: {json.dumps(record.details, ensure_ascii=False)}"
        
        level_map = {
            AuditLevel.INFO: logging.INFO,
            AuditLevel.WARNING: logging.WARNING,
            AuditLevel.ERROR: logging.ERROR,
            AuditLevel.CRITICAL: logging.CRITICAL
        }
        
        self.audit_logger.log(level_map[record.level], log_message)
    
    def _write_to_file(self, record: AuditRecord):
        """写入审计文件"""
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                record_dict = asdict(record)
                record_dict['action'] = record.action.value
                record_dict['level'] = record.level.value
                f.write(json.dumps(record_dict, ensure_ascii=False) + '\n')
        except Exception as e:
            logger.error(f"写入审计文件失败: {e}")
    
    def get_records(
        self,
        action: Optional[AuditAction] = None,
        user_id: Optional[str] = None,
        resource_type: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        limit: int = 100
    ) -> List[AuditRecord]:
        """
        获取审计记录
        
        Args:
            action: 操作类型过滤
            user_id: 用户ID过滤
            resource_type: 资源类型过滤
            start_time: 开始时间过滤
            end_time: 结束时间过滤
            limit: 返回记录数限制
            
        Returns:
            审计记录列表
        """
        with self._lock:
            filtered_records = self._records
            
            if action:
                filtered_records = [r for r in filtered_records if r.action == action]
            
            if user_id:
                filtered_records = [r for r in filtered_records if r.user_id == user_id]
            
            if resource_type:
                filtered_records = [r for r in filtered_records if r.resource_type == resource_type]
            
            if start_time:
                filtered_records = [r for r in filtered_records if r.timestamp >= start_time]
            
            if end_time:
                filtered_records = [r for r in filtered_records if r.timestamp <= end_time]
            
            return filtered_records[-limit:] if limit > 0 else filtered_records
    
    def get_statistics(self, days: int = 7) -> Dict[str, Any]:
        """
        获取审计统计信息
        
        Args:
            days: 统计天数
            
        Returns:
            统计信息字典
        """
        cutoff_time = time.time() - (days * 24 * 3600)
        
        with self._lock:
            recent_records = [r for r in self._records if r.timestamp >= cutoff_time]
        
        if not recent_records:
            return {}
        
        # 按操作类型统计
        action_counts = {}
        for record in recent_records:
            action = record.action.value
            action_counts[action] = action_counts.get(action, 0) + 1
        
        # 按用户统计
        user_counts = {}
        for record in recent_records:
            if record.user_id:
                user_counts[record.user_id] = user_counts.get(record.user_id, 0) + 1
        
        # 成功率统计
        success_count = sum(1 for r in recent_records if r.success)
        total_count = len(recent_records)
        success_rate = success_count / total_count if total_count > 0 else 0
        
        return {
            "total_actions": total_count,
            "success_rate": success_rate,
            "action_breakdown": action_counts,
            "user_activity": user_counts,
            "time_range": {
                "start": cutoff_time,
                "end": time.time()
            }
        }


# 全局审计日志记录器实例
_audit_logger: Optional[AuditLogger] = None


def initialize_audit_logger(log_file: Optional[str] = None) -> AuditLogger:
    """
    初始化全局审计日志记录器
    
    Args:
        log_file: 日志文件路径
        
    Returns:
        审计日志记录器实例
    """
    global _audit_logger
    _audit_logger = AuditLogger(log_file)
    return _audit_logger


def get_audit_logger() -> Optional[AuditLogger]:
    """
    获取全局审计日志记录器
    
    Returns:
        审计日志记录器实例，如果未初始化则返回 None
    """
    return _audit_logger


def log_audit_action(
    action: AuditAction,
    resource_type: str,
    resource_id: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    level: AuditLevel = AuditLevel.INFO,
    success: bool = True,
    error_message: Optional[str] = None,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None
):
    """
    记录审计操作（便捷函数）
    
    Args:
        action: 操作类型
        resource_type: 资源类型
        resource_id: 资源ID
        user_id: 用户ID
        session_id: 会话ID
        details: 操作详情
        level: 审计级别
        success: 操作是否成功
        error_message: 错误信息
        ip_address: IP地址
        user_agent: 用户代理
    """
    if _audit_logger:
        _audit_logger.log_action(
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            user_id=user_id,
            session_id=session_id,
            details=details,
            level=level,
            success=success,
            error_message=error_message,
            ip_address=ip_address,
            user_agent=user_agent
        )
