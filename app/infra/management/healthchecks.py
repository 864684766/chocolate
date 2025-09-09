"""
系统健康检查

提供系统各组件健康状态检查功能
"""

import time
import logging
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import threading

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """健康状态枚举"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """健康检查结果"""
    component: str
    status: HealthStatus
    message: str
    details: Dict[str, Any]
    timestamp: float
    response_time: float


class HealthChecker:
    """健康检查器"""
    
    def __init__(self):
        """初始化健康检查器"""
        self._checks: Dict[str, Callable[[], HealthCheckResult]] = {}
        self._results: Dict[str, HealthCheckResult] = {}
        self._lock = threading.Lock()
    
    def register_check(self, component: str, check_func: Callable[[], HealthCheckResult]):
        """
        注册健康检查函数
        
        Args:
            component: 组件名称
            check_func: 检查函数
        """
        self._checks[component] = check_func
        logger.info(f"注册健康检查: {component}")
    
    def unregister_check(self, component: str):
        """
        取消注册健康检查函数
        
        Args:
            component: 组件名称
        """
        if component in self._checks:
            del self._checks[component]
            logger.info(f"取消注册健康检查: {component}")
    
    def run_check(self, component: str) -> HealthCheckResult:
        """
        运行单个组件的健康检查
        
        Args:
            component: 组件名称
            
        Returns:
            健康检查结果
        """
        if component not in self._checks:
            return HealthCheckResult(
                component=component,
                status=HealthStatus.UNKNOWN,
                message=f"未注册的健康检查: {component}",
                details={},
                timestamp=time.time(),
                response_time=0.0
            )
        
        start_time = time.time()
        try:
            result = self._checks[component]()
            result.response_time = time.time() - start_time
            
            with self._lock:
                self._results[component] = result
            
            return result
            
        except Exception as e:
            error_result = HealthCheckResult(
                component=component,
                status=HealthStatus.UNHEALTHY,
                message=f"健康检查异常: {str(e)}",
                details={"error": str(e)},
                timestamp=time.time(),
                response_time=time.time() - start_time
            )
            
            with self._lock:
                self._results[component] = error_result
            
            return error_result
    
    def run_all_checks(self) -> Dict[str, HealthCheckResult]:
        """
        运行所有组件的健康检查
        
        Returns:
            所有组件的健康检查结果
        """
        results = {}
        for component in self._checks.keys():
            results[component] = self.run_check(component)
        
        return results
    
    def get_last_result(self, component: str) -> Optional[HealthCheckResult]:
        """
        获取组件的最后一次检查结果
        
        Args:
            component: 组件名称
            
        Returns:
            健康检查结果，如果未检查过则返回 None
        """
        with self._lock:
            return self._results.get(component)
    
    def get_all_results(self) -> Dict[str, HealthCheckResult]:
        """
        获取所有组件的检查结果
        
        Returns:
            所有组件的健康检查结果
        """
        with self._lock:
            return self._results.copy()
    
    def get_overall_status(self) -> HealthStatus:
        """
        获取整体系统健康状态
        
        Returns:
            整体健康状态
        """
        with self._lock:
            if not self._results:
                return HealthStatus.UNKNOWN
            
            statuses = [result.status for result in self._results.values()]
            
            if HealthStatus.UNHEALTHY in statuses:
                return HealthStatus.UNHEALTHY
            elif HealthStatus.DEGRADED in statuses:
                return HealthStatus.DEGRADED
            elif all(status == HealthStatus.HEALTHY for status in statuses):
                return HealthStatus.HEALTHY
            else:
                return HealthStatus.UNKNOWN


# 预定义的健康检查函数

def check_database_health() -> HealthCheckResult:
    """检查数据库健康状态"""
    try:
        from ..database.chroma.db_helper import ChromaDBHelper
        
        helper = ChromaDBHelper()
        client = helper.get_client()
        client.heartbeat()
        
        return HealthCheckResult(
            component="database",
            status=HealthStatus.HEALTHY,
            message="数据库连接正常",
            details={"type": "chromadb"},
            timestamp=time.time(),
            response_time=0.0
        )
    except Exception as e:
        return HealthCheckResult(
            component="database",
            status=HealthStatus.UNHEALTHY,
            message=f"数据库连接失败: {str(e)}",
            details={"error": str(e)},
            timestamp=time.time(),
            response_time=0.0
        )


def check_config_health() -> HealthCheckResult:
    """检查配置健康状态"""
    try:
        from ...config import get_config_manager
        
        config_manager = get_config_manager()
        config = config_manager.get_config()
        
        if not config:
            return HealthCheckResult(
                component="config",
                status=HealthStatus.UNHEALTHY,
                message="配置加载失败",
                details={},
                timestamp=time.time(),
                response_time=0.0
            )
        
        return HealthCheckResult(
            component="config",
            status=HealthStatus.HEALTHY,
            message="配置加载正常",
            details={"config_keys": list(config.keys())},
            timestamp=time.time(),
            response_time=0.0
        )
    except Exception as e:
        return HealthCheckResult(
            component="config",
            status=HealthStatus.UNHEALTHY,
            message=f"配置检查失败: {str(e)}",
            details={"error": str(e)},
            timestamp=time.time(),
            response_time=0.0
        )


def check_memory_health() -> HealthCheckResult:
    """检查内存健康状态"""
    try:
        import psutil
        
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        
        if memory_usage > 90:
            status = HealthStatus.UNHEALTHY
            message = f"内存使用率过高: {memory_usage:.1f}%"
        elif memory_usage > 80:
            status = HealthStatus.DEGRADED
            message = f"内存使用率较高: {memory_usage:.1f}%"
        else:
            status = HealthStatus.HEALTHY
            message = f"内存使用正常: {memory_usage:.1f}%"
        
        return HealthCheckResult(
            component="memory",
            status=status,
            message=message,
            details={
                "usage_percent": memory_usage,
                "total": memory.total,
                "available": memory.available
            },
            timestamp=time.time(),
            response_time=0.0
        )
    except ImportError:
        return HealthCheckResult(
            component="memory",
            status=HealthStatus.UNKNOWN,
            message="psutil 未安装，无法检查内存状态",
            details={},
            timestamp=time.time(),
            response_time=0.0
        )
    except Exception as e:
        return HealthCheckResult(
            component="memory",
            status=HealthStatus.UNHEALTHY,
            message=f"内存检查失败: {str(e)}",
            details={"error": str(e)},
            timestamp=time.time(),
            response_time=0.0
        )


def check_disk_health() -> HealthCheckResult:
    """检查磁盘健康状态"""
    try:
        import psutil
        
        disk = psutil.disk_usage('/')
        disk_usage = (disk.used / disk.total) * 100
        
        if disk_usage > 95:
            status = HealthStatus.UNHEALTHY
            message = f"磁盘使用率过高: {disk_usage:.1f}%"
        elif disk_usage > 85:
            status = HealthStatus.DEGRADED
            message = f"磁盘使用率较高: {disk_usage:.1f}%"
        else:
            status = HealthStatus.HEALTHY
            message = f"磁盘使用正常: {disk_usage:.1f}%"
        
        return HealthCheckResult(
            component="disk",
            status=status,
            message=message,
            details={
                "usage_percent": disk_usage,
                "total": disk.total,
                "used": disk.used,
                "free": disk.free
            },
            timestamp=time.time(),
            response_time=0.0
        )
    except ImportError:
        return HealthCheckResult(
            component="disk",
            status=HealthStatus.UNKNOWN,
            message="psutil 未安装，无法检查磁盘状态",
            details={},
            timestamp=time.time(),
            response_time=0.0
        )
    except Exception as e:
        return HealthCheckResult(
            component="disk",
            status=HealthStatus.UNHEALTHY,
            message=f"磁盘检查失败: {str(e)}",
            details={"error": str(e)},
            timestamp=time.time(),
            response_time=0.0
        )


# 全局健康检查器实例
_health_checker: Optional[HealthChecker] = None


def initialize_health_checker() -> HealthChecker:
    """
    初始化全局健康检查器
    
    Returns:
        健康检查器实例
    """
    global _health_checker
    _health_checker = HealthChecker()
    
    # 注册默认健康检查
    _health_checker.register_check("database", check_database_health)
    _health_checker.register_check("config", check_config_health)
    _health_checker.register_check("memory", check_memory_health)
    _health_checker.register_check("disk", check_disk_health)
    
    return _health_checker


def get_health_checker() -> Optional[HealthChecker]:
    """
    获取全局健康检查器
    
    Returns:
        健康检查器实例，如果未初始化则返回 None
    """
    return _health_checker


def get_system_health() -> Dict[str, Any]:
    """
    获取系统健康状态（便捷函数）
    
    Returns:
        系统健康状态信息
    """
    if not _health_checker:
        return {"status": "unknown", "message": "健康检查器未初始化"}
    
    results = _health_checker.run_all_checks()
    overall_status = _health_checker.get_overall_status()
    
    return {
        "status": overall_status.value,
        "timestamp": time.time(),
        "components": {
            component: {
                "status": result.status.value,
                "message": result.message,
                "details": result.details,
                "response_time": result.response_time
            }
            for component, result in results.items()
        }
    }
