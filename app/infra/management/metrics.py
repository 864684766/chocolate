"""
系统监控指标收集

提供处理速率、向量生成速率、查询延迟、模型可用性等监控功能
"""

import time
import logging
from typing import Dict, Any, Optional, Callable
from functools import wraps
from dataclasses import dataclass, field
from collections import defaultdict, deque
import threading

logger = logging.getLogger(__name__)


@dataclass
class MetricData:
    """指标数据结构"""
    name: str
    value: float
    timestamp: float = field(default_factory=time.time)
    tags: Dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """指标收集器"""
    
    def __init__(self, max_history: int = 1000):
        """
        初始化指标收集器
        
        Args:
            max_history: 最大历史记录数量
        """
        self._metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self._lock = threading.Lock()
    
    def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """
        记录指标
        
        Args:
            name: 指标名称
            value: 指标值
            tags: 标签字典
        """
        with self._lock:
            metric = MetricData(name=name, value=value, tags=tags or {})
            self._metrics[name].append(metric)
    
    def get_metric_stats(self, name: str, window_seconds: int = 300) -> Dict[str, Any]:
        """
        获取指标统计信息
        
        Args:
            name: 指标名称
            window_seconds: 时间窗口（秒）
            
        Returns:
            包含统计信息的字典
        """
        with self._lock:
            if name not in self._metrics:
                return {}
            
            cutoff_time = time.time() - window_seconds
            recent_metrics = [m for m in self._metrics[name] if m.timestamp >= cutoff_time]
            
            if not recent_metrics:
                return {}
            
            values = [m.value for m in recent_metrics]
            return {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "avg": sum(values) / len(values),
                "latest": values[-1] if values else 0
            }
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """获取所有指标的统计信息"""
        return {name: self.get_metric_stats(name) for name in self._metrics.keys()}


# 全局指标收集器实例
_metrics_collector = MetricsCollector()


def record_processing_rate(operation: str, duration: float, count: int = 1):
    """
    记录处理速率指标
    
    Args:
        operation: 操作类型（如 "text_processing", "vector_generation"）
        duration: 处理耗时（秒）
        count: 处理数量
    """
    rate = count / duration if duration > 0 else 0
    _metrics_collector.record_metric(
        f"processing_rate_{operation}",
        rate,
        {"operation": operation}
    )


def record_query_latency(operation: str, latency: float):
    """
    记录查询延迟指标
    
    Args:
        operation: 操作类型（如 "vector_search", "text_retrieval"）
        latency: 延迟时间（秒）
    """
    _metrics_collector.record_metric(
        f"query_latency_{operation}",
        latency,
        {"operation": operation}
    )


def record_model_availability(model_name: str, is_available: bool):
    """
    记录模型可用性指标
    
    Args:
        model_name: 模型名称
        is_available: 是否可用
    """
    _metrics_collector.record_metric(
        "model_availability",
        1.0 if is_available else 0.0,
        {"model": model_name}
    )


def record_system_health(component: str, is_healthy: bool):
    """
    记录系统健康状态指标
    
    Args:
        component: 组件名称
        is_healthy: 是否健康
    """
    _metrics_collector.record_metric(
        "system_health",
        1.0 if is_healthy else 0.0,
        {"component": component}
    )


def monitor_performance(operation_name: str):
    """
    性能监控装饰器
    
    Args:
        operation_name: 操作名称
        
    Returns:
        装饰器函数
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                record_processing_rate(operation_name, duration)
                return result
            except Exception:
                duration = time.time() - start_time
                record_processing_rate(operation_name, duration)
                raise
        return wrapper
    return decorator


def get_metrics_summary() -> Dict[str, Any]:
    """
    获取指标摘要
    
    Returns:
        包含所有指标统计的字典
    """
    return _metrics_collector.get_all_metrics()


def get_metrics_collector() -> MetricsCollector:
    """
    获取指标收集器实例
    
    Returns:
        指标收集器实例
    """
    return _metrics_collector
