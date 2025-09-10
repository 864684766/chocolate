"""
日志配置模块

提供全局日志配置功能，支持控制台和文件输出。
"""

from .config import setup_logging, get_logger

__all__ = ["setup_logging", "get_logger"]