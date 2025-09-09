"""
配置热更新管理

提供配置文件监控、热更新和多环境配置隔离功能
"""

import json
import time
import logging
from typing import Dict, Any, Optional, Callable, List
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import threading

logger = logging.getLogger(__name__)


class ConfigChangeHandler(FileSystemEventHandler):
    """配置文件变更处理器"""
    
    def __init__(self, config_path: str, callback: Callable[[Dict[str, Any]], None]):
        """
        初始化配置变更处理器
        
        Args:
            config_path: 配置文件路径
            callback: 配置变更回调函数
        """
        self.config_path = config_path
        self.callback = callback
        self.last_modified = 0
    
    def on_modified(self, event):
        """文件修改事件处理"""
        if event.is_directory:
            return
        
        if event.src_path == self.config_path:
            # 避免重复触发
            current_time = time.time()
            if current_time - self.last_modified < 1.0:
                return
            self.last_modified = current_time
            
            try:
                self._reload_config()
            except Exception as e:
                logger.error(f"配置重载失败: {e}")
    
    def _reload_config(self):
        """重新加载配置"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                new_config = json.load(f)
            
            logger.info(f"配置文件 {self.config_path} 已更新，正在重载...")
            self.callback(new_config)
            logger.info("配置重载完成")
            
        except Exception as e:
            logger.error(f"重载配置文件失败: {e}")


class ConfigWatcher:
    """配置监控器"""
    
    def __init__(self, config_path: str, environment: str = "default"):
        """
        初始化配置监控器
        
        Args:
            config_path: 配置文件路径
            environment: 环境名称
        """
        self.config_path = Path(config_path)
        self.environment = environment
        self.observer = Observer()
        self.handler: Optional[ConfigChangeHandler] = None
        self._config_cache: Dict[str, Any] = {}
        self._callbacks: List[Callable[[Dict[str, Any]], None]] = []
        self._lock = threading.Lock()
    
    def add_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """
        添加配置变更回调
        
        Args:
            callback: 回调函数
        """
        with self._lock:
            self._callbacks.append(callback)
    
    def remove_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """
        移除配置变更回调
        
        Args:
            callback: 回调函数
        """
        with self._lock:
            if callback in self._callbacks:
                self._callbacks.remove(callback)
    
    def _notify_callbacks(self, new_config: Dict[str, Any]):
        """通知所有回调函数"""
        with self._lock:
            for callback in self._callbacks:
                try:
                    callback(new_config)
                except Exception as e:
                    logger.error(f"配置变更回调执行失败: {e}")
    
    def start_watching(self):
        """开始监控配置文件"""
        if not self.config_path.exists():
            logger.error(f"配置文件不存在: {self.config_path}")
            return
        
        self.handler = ConfigChangeHandler(str(self.config_path), self._notify_callbacks)
        self.observer.schedule(self.handler, str(self.config_path.parent), recursive=False)
        self.observer.start()
        
        logger.info(f"开始监控配置文件: {self.config_path}")
    
    def stop_watching(self):
        """停止监控配置文件"""
        if self.observer.is_alive():
            self.observer.stop()
            self.observer.join()
            logger.info("停止监控配置文件")
    
    def load_config(self) -> Dict[str, Any]:
        """
        加载配置文件
        
        Returns:
            配置字典
        """
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # 应用环境特定配置
            config = self._apply_environment_config(config)
            self._config_cache = config
            return config
            
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            return self._config_cache
    
    def _apply_environment_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        应用环境特定配置
        
        Args:
            config: 基础配置
            
        Returns:
            应用环境配置后的配置
        """
        if self.environment == "default":
            return config
        
        # 查找环境特定配置
        env_config = config.get("environments", {}).get(self.environment, {})
        if not env_config:
            return config
        
        # 合并配置
        merged_config = config.copy()
        self._deep_merge(merged_config, env_config)
        
        return merged_config
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]):
        """
        深度合并配置字典
        
        Args:
            base: 基础配置
            override: 覆盖配置
        """
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
    
    def get_config_value(self, key_path: str, default: Any = None) -> Any:
        """
        获取配置值
        
        Args:
            key_path: 配置键路径，用点分隔（如 "database.host"）
            default: 默认值
            
        Returns:
            配置值
        """
        keys = key_path.split('.')
        value = self._config_cache
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def is_watching(self) -> bool:
        """检查是否正在监控"""
        return self.observer.is_alive()


# 全局配置监控器实例
_config_watcher: Optional[ConfigWatcher] = None


def initialize_config_watcher(config_path: str, environment: str = "default") -> ConfigWatcher:
    """
    初始化全局配置监控器
    
    Args:
        config_path: 配置文件路径
        environment: 环境名称
        
    Returns:
        配置监控器实例
    """
    global _config_watcher
    _config_watcher = ConfigWatcher(config_path, environment)
    _config_watcher.start_watching()
    return _config_watcher


def get_config_watcher() -> Optional[ConfigWatcher]:
    """
    获取全局配置监控器
    
    Returns:
        配置监控器实例，如果未初始化则返回 None
    """
    return _config_watcher


def add_config_callback(callback: Callable[[Dict[str, Any]], None]):
    """
    添加全局配置变更回调
    
    Args:
        callback: 回调函数
    """
    if _config_watcher:
        _config_watcher.add_callback(callback)


def remove_config_callback(callback: Callable[[Dict[str, Any]], None]):
    """
    移除全局配置变更回调
    
    Args:
        callback: 回调函数
    """
    if _config_watcher:
        _config_watcher.remove_callback(callback)
