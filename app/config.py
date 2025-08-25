import os
import json
from dataclasses import dataclass
from typing import Optional, Dict, Any
from pathlib import Path

from dotenv import load_dotenv


@dataclass
class Settings:
    provider: str = "google"
    model: str = "gemini-2.5-flash"
    temperature: float = 0.7
    request_timeout: int = 30
    # 大模型的key
    api_key: Optional[str] = None


class ConfigManager:
    """配置管理器，支持从JSON文件读取配置，为后续接入nacos做准备"""
    
    def __init__(self, config_file: str = "config/app_config.json"):
        self.config_file = config_file
        self._config_data = self._load_config()
        self._load_env()
    
    def _load_config(self) -> Dict[str, Any]:
        """从JSON文件加载配置"""
        config_path = Path(self.config_file)
        if not config_path.exists():
            # 如果配置文件不存在，返回默认配置
            return self._get_default_config()
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"警告：无法加载配置文件 {self.config_file}，使用默认配置: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "app": {
                "name": "chocolate",
                "version": "1.0.0",
                "description": "智能AI助手系统"
            },
            "server": {
                "host": "0.0.0.0",
                "port": 8000,
                "reload": True,
                "description": "API服务器配置"
            },
            "llm": {
                "default_provider": "google",
                "default_model": "gemini-2.5-flash",
                "default_temperature": 0.7,
                "request_timeout": 30,
                "description": "大语言模型基础配置"
            },
            "cache": {
                "max_cache_size": 10,
                "description": "模型实例缓存配置"
            },
            "agent": {
                "verbose": False,
                "max_iterations": 5,
                "max_execution_time": 30,
                "handle_parsing_errors": True,
                "return_intermediate_steps": False,
                "description": "Agent执行器配置"
            },
            "ai_types": {
                "gpt4": {
                    "provider": "openai",
                    "model": "gpt-4",
                    "api_key_env": "GPT4_API_KEY",
                    "description": "OpenAI GPT-4 模型",
                    "max_tokens": 4096,
                    "temperature": 0.7
                },
                "gpt35": {
                    "provider": "openai",
                    "model": "gpt-3.5-turbo",
                    "api_key_env": "GPT35_API_KEY",
                    "description": "OpenAI GPT-3.5 Turbo 模型",
                    "max_tokens": 4096,
                    "temperature": 0.7
                },
                "gemini": {
                    "provider": "google",
                    "model": "gemini-2.5-flash",
                    "api_key_env": "GEMINI_API_KEY",
                    "description": "Google Gemini 2.5 Flash 模型",
                    "max_tokens": 8192,
                    "temperature": 0.7
                },
                "claude": {
                    "provider": "anthropic",
                    "model": "claude-3-sonnet-20240229",
                    "api_key_env": "CLAUDE_API_KEY",
                    "description": "Anthropic Claude 3 Sonnet 模型",
                    "max_tokens": 4096,
                    "temperature": 0.7
                }
            },
            "environment": {
                "env_file": ".env.local",
                "default_api_key_env": "DEFAULT_API_KEY",
                "description": "环境变量配置"
            },
            "tools": {
                "available_tools": ["search_docs", "http_get", "calc"],
                "description": "可用工具列表"
            },
            "prompts": {
                "react_template": "你是一个智能助理，能够使用工具来帮助回答问题。你有以下工具可用：\n\n{tools}\n可以使用的工具名称有：{tool_names}\n\n这是你和用户之前的对话历史：{chat_history}\n\n使用以下格式：\n\nQuestion: 用户的问题\nThought: 我需要思考如何回答这个问题\nAction: 要使用的工具名称\nAction Input: 工具的输入参数\nObservation: 工具返回的结果\n... (这个 Thought/Action/Action Input/Observation 可以重复多次)\nThought: 现在我知道最终答案了\nFinal Answer: 给用户的最终回答\n\n重要提示：\n1. 用中文思考和回答\n2. 如果需要计算，使用 calc 工具\n3. 如果需要搜索信息，使用 search_docs 工具\n4. 如果需要获取网页内容，使用 http_get 工具\n5. 可以组合使用多个工具来解决复杂问题\n6. 最终答案要简洁明了\n\nQuestion: {input}\nThought: {agent_scratchpad}",
                "description": "Agent提示词模板"
            }
        }
    
    def _load_env(self):
        """加载环境变量"""
        env_file = self._config_data.get("environment", {}).get("env_file", ".env.local")
        load_dotenv(dotenv_path=env_file)
    
    def get_config(self, section: str = None) -> Dict[str, Any]:
        """获取配置数据"""
        if section:
            return self._config_data.get(section, {})
        return self._config_data
    
    def get_ai_type_mappings(self) -> Dict[str, Dict[str, Any]]:
        """获取AI类型映射"""
        return self._config_data.get("ai_types", {})
    
    def get_settings(self, ai_type: Optional[str] = None, provider: Optional[str] = None) -> Settings:
        """获取配置，支持AI类型和提供商参数"""
        
        # 确定提供商
        if provider:
            cur_provider = provider.lower()
        elif ai_type and ai_type.lower() in self._config_data.get("ai_types", {}):
            cur_provider = self._config_data["ai_types"][ai_type.lower()]["provider"]
        else:
            cur_provider = os.environ.get("DEFAULT_PROVIDER", 
                                        self._config_data["llm"]["default_provider"]).strip().lower()
        
        # 确定模型
        if ai_type and ai_type.lower() in self._config_data.get("ai_types", {}):
            model = self._config_data["ai_types"][ai_type.lower()]["model"]
        else:
            model = ai_type.lower().replace('_', '-') if ai_type else self._config_data["llm"]["default_model"]
        
        # 获取API密钥
        api_key = None
        if ai_type and ai_type.lower() in self._config_data.get("ai_types", {}):
            api_key_env = self._config_data["ai_types"][ai_type.lower()]["api_key_env"]
            api_key = os.environ.get(api_key_env)
        else:
            default_api_key_env = self._config_data["environment"]["default_api_key_env"]
            api_key = os.environ.get(default_api_key_env)
        
        # 其他配置
        temperature_env = os.environ.get("TEMPERATURE")
        try:
            temperature = float(temperature_env) if temperature_env else self._config_data["llm"]["default_temperature"]
        except (ValueError, TypeError):
            temperature = self._config_data["llm"]["default_temperature"]
        
        request_timeout_env = os.environ.get("REQUEST_TIMEOUT")
        try:
            request_timeout = int(request_timeout_env) if request_timeout_env else self._config_data["llm"]["request_timeout"]
        except (ValueError, TypeError):
            request_timeout = self._config_data["llm"]["request_timeout"]
        
        return Settings(
            provider=cur_provider,
            model=model,
            temperature=temperature,
            request_timeout=request_timeout,
            api_key=api_key,
        )
    
    def get_agent_config(self) -> Dict[str, Any]:
        """获取Agent配置"""
        return self._config_data.get("agent", {})
    
    def get_cache_config(self) -> Dict[str, Any]:
        """获取缓存配置"""
        return self._config_data.get("cache", {})
    
    def get_server_config(self) -> Dict[str, Any]:
        """获取服务器配置"""
        return self._config_data.get("server", {})
    
    def get_prompts_config(self) -> Dict[str, Any]:
        """获取提示词配置"""
        return self._config_data.get("prompts", {})
    
    def get_tools_config(self) -> Dict[str, Any]:
        """获取工具配置"""
        return self._config_data.get("tools", {})
    
    def add_ai_type_mapping(self, ai_type: str, provider: str, model: str, api_key_env: str):
        """动态添加AI类型映射"""
        if "ai_types" not in self._config_data:
            self._config_data["ai_types"] = {}
        
        self._config_data["ai_types"][ai_type.lower()] = {
            "provider": provider,
            "model": model,
            "api_key_env": api_key_env
        }
    
    def get_available_ai_types(self) -> list:
        """获取所有可用的AI类型"""
        return list(self._config_data.get("ai_types", {}).keys())
    
    def reload_config(self):
        """重新加载配置文件"""
        self._config_data = self._load_config()
        self._load_env()


# 全局配置管理器实例
_config_manager = ConfigManager()


def get_settings(ai_type: Optional[str] = None, provider: Optional[str] = None) -> Settings:
    """获取配置的便捷函数"""
    return _config_manager.get_settings(ai_type, provider)


def get_config_manager() -> ConfigManager:
    """获取配置管理器实例"""
    return _config_manager


def get_config(section: str = None) -> Dict[str, Any]:
    """获取配置数据的便捷函数"""
    return _config_manager.get_config(section)


