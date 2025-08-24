import os
from dataclasses import dataclass
from typing import Optional, Dict, Any

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
    """配置管理器，支持多配置实例和AI类型映射"""
    
    def __init__(self):
        self._load_env()
        self._ai_type_mappings = self._init_ai_type_mappings()
    
    def _load_env(self):
        """加载环境变量"""
        load_dotenv(dotenv_path='.env.local')
    
    def _init_ai_type_mappings(self) -> Dict[str, Dict[str, Any]]:
        """初始化AI类型与提供商、模型的映射关系"""
        return {
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
        }
    
    def get_settings(self, ai_type: Optional[str] = None, provider: Optional[str] = None) -> Settings:
        """获取配置，支持AI类型和提供商参数"""
        
        # 确定提供商
        if provider:
            cur_provider = provider.lower()
        elif ai_type and ai_type.lower() in self._ai_type_mappings:
            cur_provider = self._ai_type_mappings[ai_type.lower()]["provider"]
        else:
            cur_provider = os.environ.get("DEFAULT_PROVIDER", "google").strip().lower()
        
        # 确定模型
        if ai_type and ai_type.lower() in self._ai_type_mappings:
            model = self._ai_type_mappings[ai_type.lower()]["model"]
        else:
            model = ai_type.lower().replace('_', '-') if ai_type else "gemini-2.5-flash"
        
        # 获取API密钥
        api_key = None
        if ai_type and ai_type.lower() in self._ai_type_mappings:
            api_key_env = self._ai_type_mappings[ai_type.lower()]["api_key_env"]
            api_key = os.environ.get(api_key_env)
        else:
            api_key = os.environ.get("DEFAULT_API_KEY")
        
        # 其他配置
        temperature = float(os.environ.get("TEMPERATURE", "0.7"))
        request_timeout = int(os.environ.get("REQUEST_TIMEOUT", "30"))
        
        return Settings(
            provider=cur_provider,
            model=model,
            temperature=temperature,
            request_timeout=request_timeout,
            api_key=api_key,
        )
    
    def add_ai_type_mapping(self, ai_type: str, provider: str, model: str, api_key_env: str):
        """动态添加AI类型映射"""
        self._ai_type_mappings[ai_type.lower()] = {
            "provider": provider,
            "model": model,
            "api_key_env": api_key_env
        }
    
    def get_available_ai_types(self) -> list:
        """获取所有可用的AI类型"""
        return list(self._ai_type_mappings.keys())


# 全局配置管理器实例
_config_manager = ConfigManager()


def get_settings(ai_type: Optional[str] = None, provider: Optional[str] = None) -> Settings:
    """获取配置的便捷函数"""
    return _config_manager.get_settings(ai_type, provider)


def get_config_manager() -> ConfigManager:
    """获取配置管理器实例"""
    return _config_manager


