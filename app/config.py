import json
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
from pathlib import Path


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

    def _load_config(self) -> Dict[str, Any]:
        """从JSON文件加载配置"""
        current_script_dir = Path(__file__).parent.parent

        # 将当前脚本目录与配置文件名拼接，形成配置文件的绝对路径
        # 这样无论你从哪个目录启动程序，它总能找到相对于 config.py 的 config/app_config.json
        config_path = current_script_dir / self.config_file

        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_file}")

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            raise RuntimeError(f"无法加载配置文件 {self.config_file}: {e}")

    def get_config(self, section: str = None) -> Dict[str, Any]:
        """获取配置数据"""
        if section:
            return self._config_data.get(section, {})
        return self._config_data

    def get_settings(self, ai_type: Optional[str] = None, provider: Optional[str] = None) -> Settings:
        """获取配置，支持AI类型和提供商参数"""
        if provider:
            cur_provider = provider.lower()
            model = ai_type if ai_type else self._config_data["llm"]["default_model"]
        elif ai_type:
            # 通过别名查找模型
            found = self._find_model_by_alias(ai_type.lower())
            if found:
                cur_provider, model = found
            else:
                # 如果没找到别名，尝试直接作为模型名使用
                cur_provider = self._config_data["llm"]["default_provider"]
                model = ai_type.lower().replace('_', '-')
        else:
            cur_provider = self._config_data["llm"]["default_provider"]
            model = self._config_data["llm"]["default_model"]

        # 获取API密钥和模型配置 - 从providers配置中读取（统一字段 model）
        api_key = None
        temperature = self._config_data["llm"]["default_temperature"]

        if cur_provider in self._config_data.get("providers", {}):
            provider_config = self._config_data["providers"][cur_provider]
            api_key = provider_config.get("api_key")

            # 如果指定了具体模型，获取该模型的配置
            if model in provider_config.get("models", {}):
                model_config = provider_config["models"][model]
                temperature = model_config.get("temperature", temperature)

        request_timeout = self._config_data["llm"]["request_timeout"]

        return Settings(
            provider=cur_provider,
            model=model,
            temperature=temperature,
            request_timeout=request_timeout,
            api_key=api_key,
        )

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

    def get_providers_config(self) -> Dict[str, Any]:
        """获取提供商配置"""
        return self._config_data.get("providers", {})

    def get_provider_config(self, provider: str) -> Dict[str, Any]:
        """获取指定提供商的配置"""
        return self._config_data.get("providers", {}).get(provider, {})

    def get_model_config(self, provider: str, model: str) -> Dict[str, Any]:
        """获取指定提供商和模型的配置"""
        provider_config = self.get_provider_config(provider)
        return provider_config.get("models", {}).get(model, {})

    def get_vector_database_config(self) -> Dict[str, Any]:
        """获取向量数据库的配置（仅 databases.chroma）。"""
        databases = self._config_data.get("databases", {})
        return databases.get("chroma", {})

    def get_meilisearch_database_config(self) -> Dict[str, Any]:
        """获取 Meilisearch 连接配置（仅 databases.meilisearch）。"""
        databases = self._config_data.get("databases", {})
        return databases.get("meilisearch", {})

    def get_neo4j_config(self) -> Dict[str, Any]:
        """获取 Neo4j 连接配置（databases.neo4j）。"""
        databases = self._config_data.get("databases", {})
        return databases.get("neo4j", {})

    def get_media_processing_config(self) -> Dict[str, Any]:
        """获取媒体处理配置"""
        return self._config_data.get("media_processing", {})

    def get_image_captioning_config(self) -> Dict[str, Any]:
        """获取图像描述配置"""
        return self.get_media_processing_config().get("image_captioning", {})

    def get_ocr_config(self) -> Dict[str, Any]:
        """获取OCR配置"""
        return self.get_media_processing_config().get("ocr", {})

    def get_video_processing_config(self) -> Dict[str, Any]:
        """获取视频处理配置"""
        return self.get_media_processing_config().get("video_processing", {})

    def get_speech_recognition_config(self) -> Dict[str, Any]:
        """获取语音识别配置
        
        用处：从 media_processing.speech_recognition 读取配置，
        供视频和音频处理共用。
        
        Returns:
            Dict[str, Any]: 语音识别配置字典，包含 model 等配置项
        """
        return self.get_media_processing_config().get("speech_recognition", {})

    def get_logging_config(self) -> Dict[str, Any]:
        """获取日志配置"""
        return self._config_data.get("logging", {})

    def get_language_processing_config(self) -> Dict[str, Any]:
        """获取语言处理配置"""
        return self._config_data.get("language_processing", {})

    def get_chinese_processing_config(self) -> Dict[str, Any]:
        """获取中文处理配置"""
        return self.get_language_processing_config().get("chinese", {})

    def _find_model_by_alias(self, alias: str) -> Optional[Tuple[str, str]]:
        """通过别名查找提供商和模型"""
        providers = self._config_data.get("providers", {})
        for provider_name, provider_config in providers.items():
            models = provider_config.get("models", {})
            for model_name, model_config in models.items():
                aliases = model_config.get("aliases", [])
                if alias in aliases:
                    return provider_name, model_name
        return None

    def get_available_ai_types(self) -> list:
        """获取所有可用的AI类型别名"""
        aliases = []
        providers = self._config_data.get("providers", {})
        for provider_config in providers.values():
            models = provider_config.get("models", {})
            for model_config in models.values():
                aliases.extend(model_config.get("aliases", []))
        return aliases

    def reload_config(self):
        """重新加载配置文件"""
        self._config_data = self._load_config()


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
