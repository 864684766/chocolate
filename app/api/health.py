from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict, List, Any

from ..config import get_config_manager
from ..llm_adapters.factory import LLMProviderFactory
from ..core.agent_service import get_agent_service
from .schemas import BaseResponse, ResponseCode, ResponseMessage
from datetime import datetime

router = APIRouter()


class HealthData(BaseModel):
    status: str
    message: str
    ai_types: List[str]
    providers: List[str]
    cached_models: Dict[str, Any]
    cached_chains: Dict[str, Any]
    cache_stats: Dict[str, Any]


@router.get("/health", response_model=BaseResponse)
def health_check():
    """健康检查接口，返回系统状态和可用资源信息"""
    try:
        # 获取配置管理器
        config_manager = get_config_manager()
        
        # 获取可用的AI类型
        ai_types = config_manager.get_available_ai_types()
        
        # 获取已注册的提供商
        providers = LLMProviderFactory.get_registered_providers()
        
        # 获取缓存的模型实例
        cached_models = LLMProviderFactory.get_cached_models()
        cached_models_info = {
            f"{ai_type}_{provider}": "已缓存" 
            for (ai_type, provider) in cached_models.keys()
        }
        
        # 获取缓存的Agent链
        agent_service = get_agent_service()
        cached_chains = agent_service.get_cached_chains()
        cached_chains_info = {
            f"{ai_type}_{provider}": "已缓存" 
            for (ai_type, provider) in cached_chains.keys()
        }
        
        # 缓存统计信息
        cache_stats = {
            "model_cache_size": len(cached_models),
            "chain_cache_size": len(cached_chains),
            "model_cache_max_size": 10,  # 从工厂类获取
            "chain_cache_max_size": 5,   # 从服务类获取
            "model_cache_usage": f"{len(cached_models)}/10",
            "chain_cache_usage": f"{len(cached_chains)}/5"
        }
        
        return BaseResponse(
            code=ResponseCode.OK,
            message=ResponseMessage.SUCCESS,
            data=HealthData(
                status="healthy",
                message="系统运行正常",
                ai_types=ai_types,
                providers=providers,
                cached_models=cached_models_info,
                cached_chains=cached_chains_info,
                cache_stats=cache_stats
            ).dict(),
            request_id="",
            timestamp=datetime.utcnow().isoformat() + "Z",
        )
    except Exception as e:
        return BaseResponse(
            code=ResponseCode.HEALTH_ERROR,
            message=ResponseMessage.HEALTH_ERROR,
            data=HealthData(
                status="unhealthy",
                message=f"系统异常: {str(e)}",
                ai_types=[],
                providers=[],
                cached_models={},
                cached_chains={},
                cache_stats={},
            ).dict(),
            request_id="",
            timestamp=datetime.utcnow().isoformat() + "Z",
        )


class CacheInfoResponse(BaseModel):
    model_cache_count: int
    chain_cache_count: int
    model_cache_keys: List[str]
    chain_cache_keys: List[str]


@router.get("/cache-info", response_model=BaseResponse)
def get_cache_info():
    """获取缓存信息"""
    # 获取模型缓存信息
    cached_models = LLMProviderFactory.get_cached_models()
    model_cache_keys = [f"{ai_type}_{provider}" for (ai_type, provider) in cached_models.keys()]
    
    # 获取Agent链缓存信息
    agent_service = get_agent_service()
    cached_chains = agent_service.get_cached_chains()
    chain_cache_keys = [f"{ai_type}_{provider}" for (ai_type, provider) in cached_chains.keys()]
    
    return BaseResponse(
        code=ResponseCode.OK,
        message=ResponseMessage.SUCCESS,
        data=CacheInfoResponse(
            model_cache_count=len(cached_models),
            chain_cache_count=len(cached_chains),
            model_cache_keys=model_cache_keys,
            chain_cache_keys=chain_cache_keys
        ).dict(),
        request_id="",
        timestamp=datetime.utcnow().isoformat() + "Z",
    )


@router.post("/clear-cache", response_model=BaseResponse)
def clear_cache():
    """清除所有缓存"""
    try:
        # 清除模型缓存
        LLMProviderFactory.clear_cache()
        
        # 清除Agent链缓存
        agent_service = get_agent_service()
        agent_service.clear_cache()
        
        return BaseResponse(
            code=ResponseCode.OK,
            message=ResponseMessage.SUCCESS,
            data={"message": "缓存已清除"},
            request_id="",
            timestamp=datetime.utcnow().isoformat() + "Z",
        )
    except Exception as e:
        return BaseResponse(
            code=ResponseCode.BAD_REQUEST,
            message=ResponseMessage.BAD_REQUEST,
            data=None,
            request_id="",
            timestamp=datetime.utcnow().isoformat() + "Z",
        )