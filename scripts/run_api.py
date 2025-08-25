import uvicorn
from app.config import get_config_manager

if __name__ == "__main__":
    # 从配置文件获取服务器配置
    config_manager = get_config_manager()
    server_config = config_manager.get_server_config()
    
    host = server_config.get("host", "0.0.0.0")
    port = server_config.get("port", 8000)
    reload = server_config.get("reload", True)
    
    print(f"启动API服务器: {host}:{port}")
    uvicorn.run("app.api:app", host=host, port=port, reload=reload)