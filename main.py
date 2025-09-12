import uvicorn
from app.api import create_app
from app.infra.logging import setup_logging
from app.config import get_config_manager

# 初始化日志系统
setup_logging()

app = create_app()

if __name__ == '__main__':
    # 从配置文件读取服务器配置
    config = get_config_manager().get_config()
    server_config = config.get("server", {})
    
    host = server_config.get("host", "0.0.0.0")
    port = server_config.get("port", 8000)
    reload = server_config.get("reload", False)
    
    if reload:
        # 使用字符串形式启用热重载
        uvicorn.run("main:app", host=host, port=port, reload=True)
    else:
        # 直接传递应用对象
        uvicorn.run(app, host=host, port=port, reload=False)