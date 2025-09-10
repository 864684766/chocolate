import uvicorn
from app.api import create_app
from app.infra.logging import setup_logging

# 初始化日志系统
setup_logging()

app = create_app()
if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)