from .api import create_app

# 兼容现有启动脚本 scripts/run_api.py 使用的 `app.api:app`
app = create_app()