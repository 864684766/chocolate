import requests
from langchain_core.tools import tool
from ..config import get_settings

@tool
def http_get(url: str) -> str:
    """HTTP 请求工具：对给定 URL 发送 GET 请求，返回文本内容（截断至 2000 字符）。
    使用环境配置中的 REQUEST_TIMEOUT 作为超时设置。
    """
    print("HTTP GET", url)
    settings = get_settings()
    try:
        resp = requests.get(url, timeout=settings.request_timeout)
        resp.raise_for_status()
        text = resp.text or ""
        return text[:2000]
    except Exception as e:
        return f"请求失败: {e}"