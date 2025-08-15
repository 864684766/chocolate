import argparse
import requests
import json
from typing import Optional


def call_agent_api(query: str, api_url: str = "http://localhost:8000") -> str:
    """通过 HTTP API 调用 Agent。"""
    try:
        response = requests.post(
            f"{api_url}/agent/invoke",
            json={"input": query},
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        response.raise_for_status()
        result = response.json()
        return result.get("answer", "未收到回答")
    except requests.exceptions.ConnectionError:
        return "连接失败：请确保 API 服务已启动（运行 python scripts/run_api.py）"
    except requests.exceptions.Timeout:
        return "请求超时：API 服务响应时间过长"
    except requests.exceptions.RequestException as e:
        return f"API 调用错误：{e}"
    except json.JSONDecodeError:
        return "API 返回格式错误：无法解析 JSON 响应"


def run_demo() -> None:
    """通过 API 调用演示两个示例问题。"""
    print("\n--- API 演示模式 ---")
    print("注意：需要先启动 API 服务（python scripts/run_api.py）\n")
    
    # 示例1：历史问题
    print("--- 历史问题示例 ---")
    history_question = "谁是秦始皇？他做了什么？"
    print(f"问题: {history_question}")
    history_answer = call_agent_api(history_question)
    print(f"\n--- Agent 的回答 ---")
    print(history_answer)
    
    print("\n" + "="*50 + "\n")
    
    # 示例2：诗意描述
    print("--- 诗意向导示例 ---")
    travel_question = "给我描述一下北京故宫，用诗意的语言。"
    print(f"问题: {travel_question}")
    travel_answer = call_agent_api(travel_question)
    print(f"\n--- Agent 的回答 ---")
    print(travel_answer)


def run_agent_cli(query: Optional[str]) -> None:
    """通过 API 运行 Agent 客户端。"""
    if query:
        # 单轮模式
        answer = call_agent_api(query)
        print(answer)
        return

    # 交互模式
    print("进入交互模式，输入内容后回车；Ctrl+C 退出。")
    print("注意：需要先启动 API 服务（python scripts/run_api.py）\n")
    
    try:
        while True:
            user = input("你: ")
            if not user.strip():
                continue
            answer = call_agent_api(user)
            print(f"助理: {answer}\n")
    except KeyboardInterrupt:
        print("\n已退出。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LangChain Agent / Demo 启动入口")
    parser.add_argument("--demo", action="store_true", help="运行原始 LLM 演示")
    parser.add_argument("-q", "--query", default=None, help="单轮问答（Agent 模式）")
    args = parser.parse_args()

    if args.demo:
        print("\n--- 运行 Demo ---")
        run_demo()
    else:
        print("\n--- 运行 Agent ---")
        run_agent_cli(args.query)