import argparse

from app.agent import build_agent


def main() -> None:
    parser = argparse.ArgumentParser(description="Run interactive LangChain agent")
    parser.add_argument("-q", "--query", help="单次问答问题（非交互模式）", default=None)
    args = parser.parse_args()

    chain = build_agent()

    if args.query:
        result = chain.invoke({"input": args.query})
        answer = result.get("output", result) if isinstance(result, dict) else result
        print(answer)
        return

    print("进入交互模式，输入内容后回车；Ctrl+C 退出。\n")
    try:
        while True:
            user = input("你: ")
            if not user.strip():
                continue
            result = chain.invoke({"input": user})
            answer = result.get("output", result) if isinstance(result, dict) else result
            print(f"助理: {answer}\n")
    except KeyboardInterrupt:
        print("\n已退出。")


if __name__ == "__main__":
    main()


