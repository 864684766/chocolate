import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. 加载环境变量 (这通常放在文件顶部，因为它们在任何函数或主逻辑运行前都需要)
load_dotenv()

# 2. 获取 Google API Key
google_api_key = os.environ.get("GOOGLE_API_KEY")

# 检查 API Key 是否成功加载，提供更友好的提示
if not google_api_key:
    # 这里的错误检查可以合并，避免重复的 ValueError
    raise ValueError(
        "GOOGLE_API_KEY not found in environment variables. "
        "Please ensure it's set in a .env file (e.g., GOOGLE_API_KEY=\"YOUR_KEY\") or as an environment variable."
    )
else:
    # DEBUG: 仅在调试时打印密钥前缀，完成后请移除或注释此行
    print(f"DEBUG: GOOGLE_API_KEY loaded (truncated): {google_api_key[:5]}...")


# ====================================================================
# 将所有主要执行逻辑放入 if __name__ == "__main__": 块中
# ====================================================================
if __name__ == "__main__":
    print("\n--- LangChain + Google Gemini 示例开始运行 ---")

    try:
        # 3. 初始化 ChatGoogleGenerativeAI 模型
        # model="gemini-pro" 是 Google Gemini 的文本模型
        # temperature 控制模型的创造性（0.0 最保守，1.0 最创造）
        # 如果您遇到 404 错误，可以尝试 model="gemini-2.0-flash" 或其他可用模型
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", # 或者 "gemini-2.0-flash"
            google_api_key=google_api_key,
            temperature=0.7
        )

        # 4. 定义对话消息并调用 LLM (历史问题示例)
        print("\n--- 历史问题示例 ---")
        history_messages = [
            SystemMessage(content="你是一个乐于助人的AI助手，专门解答关于历史的问题。"),
            HumanMessage(content="谁是秦始皇？他做了什么？"),
        ]
        try:
            response_history = llm.invoke(history_messages)
            print("\n--- Gemini (历史助手) 的回答 ---")
            print(response_history.content)
        except Exception as e:
            print(f"调用历史模型时发生错误: {e}")
            print("请检查您的API密钥是否有效，以及所选模型是否在您的区域可用。")


        # 5. 另一个链式调用示例 (诗意向导)
        print("\n--- 诗意向导示例 ---")
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "你是一个富有诗意的旅行向导。"),
            ("user", "{question}"),
        ])

        chain = prompt_template | llm | StrOutputParser()

        travel_question = "给我描述一下北京故宫，用诗意的语言。"
        print(f"\n问题: {travel_question}")
        try:
            response_chain = chain.invoke({"question": travel_question})
            print("\n--- Gemini (诗意向导) 的回答 ---")
            print(response_chain)
        except Exception as e:
            print(f"调用诗意向导时发生错误: {e}")
            print("请检查您的API密钥是否有效，以及所选模型是否在您的区域可用。")

    except ValueError as ve:
        # 捕获 API Key 未找到的错误
        print(f"配置错误: {ve}")
    except Exception as general_e:
        # 捕获其他未预料的通用错误
        print(f"程序运行过程中发生未知错误: {general_e}")

    print("\n--- 项目运行完毕！---")