import chromadb
import traceback # 导入 traceback 模块

if __name__ == "__main__":
    # 尝试连接 ChromaDB 客户端
    chroma_client = None  # 初始化为 None
    try:
        chroma_client = chromadb.HttpClient(host='124.71.135.104', port=8000)
        print("ChromaDB 客户端连接对象创建成功。")
    except Exception as e:
        print(f"创建 ChromaDB 客户端失败: {e}")
        traceback.print_exc()  # 打印完整的异常堆栈
        exit(1)  # 如果客户端都创建失败，就退出

    if chroma_client is None:
        print("ChromaDB 客户端未成功初始化，无法继续。")
        exit(1)

    try:
        # 尝试创建 Collection
        print(f"尝试创建或获取 Collection: my_collection")
        # 尝试 get_or_create_collection 避免重复创建的错误
        collection = chroma_client.get_or_create_collection(name="my_collection")
        print("Collection 创建或获取成功！")

        # 仅在 Collection 为空时添加数据（避免重复添加）
        if collection.count() == 0:
            print("Collection 为空，添加文档...")
            collection.add(
                ids=["id1", "id2"],
                documents=[
                    "This is a document about pineapple",
                    "This is a document about oranges"
                ]
            )
            print("文档添加成功！")
        else:
            print("Collection 中已有数据，跳过添加。")

        results = collection.query(
            query_texts=["This is a document about ocean"],
            n_results=2
        )
        print("查询结果:")
        print(results)

    except Exception as e:
        print(f"操作 ChromaDB 失败: {e}")
        traceback.print_exc() # 打印完整的异常堆栈
        exit(1) # 操作失败，退出