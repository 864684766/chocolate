"""
ChromaDBHelper 使用示例

这个文件展示了如何使用重构后的 ChromaDBHelper 类，包括完整的增删改查操作
"""

from .db_helper import ChromaDBHelper, get_chroma_client
from ....config import get_config_manager


def example_basic_usage():
    """基本使用示例"""
    
    # 方式1：使用便捷函数（简单但资源管理较差）
    try:
        client = get_chroma_client()
        collections = client.list_collections()
        print(f"现有集合: {collections}")
    except Exception as e:
        print(f"获取客户端失败: {e}")
    
    # 方式2：使用 ChromaDBHelper 类（推荐）
    helper = ChromaDBHelper()
    
    try:
        # 自动连接
        client = helper.get_client()
        
        # 获取或创建集合
        collection = helper.get_collection("my_collection")
        
        # 列出所有集合
        collections = helper.list_collections()
        print(f"所有集合: {collections}")
        
        # 检查连接状态
        if helper.is_connected():
            print("数据库连接正常")
        
    except Exception as e:
        print(f"操作失败: {e}")
    finally:
        # 手动断开连接
        helper.disconnect()


def example_context_manager():
    """使用上下文管理器示例（推荐）"""
    
    with ChromaDBHelper() as helper:
        try:
            # 在 with 块中自动管理连接
            collection = helper.get_collection("test_collection")
            
            # 执行数据库操作
            collections = helper.list_collections()
            print(f"集合列表: {collections}")
            
        except Exception as e:
            print(f"操作失败: {e}")


def example_custom_config():
    """使用自定义配置示例"""
    
    # 可以传入自定义的配置管理器
    custom_config_manager = get_config_manager()
    helper = ChromaDBHelper(config_manager=custom_config_manager)
    
    try:
        client = helper.get_client()
        print("使用自定义配置连接成功")
    except Exception as e:
        print(f"连接失败: {e}")
    finally:
        helper.disconnect()


def example_collection_operations():
    """集合操作示例"""
    
    helper = ChromaDBHelper()
    
    try:
        # 创建集合
        collection = helper.get_collection("documents", create_if_not_exists=True)
        
        # 获取现有集合（不创建）
        try:
            existing_collection = helper.get_collection("documents", create_if_not_exists=False)
            print("获取现有集合成功")
        except Exception as e:
            print(f"集合不存在: {e}")
        
        # 列出所有集合
        collections = helper.list_collections()
        print(f"当前集合: {collections}")
        
        # 删除集合
        if "documents" in collections:
            success = helper.delete_collection("documents")
            print(f"删除集合结果: {success}")
            
    except Exception as e:
        print(f"操作失败: {e}")
    finally:
        helper.disconnect()


def example_crud_operations():
    """完整的增删改查操作示例"""
    
    helper = ChromaDBHelper()
    
    try:
        collection_name = "example_documents"
        
        # 1. 添加数据 (Create)
        print("=== 添加数据 ===")
        helper.add(
            collection_name=collection_name,
            documents=[
                "这是第一个文档，关于人工智能",
                "这是第二个文档，关于机器学习",
                "这是第三个文档，关于深度学习"
            ],
            metadatas=[
                {"category": "AI", "author": "张三", "year": 2024},
                {"category": "ML", "author": "李四", "year": 2024},
                {"category": "DL", "author": "王五", "year": 2024}
            ],
            ids=["doc1", "doc2", "doc3"]
        )
        print("数据添加成功")
        
        # 2. 查询数据 (Read)
        print("\n=== 查询数据 ===")
        
        # 文本查询
        query_result = helper.query(
            collection_name=collection_name,
            query_texts=["人工智能"],
            n_results=5
        )
        print(f"文本查询结果: {query_result}")
        
        # 获取所有数据
        all_data = helper.get(
            collection_name=collection_name,
            include=["documents", "metadatas", "embeddings"]
        )
        print(f"所有数据: {all_data}")
        
        # 条件查询
        filtered_data = helper.get(
            collection_name=collection_name,
            where={"category": "AI"}
        )
        print(f"过滤后的数据: {filtered_data}")
        
        # 3. 更新数据 (Update)
        print("\n=== 更新数据 ===")
        helper.update(
            collection_name=collection_name,
            ids=["doc1"],
            metadatas=[{"category": "AI", "author": "张三", "year": 2024, "updated": True}],
            documents=["这是更新后的第一个文档，关于人工智能"]
        )
        print("数据更新成功")
        
        # 4. 删除数据 (Delete)
        print("\n=== 删除数据 ===")
        helper.delete(
            collection_name=collection_name,
            ids=["doc3"]
        )
        print("数据删除成功")
        
        # 5. 其他操作
        print("\n=== 其他操作 ===")
        
        # 获取数据数量
        count = helper.count(collection_name)
        print(f"集合中的数据数量: {count}")
        
        # 预览数据
        preview = helper.peek(collection_name, limit=2)
        print(f"预览数据: {preview}")
        
        # 修改集合属性
        helper.modify(
            collection_name=collection_name,
            metadata={"description": "示例文档集合", "last_updated": "2024-01-01"}
        )
        print("集合属性修改成功")
        
    except Exception as e:
        print(f"操作失败: {e}")
    finally:
        helper.disconnect()


def example_advanced_queries():
    """高级查询示例"""
    
    helper = ChromaDBHelper()
    
    try:
        collection_name = "advanced_documents"
        
        # 创建测试数据
        helper.add(
            collection_name=collection_name,
            documents=[
                "Python是一种高级编程语言",
                "JavaScript是Web开发的主要语言",
                "Java是企业级应用开发的首选",
                "Go语言具有优秀的并发性能"
            ],
            metadatas=[
                {"language": "Python", "type": "scripting", "popularity": "high"},
                {"language": "JavaScript", "type": "scripting", "popularity": "very_high"},
                {"language": "Java", "type": "compiled", "popularity": "high"},
                {"language": "Go", "type": "compiled", "popularity": "medium"}
            ],
            ids=["py", "js", "java", "go"]
        )
        
        # 复杂条件查询
        print("=== 复杂条件查询 ===")
        
        # 元数据过滤
        result1 = helper.query(
            collection_name=collection_name,
            query_texts=["编程语言"],
            where={"type": "scripting", "popularity": "high"},
            n_results=10
        )
        print(f"脚本语言且高人气: {result1}")
        
        # 文档内容过滤
        result2 = helper.query(
            collection_name=collection_name,
            query_texts=["性能"],
            where_document={"$contains": "性能"},
            n_results=5
        )
        print(f"包含'性能'的文档: {result2}")
        
        # 指定返回字段
        result3 = helper.get(
            collection_name=collection_name,
            include=["documents", "metadatas"]
        )
        print(f"指定字段的数据: {result3}")
        
    except Exception as e:
        print(f"高级查询失败: {e}")
    finally:
        helper.disconnect()


if __name__ == "__main__":
    print("=== ChromaDBHelper 使用示例 ===")
    
    print("\n1. 基本使用:")
    example_basic_usage()
    
    print("\n2. 上下文管理器:")
    example_context_manager()
    
    print("\n3. 自定义配置:")
    example_custom_config()
    
    print("\n4. 集合操作:")
    example_collection_operations()
    
    print("\n5. 完整的增删改查操作:")
    example_crud_operations()
    
    print("\n6. 高级查询:")
    example_advanced_queries()

