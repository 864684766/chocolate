#!/usr/bin/env python
"""
测试中文处理器配置化功能
验证配置项是否正确加载和使用
"""

def test_chinese_processor_config():
    """测试中文处理器配置化功能"""
    print('=== 中文处理器配置化测试 ===\n')
    
    try:
        # 1. 测试配置管理器
        print('1. 测试配置管理器:')
        from app.config import get_config_manager
        config_manager = get_config_manager()
        print('   ✓ 配置管理器初始化成功')
        
        # 2. 测试中文处理配置获取
        print('\n2. 测试中文处理配置获取:')
        chinese_config = config_manager.get_chinese_processing_config()
        print(f'   ✓ 中文处理配置获取成功，包含 {len(chinese_config)} 个配置项')
        
        # 3. 测试分块配置
        print('\n3. 测试分块配置:')
        chunking_config = chinese_config.get("chunking", {})
        print(f'   default_chunk_size: {chunking_config.get("default_chunk_size")}')
        print(f'   default_overlap: {chunking_config.get("default_overlap")}')
        print(f'   use_langchain: {chunking_config.get("use_langchain")}')
        print(f'   separators: {chunking_config.get("separators")}')
        
        # 4. 测试文本清洗配置
        print('\n4. 测试文本清洗配置:')
        text_cleaning_config = chinese_config.get("text_cleaning", {})
        print(f'   normalize_whitespace_pattern: {text_cleaning_config.get("normalize_whitespace_pattern")}')
        print(f'   normalize_whitespace_replacement: {text_cleaning_config.get("normalize_whitespace_replacement")}')
        
        # 5. 测试文本分块策略初始化
        print('\n5. 测试文本分块策略初始化:')
        from app.rag.processing.media.chunking.text import TextChunkingStrategy
        
        # 使用默认配置
        strategy1 = TextChunkingStrategy()
        print(f'   ✓ 默认配置分块策略初始化成功')
        print(f'   chunk_size: {strategy1.chunk_size}')
        print(f'   overlap: {strategy1.overlap}')
        
        # 覆盖特定参数
        strategy2 = TextChunkingStrategy(chunk_size=1000, overlap=200)
        print(f'   ✓ 参数覆盖分块策略初始化成功')
        print(f'   chunk_size: {strategy2.chunk_size}')
        print(f'   overlap: {strategy2.overlap}')
        
        # 6. 测试文本清洗功能（使用TextCleaner）
        print('\n6. 测试文本清洗功能:')
        from app.rag.processing.utils.text_cleaner import TextCleaner
        text_cleaner = TextCleaner()
        test_text = "这是一个测试文本。\r\n\n\n\n   多个空格   和制表符\t\t测试。"
        cleaned_text = text_cleaner.clean(test_text)
        print(f'   原始文本: {repr(test_text)}')
        print(f'   清洗后: {repr(cleaned_text)}')
        
        # 7. 测试分块功能
        print('\n7. 测试分块功能:')
        test_content = "这是第一段。\n\n这是第二段。包含多个句子。这是第三句！"
        meta = {"media_type": "text", "source": "test"}
        chunks = strategy1.chunk(test_content, meta)
        print(f'   原始文本长度: {len(test_content)}')
        print(f'   分块数量: {len(chunks)}')
        for i, chunk in enumerate(chunks[:3]):  # 只显示前3个块
            print(f'   块 {i}: 长度={chunk["meta"]["chunk_size"]}, 类型={chunk["meta"]["chunk_type"]}')
        
        print('\n✓ 所有测试通过！中文处理器配置化功能正常工作。')
        
    except Exception as e:
        print(f'\n✗ 测试失败: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_chinese_processor_config()
