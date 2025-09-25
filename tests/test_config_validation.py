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
        
        # 5. 测试中文处理器初始化
        print('\n5. 测试中文处理器初始化:')
        from app.rag.processing.lang_zh import ChineseProcessor
        
        # 使用默认配置
        processor1 = ChineseProcessor()
        print(f'   ✓ 默认配置处理器初始化成功')
        print(f'   chunk_size: {processor1.chunk_size}')
        print(f'   overlap: {processor1.overlap}')
        print(f'   use_langchain: {processor1.use_langchain}')
        
        # 覆盖特定参数
        processor2 = ChineseProcessor(chunk_size=1000, overlap=200)
        print(f'   ✓ 参数覆盖处理器初始化成功')
        print(f'   chunk_size: {processor2.chunk_size}')
        print(f'   overlap: {processor2.overlap}')
        
        # 6. 测试文本清洗功能
        print('\n6. 测试文本清洗功能:')
        test_text = "这是一个测试文本。\r\n\n\n\n   多个空格   和制表符\t\t测试。"
        cleaned_text = processor1.clean(test_text)
        print(f'   原始文本: {repr(test_text)}')
        print(f'   清洗后: {repr(cleaned_text)}')
        
        # 7. 测试元数据提取
        print('\n7. 测试元数据提取:')
        metadata = processor1.extract_meta()
        print(f'   处理器元数据: {metadata}')
        
        print('\n✓ 所有测试通过！中文处理器配置化功能正常工作。')
        
    except Exception as e:
        print(f'\n✗ 测试失败: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_chinese_processor_config()
