"""
媒体处理配置使用示例

这个文件展示了如何使用新的配置系统来控制图像描述生成和OCR处理。
"""

from app.config import get_config_manager
from app.rag.processing.media_extractors import ImageVisionExtractor, ImageOCRExtractor
from typing import Dict, Any


def example_image_captioning_with_config():
    """展示如何使用配置控制图像描述生成"""
    
    # 获取配置管理器
    config_manager = get_config_manager()
    
    # 获取图像描述配置
    caption_config = config_manager.get_image_captioning_config()
    print("当前图像描述配置:")
    print(f"  启用状态: {caption_config.get('enabled', True)}")
    print(f"  可用模型: {list(caption_config.get('models', {}).keys())}")
    print(f"  生成参数: {caption_config.get('generation', {})}")
    
    # 创建提取器
    extractor = ImageVisionExtractor()
    
    # 模拟图像数据
    image_bytes = b"fake_image_data"  # 实际使用时应该是真实的图像字节数据
    meta = {
        "media_type": "image",
        "image_format": "jpg",
        "caption_model_type": "chinese_friendly"  # 指定使用中文友好模型
    }
    
    # 提取内容
    result = extractor.extract(image_bytes, meta)
    
    print("\n提取结果:")
    print(f"  描述数量: {len(result.get('captions', []))}")
    print(f"  描述内容: {result.get('captions', [])}")
    print(f"  是否生成嵌入: {result.get('caption_embedding') is not None}")
    print(f"  使用的模型: {result.get('vision_meta', {}).get('caption_model')}")


def example_ocr_with_config():
    """展示如何使用配置控制OCR处理"""
    
    # 获取配置管理器
    config_manager = get_config_manager()
    
    # 获取OCR配置
    ocr_config = config_manager.get_ocr_config()
    print("\n当前OCR配置:")
    print(f"  可用引擎: {ocr_config.get('engines', [])}")
    print(f"  支持语言: {ocr_config.get('languages', [])}")
    print(f"  置信度阈值: {ocr_config.get('confidence_threshold', 0.5)}")
    
    # 创建提取器
    extractor = ImageOCRExtractor()
    
    # 模拟图像数据
    image_bytes = b"fake_image_data"  # 实际使用时应该是真实的图像字节数据
    meta = {
        "media_type": "image",
        "image_format": "png"
    }
    
    # 提取内容
    result = extractor.extract(image_bytes, meta)
    
    print("\nOCR提取结果:")
    print(f"  OCR结果数量: {len(result.get('ocr_results', []))}")
    print(f"  使用的引擎: {result.get('image_meta', {}).get('ocr_engine')}")
    print(f"  是否回退到视觉描述: {'captions' in result}")


def example_different_language_types():
    """展示如何使用不同的语言类型"""
    
    extractor = ImageVisionExtractor()
    image_bytes = b"fake_image_data"
    
    # 测试不同的语言类型
    language_types = ["default", "chinese_friendly", "english_optimized", "multilingual"]
    
    for language_type in language_types:
        meta = {
            "media_type": "image",
            "image_format": "jpg",
            "caption_model_type": language_type
        }
        
        print(f"\n使用 {language_type} 语言类型:")
        result = extractor.extract(image_bytes, meta)
        vision_meta = result.get('vision_meta', {})
        print(f"  模型: {vision_meta.get('caption_model')}")
        print(f"  语言提示词: {vision_meta.get('language_prompt')}")
        print(f"  语言类型: {vision_meta.get('model_type')}")


def example_configuration_modification():
    """展示如何动态修改配置"""
    
    config_manager = get_config_manager()
    
    # 获取当前配置
    caption_config = config_manager.get_image_captioning_config()
    print("修改前的配置:")
    print(f"  描述数量: {caption_config.get('generation', {}).get('num_captions', 1)}")
    print(f"  最大长度: {caption_config.get('generation', {}).get('max_length', 50)}")
    
    # 注意：这里只是展示如何访问配置，实际修改需要重新加载配置文件
    # 或者实现配置的动态更新机制
    
    print("\n要修改配置，请编辑 config/app_config.json 文件中的 media_processing 部分")


if __name__ == "__main__":
    print("=== 媒体处理配置使用示例 ===\n")
    
    try:
        example_image_captioning_with_config()
        example_ocr_with_config()
        example_different_language_types()
        example_configuration_modification()
        
    except Exception as e:
        print(f"运行示例时出错: {e}")
        print("请确保已正确安装相关依赖库")
