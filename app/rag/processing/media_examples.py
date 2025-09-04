"""
多媒体分块策略使用示例

这个文件展示了如何使用不同的媒体分块策略来处理各种类型的文件。
"""

import os
from typing import Dict
from .media_chunking import (
    ChunkingStrategyFactory,
)


def process_image_example(image_path: str) -> None:
    """图片处理示例"""
    print(f"=== 处理图片: {image_path} ===")
    
    try:
        # 读取图片文件
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        
        # 创建图片分块策略
        strategy = ChunkingStrategyFactory.create_strategy(
            "image",
            chunk_size=800,
            overlap=150
        )
        
        # 设置元数据
        meta = {
            "filename": os.path.basename(image_path),
            "file_size": len(image_bytes),
            "image_format": image_path.split('.')[-1].lower(),
            "media_type": "image"
        }
        
        # 执行分块
        chunks = strategy.chunk(image_bytes, meta)
        
        print(f"生成了 {len(chunks)} 个文本块:")
        for i, chunk in enumerate(chunks):
            print(f"  块 {i+1}: {chunk['text'][:100]}...")
            print(f"    元数据: {chunk['meta']}")
            print()
            
    except Exception as e:
        print(f"处理图片时出错: {e}")


def process_video_example(video_path: str) -> None:
    """视频处理示例"""
    print(f"=== 处理视频: {video_path} ===")
    
    try:
        # 读取视频文件
        with open(video_path, 'rb') as f:
            video_bytes = f.read()
        
        # 创建视频分块策略
        strategy = ChunkingStrategyFactory.create_strategy(
            "video",
            chunk_size=800,
            overlap=150
        )
        
        # 设置元数据
        meta = {
            "filename": os.path.basename(video_path),
            "file_size": len(video_bytes),
            "video_format": video_path.split('.')[-1].lower(),
            "media_type": "video"
        }
        
        # 执行分块
        chunks = strategy.chunk(video_bytes, meta)
        
        print(f"生成了 {len(chunks)} 个文本块:")
        for i, chunk in enumerate(chunks):
            print(f"  块 {i+1}: {chunk['text'][:100]}...")
            print(f"    元数据: {chunk['meta']}")
            print()
            
    except Exception as e:
        print(f"处理视频时出错: {e}")


def process_text_example(text_content: str) -> None:
    """文本处理示例"""
    print("=== 处理文本内容 ===")
    
    try:
        # 创建文本分块策略
        strategy = ChunkingStrategyFactory.create_strategy(
            "text",
            chunk_size=800,
            overlap=150
        )
        
        # 设置元数据
        meta = {
            "content_type": "text",
            "media_type": "text",
            "language": "zh"
        }
        
        # 执行分块
        chunks = strategy.chunk(text_content, meta)
        
        print(f"生成了 {len(chunks)} 个文本块:")
        for i, chunk in enumerate(chunks):
            print(f"  块 {i+1}: {chunk['text'][:100]}...")
            print(f"    元数据: {chunk['meta']}")
            print()
            
    except Exception as e:
        print(f"处理文本时出错: {e}")


def check_dependencies() -> Dict[str, bool]:
    """检查各种依赖库是否可用"""
    print("=== 检查依赖库 ===")
    
    dependencies = {}
    
    # 检查 OCR 库
    try:
        import easyocr
        dependencies["easyocr"] = True
        print("✓ EasyOCR 可用")
    except ImportError:
        dependencies["easyocr"] = False
        print("✗ EasyOCR 不可用")
    
    try:
        import paddleocr
        dependencies["paddleocr"] = True
        print("✓ PaddleOCR 可用")
    except ImportError:
        dependencies["paddleocr"] = False
        print("✗ PaddleOCR 不可用")
    
    try:
        import pytesseract
        dependencies["pytesseract"] = True
        print("✓ Tesseract 可用")
    except ImportError:
        dependencies["pytesseract"] = False
        print("✗ Tesseract 不可用")
    
    # 检查视频处理库
    try:
        import cv2
        dependencies["opencv"] = True
        print("✓ OpenCV 可用")
    except ImportError:
        dependencies["opencv"] = False
        print("✗ OpenCV 不可用")
    
    # 检查语音识别库
    try:
        import whisper
        dependencies["whisper"] = True
        print("✓ Whisper 可用")
    except ImportError:
        dependencies["whisper"] = False
        print("✗ Whisper 不可用")
    
    try:
        import speech_recognition
        dependencies["speech_recognition"] = True
        print("✓ SpeechRecognition 可用")
    except ImportError:
        dependencies["speech_recognition"] = False
        print("✗ SpeechRecognition 不可用")
    
    # 检查 LangChain
    try:
        import langchain
        dependencies["langchain"] = True
        print("✓ LangChain 可用")
    except ImportError:
        dependencies["langchain"] = False
        print("✗ LangChain 不可用")
    
    print()
    return dependencies


def main():
    """主函数 - 运行所有示例"""
    print("多媒体分块策略示例")
    print("=" * 50)
    
    # 检查依赖
    deps = check_dependencies()
    
    # 文本处理示例
    sample_text = """
    这是一个示例文本，用于演示文本分块功能。
    
    文本分块是RAG系统中的重要组成部分，它能够将长文本分割成适合向量化的片段。
    好的分块策略应该保持语义的完整性，避免在句子中间切分。
    
    中文文本的分块需要特别注意标点符号和段落结构。
    例如，句号、问号、感叹号等都是很好的分块边界。
    
    此外，还需要考虑分块的大小和重叠度，以平衡检索精度和计算效率。
    """
    
    process_text_example(sample_text)
    
    # 图片处理示例（如果有图片文件）
    image_path = "sample_image.jpg"
    if os.path.exists(image_path):
        process_image_example(image_path)
    else:
        print(f"图片文件 {image_path} 不存在，跳过图片处理示例")
    
    # 视频处理示例（如果有视频文件）
    video_path = "sample_video.mp4"
    if os.path.exists(video_path):
        process_video_example(video_path)
    else:
        print(f"视频文件 {video_path} 不存在，跳过视频处理示例")
    
    print("\n=== 安装建议 ===")
    if not deps["easyocr"] and not deps["paddleocr"] and not deps["pytesseract"]:
        print("建议安装 OCR 库:")
        print("  pip install easyocr  # 推荐，支持中文")
        print("  pip install paddleocr  # 百度开源，中文效果好")
        print("  pip install pytesseract  # 需要系统安装 Tesseract")
    
    if not deps["opencv"]:
        print("建议安装 OpenCV:")
        print("  pip install opencv-python")
    
    if not deps["whisper"] and not deps["speech_recognition"]:
        print("建议安装语音识别库:")
        print("  pip install openai-whisper  # 推荐")
        print("  pip install SpeechRecognition")
    
    if not deps["langchain"]:
        print("建议安装 LangChain:")
        print("  pip install langchain langchain-community")


if __name__ == "__main__":
    main()
