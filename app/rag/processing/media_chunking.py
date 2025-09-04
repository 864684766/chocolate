from __future__ import annotations

from typing import List, Dict, Any
from abc import ABC, abstractmethod


class MediaChunkingStrategy(ABC):
    """媒体分块策略基类"""
    
    @abstractmethod
    def chunk(self, content: Any, meta: Dict[str, Any]) -> List[Dict[str, Any]]:
        """将媒体内容分块，返回包含文本和元数据的块列表"""
        pass


class TextChunkingStrategy(MediaChunkingStrategy):
    """文本分块策略 - 使用 LangChain 智能分块"""
    
    def __init__(self, chunk_size: int = 800, overlap: int = 150):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk(self, content: str, meta: Dict[str, Any]) -> List[Dict[str, Any]]:
        """智能文本分块"""
        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            
            # 中文友好的分隔符
            separators = [
                "\n\n",      # 段落分隔
                "\n",        # 行分隔
                "。",        # 中文句号
                "！",        # 中文感叹号
                "？",        # 中文问号
                "；",        # 中文分号
                "，",        # 中文逗号
                " ",         # 空格
                ""           # 字符级别
            ]
            
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.overlap,
                separators=separators,
                length_function=len
            )
            
            chunks = splitter.split_text(content)
            
            # 为每个块添加元数据
            result = []
            for i, chunk in enumerate(chunks):
                chunk_meta = meta.copy()
                chunk_meta.update({
                    "chunk_index": i,
                    "chunk_type": "text",
                    "chunk_size": len(chunk),
                    "total_chunks": len(chunks)
                })
                result.append({
                    "text": chunk,
                    "meta": chunk_meta
                })
            
            return result
            
        except ImportError:
            # 回退到简单分块
            return self._simple_chunk(content, meta)
    
    def _simple_chunk(self, content: str, meta: Dict[str, Any]) -> List[Dict[str, Any]]:
        """简单分块（回退方案）"""
        chunks = []
        start = 0
        while start < len(content):
            end = min(start + self.chunk_size, len(content))
            chunk_text = content[start:end]
            
            chunk_meta = meta.copy()
            chunk_meta.update({
                "chunk_index": len(chunks),
                "chunk_type": "text_simple",
                "chunk_size": len(chunk_text),
                "start_pos": start,
                "end_pos": end
            })
            
            chunks.append({
                "text": chunk_text,
                "meta": chunk_meta
            })
            
            start = end - self.overlap if end - self.overlap > start else end
        
        return chunks


class PDFChunkingStrategy(MediaChunkingStrategy):
    """PDF 分块策略 - 保留文档结构"""
    
    def __init__(self, chunk_size: int = 800, overlap: int = 150):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk(self, content: Dict[str, Any], meta: Dict[str, Any]) -> List[Dict[str, Any]]:
        """PDF 内容分块，保留章节结构"""
        # content 应该包含 {"pages": [...], "toc": [...]} 等结构
        pages = content.get("pages", [])
        
        chunks = []
        chunk_index = 0
        
        for page_num, page in enumerate(pages):
            page_text = page.get("text", "")
            page_meta = page.get("meta", {})
            
            if len(page_text) <= self.chunk_size:
                # 页面内容不长，直接作为一个块
                chunk_meta = meta.copy()
                chunk_meta.update({
                    "chunk_index": chunk_index,
                    "chunk_type": "pdf_page",
                    "page_number": page_num + 1,
                    "chunk_size": len(page_text),
                    **page_meta
                })
                
                chunks.append({
                    "text": page_text,
                    "meta": chunk_meta
                })
                chunk_index += 1
            else:
                # 页面内容过长，需要分块
                page_chunks = self._split_page_content(page_text, page_num, page_meta)
                for page_chunk in page_chunks:
                    chunk_meta = meta.copy()
                    chunk_meta.update({
                        "chunk_index": chunk_index,
                        "chunk_type": "pdf_page_part",
                        **page_chunk["meta"]
                    })
                    
                    chunks.append({
                        "text": page_chunk["text"],
                        "meta": chunk_meta
                    })
                    chunk_index += 1
        
        return chunks
    
    def _split_page_content(self, text: str, page_num: int, page_meta: Dict[str, Any]) -> List[Dict[str, Any]]:
        """分割页面内容"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end]
            
            chunks.append({
                "text": chunk_text,
                "meta": {
                    "page_number": page_num + 1,
                    "chunk_size": len(chunk_text),
                    "start_pos": start,
                    "end_pos": end,
                    **page_meta
                }
            })
            
            start = end - self.overlap if end - self.overlap > start else end
        
        return chunks


class ImageChunkingStrategy(MediaChunkingStrategy):
    """图片分块策略 - 基于 OCR 结果"""
    
    def __init__(self, chunk_size: int = 800, overlap: int = 150):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk(self, content: Dict[str, Any], meta: Dict[str, Any]) -> List[Dict[str, Any]]:
        """图片 OCR 结果分块"""
        # content 应该包含 OCR 识别的文本和位置信息
        ocr_results = content.get("ocr_results", [])
        image_meta = content.get("image_meta", {})
        
        if not ocr_results:
            return []
        
        # 按区域分组文本
        regions = self._group_by_regions(ocr_results)
        
        chunks = []
        chunk_index = 0
        
        for region_idx, region in enumerate(regions):
            region_text = " ".join([item["text"] for item in region])
            
            if len(region_text) <= self.chunk_size:
                # 区域文本不长，直接作为一个块
                chunk_meta = meta.copy()
                chunk_meta.update({
                    "chunk_index": chunk_index,
                    "chunk_type": "image_region",
                    "region_index": region_idx,
                    "chunk_size": len(region_text),
                    "image_meta": image_meta,
                    "region_bounds": self._get_region_bounds(region)
                })
                
                chunks.append({
                    "text": region_text,
                    "meta": chunk_meta
                })
                chunk_index += 1
            else:
                # 区域文本过长，需要分块
                region_chunks = self._split_region_text(region_text, region_idx, image_meta)
                for region_chunk in region_chunks:
                    chunk_meta = meta.copy()
                    chunk_meta.update({
                        "chunk_index": chunk_index,
                        "chunk_type": "image_region_part",
                        **region_chunk["meta"]
                    })
                    
                    chunks.append({
                        "text": region_chunk["text"],
                        "meta": chunk_meta
                    })
                    chunk_index += 1
        
        return chunks
    
    @staticmethod
    def _group_by_regions(ocr_results: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """将 OCR 结果按空间位置分组"""
        # 简单的基于 y 坐标的分组（可以改进为更复杂的空间聚类）
        sorted_results = sorted(ocr_results, key=lambda x: (x.get("y", 0), x.get("x", 0)))
        
        regions = []
        current_region = []
        last_y = None
        
        for result in sorted_results:
            y = result.get("y", 0)
            
            if last_y is None or abs(y - last_y) < 20:  # 20px 阈值
                current_region.append(result)
            else:
                if current_region:
                    regions.append(current_region)
                current_region = [result]
            
            last_y = y
        
        if current_region:
            regions.append(current_region)
        
        return regions
    
    @staticmethod
    def _get_region_bounds(region: List[Dict[str, Any]]) -> Dict[str, Any]:
        """获取区域的边界信息"""
        if not region:
            return {}
        
        x_coordinates = [item.get("x", 0) for item in region]
        y_coordinates = [item.get("y", 0) for item in region]
        
        return {
            "min_x": min(x_coordinates),
            "max_x": max(x_coordinates),
            "min_y": min(y_coordinates),
            "max_y": max(y_coordinates)
        }
    
    def _split_region_text(self, text: str, region_idx: int, image_meta: Dict[str, Any]) -> List[Dict[str, Any]]:
        """分割区域文本"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end]
            
            chunks.append({
                "text": chunk_text,
                "meta": {
                    "region_index": region_idx,
                    "chunk_size": len(chunk_text),
                    "start_pos": start,
                    "end_pos": end,
                    "image_meta": image_meta
                }
            })
            
            start = end - self.overlap if end - self.overlap > start else end
        
        return chunks


class VideoChunkingStrategy(MediaChunkingStrategy):
    """视频分块策略 - 基于字幕和时间戳"""
    
    def __init__(self, chunk_size: int = 800, overlap: int = 150):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk(self, content: Dict[str, Any], meta: Dict[str, Any]) -> List[Dict[str, Any]]:
        """视频内容分块，基于字幕和时间信息"""
        # content 应该包含字幕和可能的语音转写结果
        subtitles = content.get("subtitles", [])
        transcript = content.get("transcript", "")
        video_meta = content.get("video_meta", {})
        
        chunks = []
        chunk_index = 0
        
        if subtitles:
            # 优先使用字幕信息
            chunks.extend(self._chunk_by_subtitles(subtitles, video_meta, chunk_index))
        elif transcript:
            # 回退到语音转写结果
            chunks.extend(self._chunk_by_transcript(transcript, video_meta, chunk_index))
        
        return chunks
    
    def _chunk_by_subtitles(self, subtitles: List[Dict[str, Any]], 
                           video_meta: Dict[str, Any], start_index: int) -> List[Dict[str, Any]]:
        """基于字幕分块"""
        chunks = []
        current_chunk = ""
        current_start_time = None
        current_end_time = None
        chunk_index = start_index
        
        for subtitle in subtitles:
            text = subtitle.get("text", "")
            start_time = subtitle.get("start_time", 0)
            end_time = subtitle.get("end_time", 0)
            
            if len(current_chunk) + len(text) <= self.chunk_size:
                # 可以添加到当前块
                if not current_chunk:
                    current_start_time = start_time
                current_chunk += " " + text if current_chunk else text
                current_end_time = end_time
            else:
                # 当前块已满，保存并开始新块
                if current_chunk:
                    chunk_meta = {
                        "chunk_index": chunk_index,
                        "chunk_type": "video_subtitle",
                        "chunk_size": len(current_chunk),
                        "start_time": current_start_time,
                        "end_time": current_end_time,
                        "video_meta": video_meta
                    }
                    
                    chunks.append({
                        "text": current_chunk.strip(),
                        "meta": chunk_meta
                    })
                    chunk_index += 1
                
                # 开始新块
                current_chunk = text
                current_start_time = start_time
                current_end_time = end_time
        
        # 处理最后一个块
        if current_chunk:
            chunk_meta = {
                "chunk_index": chunk_index,
                "chunk_type": "video_subtitle",
                "chunk_size": len(current_chunk),
                "start_time": current_start_time,
                "end_time": current_end_time,
                "video_meta": video_meta
            }
            
            chunks.append({
                "text": current_chunk.strip(),
                "meta": chunk_meta
            })
        
        return chunks
    
    def _chunk_by_transcript(self, transcript: str, video_meta: Dict[str, Any], 
                            start_index: int) -> List[Dict[str, Any]]:
        """基于语音转写结果分块"""
        # 使用文本分块策略
        text_strategy = TextChunkingStrategy(self.chunk_size, self.overlap)
        text_chunks = text_strategy.chunk(transcript, {})
        
        # 转换元数据格式
        for i, chunk in enumerate(text_chunks):
            chunk["meta"].update({
                "chunk_index": start_index + i,
                "chunk_type": "video_transcript",
                "video_meta": video_meta
            })
        
        return text_chunks


class ChunkingStrategyFactory:
    """分块策略工厂"""
    
    @staticmethod
    def create_strategy(media_type: str, **kwargs) -> MediaChunkingStrategy:
        """根据媒体类型创建对应的分块策略"""
        strategies = {
            "text": TextChunkingStrategy,
            "pdf": PDFChunkingStrategy,
            "image": ImageChunkingStrategy,
            "video": VideoChunkingStrategy,
            "audio": VideoChunkingStrategy,  # 音频使用视频策略（处理字幕/转写）
        }
        
        strategy_class = strategies.get(media_type.lower(), TextChunkingStrategy)
        return strategy_class(**kwargs)
