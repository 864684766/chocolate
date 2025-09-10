"""
图像分块策略
"""

from typing import List, Dict, Any
from .base import MediaChunkingStrategy


class ImageChunkingStrategy(MediaChunkingStrategy):
    """图片分块策略 - 基于 OCR 结果"""
    
    def __init__(self, chunk_size: int = 800, overlap: int = 150):
        """
        初始化图像分块策略
        
        Args:
            chunk_size: 分块大小
            overlap: 重叠大小
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk(self, content: Dict[str, Any], meta: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        图片内容分块 - 支持OCR结果和视觉理解描述
        
        Args:
            content: 图像内容，应包含 OCR 识别的文本和位置信息，或视觉理解生成的描述
            meta: 元数据
            
        Returns:
            List[Dict[str, Any]]: 分块结果列表
        """
        # content 应该包含 OCR 识别的文本和位置信息，或视觉理解生成的描述
        ocr_results = content.get("ocr_results", [])
        captions = content.get("captions", [])
        image_meta = content.get("image_meta", {})
        
        # 如果OCR结果为空，但有视觉理解描述，则处理描述
        if not ocr_results and captions:
            return self._chunk_captions(captions, meta, image_meta)
        
        # 如果OCR结果也为空，则返回空列表
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
    
    def _chunk_captions(self, captions: List[str], meta: Dict[str, Any], image_meta: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        处理视觉理解生成的图像描述
        
        Args:
            captions: 图像描述列表
            meta: 元数据
            image_meta: 图像元数据
            
        Returns:
            List[Dict[str, Any]]: 分块结果列表
        """
        if not captions:
            return []
        
        # 将所有描述合并为一个文本
        combined_text = " ".join(captions)
        
        chunks = []
        chunk_index = 0
        
        if len(combined_text) <= self.chunk_size:
            # 描述文本不长，直接作为一个块
            chunk_meta = meta.copy()
            chunk_meta.update({
                "chunk_index": chunk_index,
                "chunk_type": "image_caption",
                "chunk_size": len(combined_text),
                "image_meta": image_meta,
                "caption_count": len(captions)
            })
            
            chunks.append({
                "text": combined_text,
                "meta": chunk_meta
            })
        else:
            # 描述文本过长，需要分块
            start = 0
            while start < len(combined_text):
                end = min(start + self.chunk_size, len(combined_text))
                chunk_text = combined_text[start:end]
                
                chunk_meta = meta.copy()
                chunk_meta.update({
                    "chunk_index": chunk_index,
                    "chunk_type": "image_caption_part",
                    "chunk_size": len(chunk_text),
                    "start_pos": start,
                    "end_pos": end,
                    "image_meta": image_meta,
                    "caption_count": len(captions)
                })
                
                chunks.append({
                    "text": chunk_text,
                    "meta": chunk_meta
                })
                chunk_index += 1
                
                start = end - self.overlap if end - self.overlap > start else end
        
        return chunks
    
    @staticmethod
    def _group_by_regions(ocr_results: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        将 OCR 结果按空间位置分组
        
        Args:
            ocr_results: OCR识别结果列表
            
        Returns:
            List[List[Dict[str, Any]]]: 按区域分组的OCR结果
        """
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
        """
        获取区域的边界信息
        
        Args:
            region: 区域内的OCR结果列表
            
        Returns:
            Dict[str, Any]: 边界信息字典
        """
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
        """
        分割区域文本
        
        Args:
            text: 区域文本
            region_idx: 区域索引
            image_meta: 图像元数据
            
        Returns:
            List[Dict[str, Any]]: 分割后的文本块列表
        """
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
