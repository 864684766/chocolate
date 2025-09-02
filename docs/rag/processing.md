# 数据处理层（Processing）

将原始样本标准化、清洗、分块、抽取元数据，并进行语言/媒体特定处理。

## 目录结构（多语言+多媒体）

```
app/data_processing/
  pipeline.py                 # 管道编排（责任链/管道模式）
  quality_checker.py          # 质量评估
  lang/
    zh/                       # 中文
      text_cleaner.py
      chunking.py
      metadata_extractor.py
    en/                       # 英文
      text_cleaner.py
      chunking.py
      metadata_extractor.py
    ja/ ...                   # 其他语言按需新增
  media/
    pdf_extractor.py          # PDF 解析（结构/表格/书签）
    image_ocr.py              # 图片 OCR（可选 PaddleOCR/Tesseract）
    audio_asr.py              # 音频转写（可选 Whisper/Vosk）
    video_extractor.py        # 视频帧/字幕抽取
```

说明：

- 语言与媒体解耦在不同目录，通过统一接口注入到管道。
- 语言选择：运行时通过语言检测（fasttext/pycld3）或外部标注，选择 `lang/*` 下对应实现。

## Text Splitters 与多媒体

- 文本：使用 LangChain Text Splitters（Recursive/Token）
- PDF/图片/音视频：先在 `media/*` 转成“可切分文本”（正文/表格序列化/字幕），再交给 Text Splitters。

## 统一接口

- `LanguageProcessor`：clean(text)->str, chunk(text)->List[str], extract_meta(text)->dict
- `MediaExtractor`：extract(path|bytes)->{"text": str, "meta": {...}}

## 中文优化要点

- 分词：jieba + 自定义词典
- 标点与空白归一化、繁简转换（opencc）
- 章节/小节/标题保留为元数据，利于重排与引用
