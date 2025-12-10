# 数据接入层（Ingestion）

负责将多源数据统一转化为规范化的原始样本（raw sample）。

## 接口与模式

- 设计模式：策略 + 工厂 + 观察者
- 基础接口：`DataSource`（start/stop/status/process）
- 工厂：`DataSourceFactory` 通过配置创建不同数据源
- 事件：数据到达/处理进度/错误通过 Observer 推送

## 数据源类型与目录

```
app/data_ingestion/
  sources/
    manual_upload.py      # 手动上传（已实现）
```

说明：为尽快提供可用能力，目前仅落地了 `manual_upload.py`。其余模块暂未创建，待需求到来时再按此约定扩展：

- web_crawler.py（网页爬虫）
- api_integration.py（三方 API 接入）
- file_monitor.py（文件系统监控）
- manager.py（数据源管理器，统一调度与状态）
- plugins/（第三方热插拔数据源）

## 多媒体接入

- PDF：在 ingestion 只做“提取输入流”与基础元信息，其内容结构化交给 processing 层的 `PDFExtractor`。
- 图片：仅获取文件与基本 EXIF；OCR 在 processing 层完成。
- 音频/视频：仅提供媒体流与基础元信息；ASR/字幕抽取在 processing 层完成。

## 配置：支持文件后缀（新增）

在 `config/app_config.json` 中通过 `ingestion.supported_extensions` 配置允许上传/接入的文件后缀（小写）：

```json
{
  "ingestion": {
    "supported_extensions": [
      ".txt",
      ".md",
      ".pdf",
      ".doc",
      ".docx",
      ".xls",
      ".xlsx",
      ".csv",
      ".png",
      ".jpg",
      ".jpeg",
      ".webp",
      ".mp3",
      ".wav",
      ".mp4",
      ".mov",
      ".avi",
      ".mkv"
    ]
  }
}
```

validators 将从该配置读取后缀集合并进行校验，变更后无需改代码。

**文件格式说明**：

- `.doc`：旧版 Word 格式，系统会在 processing 层通过 LibreOffice 自动转换为 `.docx` 后处理
- `.csv`：CSV 文件作为 Excel 类型处理，使用 Excel 提取器和分块策略
- 所有 Office 文档（PDF、Word、Excel）的内容提取和分块都在 processing 层完成

**文件格式说明**：

- `.doc`：旧版 Word 格式，系统会在 processing 层通过 LibreOffice 自动转换为 `.docx` 后处理
- `.csv`：CSV 文件作为 Excel 类型处理，使用 Excel 提取器和分块策略
- 所有 Office 文档（PDF、Word、Excel）的内容提取和分块都在 processing 层完成

## 推荐依赖

- requests/bs4/scrapy/selenium
- watchdog（文件监控）
- tenacity（重试）
