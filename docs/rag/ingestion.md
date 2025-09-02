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

## 推荐依赖

- requests/bs4/scrapy/selenium
- watchdog（文件监控）
- tenacity（重试）
