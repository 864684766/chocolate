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
  base/                   # 抽象基类与接口
  sources/
    manual_upload.py      # 手动上传
    web_crawler.py        # 网页爬虫（可配合 selenium）
    api_integration.py    # 三方 API
    file_monitor.py       # 文件系统监控（watchdog）
  plugins/                # 第三方热插拔数据源
  manager.py              # 统一管理器
```

## 多媒体接入

- PDF：在 ingestion 只做“提取输入流”与基础元信息，其内容结构化交给 processing 层的 `PDFExtractor`。
- 图片：仅获取文件与基本 EXIF；OCR 在 processing 层完成。
- 音频/视频：仅提供媒体流与基础元信息；ASR/字幕抽取在 processing 层完成。

## 推荐依赖

- requests/bs4/scrapy/selenium
- watchdog（文件监控）
- tenacity（重试）
