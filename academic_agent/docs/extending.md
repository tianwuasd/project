# 扩展方式

先保持契约，再增加功能。新模块需有离线契约和失败测试、单独记录的真实服务探测。

## 来源

继承 `sources.Source`，实现 `_params`、`_unpack`、`_parse` 并注册到 `SOURCES`。保留原字段及源 ID，缺失信息不猜造；使用 `HttpTransport` 保持大小、重试和时间限制。V1 未内置知网、Web of Science、Scopus、Google Scholar 自动抓取，可导入授权导出的 RIS/BibTeX。

## 解析器

实现 `Parser.parse(Path, work_id) -> Document`，接入 attach 分发器。提供内容哈希、物理页码和确切字符偏移。OCR 应保留引擎、页图与识别限制，不能只把状态改为全文。

## API 后端

实现 `Backend.name` 和 `Backend.run(ModelJob) -> ModelResult`，通过 `Runner(root, backend=YourBackend())` 注入，CLI 注册另行添加。任务带指令、数据、Schema、输入和时间上限；返回实际 usage，未知费用保持未知。不能在一个 run 内隐藏多个模型调用；每次尝试须经计量层和完整领域验证。

现有 ExchangeBackend 可以接手动外部结果，请求和响应凭完整 fingerprint 配对：

```json
{"fingerprint": "从请求原样复制", "data": {"claims": [], "missing": ["method"]}}
```

data 的实际格式遵循请求中的 Schema。引入有鉴权的服务时独立配置凭据，不能进入日志、缓存键或导出。

## 领域特化与下一阶段

领域模块提供检索词、纳入标准、证据字段和写作结构，例如经济计量识别假设或材料制备参数。保留通用文献、文档、证据和陈述契约，不把某一学科提示词写进存储。

先用真实题目建立冻结样本和人工标注，再增加引文扩展、样本文献趋势、带反证检索的创新候选。统计结论需新增可复算 ComputationRecord；实验和完整论文写作需独立真实性门槛。
