# 参考项目与来源

规划阶段只读检查了用户已有 academic_tools 下的 ARS-Codex、ARS、PaperSpine、ARIS、Claude Scholar，以及 tech_information 的来源接口。部分本地入口仍是规划器或尚未完成的运行流程，因此本版采用原创内核，没有整体复制这些项目。

ARS/ARS-Codex 的本地根许可标记为 CC BY-NC 4.0，PaperSpine/ARIS/Claude Scholar 为 MIT。本版借鉴职责与证据分离的方法，没有复制这些项目的代码、提示词或数据。此记录是来源登记，不是上游全部依赖的许可审计。

## 开源设计参考

| 官方项目 | 本项目参考点 |
|---|---|
| [PaperQA2](https://github.com/Future-House/paper-qa) | 证据检索与引用问答；后续可作为真实主题基线 |
| [OpenScholar](https://github.com/AkariAsai/OpenScholar) | 检索、修订、引用归因与基准方法 |
| [GPT Researcher](https://github.com/assafelovic/gpt-researcher) | 任务拆分及研究报告组织 |
| [LangGraph](https://github.com/langchain-ai/langgraph) | 本版实际依赖，用于持久工作流；业务事实另存 SQLite |
| [Docling](https://github.com/docling-project/docling) | 后续结构化 PDF/OCR 解析候选，本版未集成 |
| [AI Scientist](https://github.com/SakanaAI/AI-Scientist) | 后期实验闭环参考，本版未集成 |

未安装或运行上列研究系统作为本版功能；只有依赖表中列出的库被实际使用。上游论文指标不代表本版指标。

## 官方接口资料

- [Codex 非交互执行](https://learn.chatgpt.com/docs/non-interactive-mode)：结构化输出与独立进程模式；具体选项另用本机 0.153.2 帮助核对并实际执行。
- [Codex SDK](https://learn.chatgpt.com/docs/codex-sdk)：后端升级路径；V1 采用已验证 CLI 适配器。
- [OpenAlex 认证](https://help.openalex.org/api/authentication/)：配额/认证不能假定永久不变。
- [Crossref REST API](https://www.crossref.org/documentation/retrieve-metadata/rest-api/) 与 [使用建议](https://www.crossref.org/documentation/retrieve-metadata/rest-api/tips-for-using-the-crossref-rest-api/)：书目、游标及出版后更新；实测相关性排序采用 query.bibliographic、score、desc。
- [Semantic Scholar API](https://api.semanticscholar.org/api-docs/)：本版 Academic Graph 搜索适配；当前匿名探测出现 HTTP 429。
- [arXiv API 手册](https://github.com/arXiv/arxiv-docs/blob/develop/source/help/api/user-manual.md)：Atom/XML、分页与版本信息。

## 直接运行依赖

Python 及其标准库，Pydantic、httpx、pypdf、LangGraph、langgraph-checkpoint-sqlite。精确版本与完整传递依赖见 uv.lock；各包保留各自分发许可。本项目未将这些依赖源码打包到核心目录。

离线测试、实际接口状态和真实 Codex 调用分别记录在 [验证结果](validation.md)，不把连通性等同于召回率、全文覆盖率或学术正确率。
