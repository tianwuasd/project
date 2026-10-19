# AI Research Skills（Orchestra Research）

> 版本说明（2026-10-19）：源码已更新；正文保留此前整理的指南，功能与安装入口以当前上游 README 为准。见[本次更新记录](../../UPDATE-2026-10-19.md)。

- **上游仓库**：[Orchestra-Research/AI-Research-SKILLs](https://github.com/Orchestra-Research/AI-Research-SKILLs)
- **本地目录**：[`skills/AI-Research-SKILLs`](../../skills/AI-Research-SKILLs/)
- **项目类型**：面向 AI/ML 研究工程的技能库与自动研究编排层
- **仓库 README 当前标注**：98 个 skill、23 个类别
- **文章评价**：第二篇文章评为“顶级”

## 一句话定位

AI Research Skills 不是泛化的“用 AI 做研究”提示词，而是为 AI 研究者准备的工程工具箱：从选题、文献和论文写作，到模型架构、微调、后训练、分布式训练、推理、评测、RAG、Agent 和 MLOps。

## 核心功能

- **研究编排**：`autoresearch` 作为中心层，按两层循环组织文献、想法、实验和写作，并路由到领域 skill。
- **模型与训练**：LitGPT、Mamba、RWKV、NanoGPT、TorchTitan、Axolotl、LLaMA-Factory、PEFT、Unsloth 等。
- **后训练与对齐**：TRL、GRPO、OpenRLHF、SimPO、verl、slime、Constitutional AI、LlamaGuard、NeMo Guardrails 和 Prompt Guard。
- **分布式与性能**：Megatron-Core、DeepSpeed、FSDP2、Accelerate、PyTorch Lightning、Ray Train、Flash Attention、bitsandbytes、GPTQ、AWQ、HQQ 和 GGUF。
- **推理、评测和可观测性**：vLLM、TensorRT-LLM、llama.cpp、SGLang、lm-evaluation-harness、W&B、MLflow、TensorBoard 等。
- **RAG 与 Agent**：Chroma、FAISS、Pinecone、Qdrant、Sentence Transformers、LangChain、LlamaIndex、CrewAI、AutoGPT、DSPy 和相关 Prompt Engineering 工具。
- **AI 论文**：ML Paper Writing、Academic Plotting 和 Agent-Native Research Artifact 等。

## 适用场景

适合深度学习、LLM、Agent、RAG、模型训练和推理方向的研究生与工程研究者。它更像“研究工程知识库”，需要用户已有研究问题、代码和硬件背景；不是通用的社科、湿实验或纯语言润色工具。

## 优势

1. 覆盖 AI 研究工程全生命周期，减少在大量框架之间切换时的上下文损耗。
2. 各 skill 强调官方资料、真实 issue、代码示例和故障排查，而非只有概念介绍。
3. 有中心编排层，也可以按类别单独安装，适合从小范围开始。
4. 对微调、训练策略和硬件约束等工程细节更有针对性。

## 局限与风险

- 98 个 skill 的广度不等于每个项目都经过同等深度验证；应按目标框架和版本检查。
- 训练和推理仍需要 GPU、依赖、数据、显存、许可证和成本；skill 不能替代实验环境。
- 自动研究编排容易把探索性建议误认为已验证结论，关键实验必须由真实日志和基线支撑。
- 深度学习代码会修改环境、下载模型或访问外部服务，执行前应审查脚本和权限。

## 文章评价

文章作者评为“顶级”，并以 GPT-2 微调为例，认为它能根据数据量和硬件给出可执行的训练策略。这个例子体现了其工程导向，但具体建议仍需在目标硬件、框架版本和数据集上复现。

## 推荐组合

- AI Research Skills 负责模型/训练工程，Scientific Agent Skills 负责科学数据处理和数据库。
- ARS-Codex 或 WenyuChiou 版本负责研究问题、证据和项目状态，避免工程任务脱离研究目标。
- PaperSpine 或 Research Paper Writing Skills 负责把实验结果映射成贡献、论文结构和审稿回复。

## 使用前核查

1. 先按类别安装目标 skill，不建议全量加载到每个会话。
2. 固定 Python、CUDA、PyTorch、模型和数据版本，并保存完整训练配置。
3. 检查示例代码的许可证、远程下载、API 以及 GPU 资源消耗。
4. 将 Agent 建议视为实验计划草案，使用真实基线、消融和独立评测确认。

## 参考

- [仓库 README](../../skills/AI-Research-SKILLs/README.md)
- [知乎文章一](https://www.zhihu.com/question/2058648818057531853/answer/2065455008460624405)
- [知乎文章二](https://www.zhihu.com/question/2058648818057531853/answer/2063668103637210722)
