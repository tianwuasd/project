# dataset1282_analysis

本分支从已完成的 0.3.0 HU 缓存运行固定划分的 Xu 解剖分割。`scripts/xu_pipeline.py run` 串联双尺度定位、严格投票后扩框 20%、四路基础分割、可用的起始段和血池分支、血管后处理及原网格评价。每一阶段单独保存日志、结果和恢复依据。[全模型运行与恢复](docs/xu-model-run.md) · [各分支与取框验收条件](docs/xu-segmentation.md)

可在 Windows 本机或 Linux 服务器调用的 NIfTI + Excel 只读数据审计工程。首版提供独立环境部署、代码更新、数据统计和本地报告；适合先弄清数据来源与质量，再决定后续处理。

审计入口只读，按文件对数与不同 PatientID 数分别统计。独立 `prepare` 入口按方案 A 建立患者分组，在私有工作区生成 nnU-Net 预处理副本。0.3.0 新增 HU、Excel 共同毫米网格、部分标签边际损失接口、V2 资源调度及可靠短测。[当前工作流](docs/hu-marginal-workflow.md)；[旧版试跑与审核格式](docs/preprocessing.md)。原始影像、标签、Excel 和历史记录保留。

## 开始使用

需要 Python 3.11（验证基线）和 Git。将 `project.local.json.example` 复制为 `project.local.json`，填写数据根、Excel、私有工作区与独立环境路径；使用绝对路径可避免不同终端目录带来的歧义。真实配置不要提交。

```text
python setup_env.py --plan
python setup_env.py --install
python setup_env.py --check
python smoke.py --profile synthetic
python run.py inventory
python smoke.py --profile capacity
python run.py analyze --stage full --run initial-audit
python run.py report --run initial-audit
```

需要代理时只为当前安装/更新进程传入，例如 `python setup_env.py --install --proxy http://127.0.0.1:7897`。Linux 常见命令为 `python3`，也可使用 `sh start.sh inventory`；Windows 使用 `start.cmd inventory`。环境未安装前，使用基础 Python 调用 `setup_env.py`。安装后 `run.py` / `smoke.py` 会优先转到配置中的专属解释器。

也可直接使用该环境的 Python 调用每个入口。环境和更新模块仅依赖标准库；不要求安装 PyTorch、CUDA、模型或 Slicer。

## 入口与输出

| 入口 | 作用 |
|---|---|
| `setup_env.py --plan / --check / --install` | 展示计划、只检查、创建并安装专属环境；检查实际 API 与锁定版本 |
| `python3 scripts/setup_gpu.py --check / --install --runtime PATH` | 先检查复用已有GPU环境；缺少时只在独立 `training` 目录安装，见[环境说明](docs/environment.md) |
| `run.py inventory` | 读取主清单，检查规范路径和文件存在情况 |
| `run.py analyze --stage metadata / headers / full` | 元数据、文件头或全体素审计，三个范围明确区分 |
| `run.py status --run NAME` | 只读取状态，不安装或启动计算 |
| `run.py resume --run NAME` | 从已完整提交且内容摘要匹配的病例恢复 |
| `run.py report --run NAME` | 生成 CSV、JSON、PNG 和 HTML/Markdown 报告 |
| `run.py prepare --run NAME` | 锁定患者候选分组；可选在独立研究环境进行源域预处理 |
| `run.py prepare --run NAME --excel-pilot --research-python PYTHON` | 用来源训练病例的逐例 Excel 间距试跑三维自动规划，输出标签回写检查 |
| `run.py prepare --run NAME --workflow POLICY --frozen-cohort COHORT --research-python PYTHON` | 按确认规则执行 HU 与部分标签预处理，支持 `--resume` |
| `python scripts/short_test.py --work-root WORK --research-python PYTHON` | 预处理、数值一致性、边际损失更新和故障恢复验收 |
| `smoke.py --profile synthetic / quick / capacity` | 正式链路合成短测、少量真实病例、真实尺寸容量测试 |
| `update.py --check` | 检查官方仓库固定子目录最新提交 |
| `update.py` / `update.py --rollback ID` | 下载校验并激活代码 release，或回滚保留版本 |

结果位于配置的 `work_root/runs/NAME/`，报告入口是 `report/report.html`。`status.json` 为最新快照，`events.jsonl` 为追加日志，`cases/` 为逐例结果及完整标记。每次运行保存配置、输入摘要与冻结代码；失败或中断时原始数据仍保留。

完整审计的“完成”仅表示计算流程完成；空间校准、标签语义、元数据关联等仍可处于待核查状态。合成短测故意含一个截断文件，预期为 2 个完成、1 个失败并验证恢复；它不是正式数据失败。

## 工程结构

```text
run.py / setup_env.py / update.py / smoke.py    独立薄入口
bootstrap.py                                 先选校验过的代码，再导入
configs/default.json                         当前版本默认参数
src/dataset1282_analysis/
  config.py / contracts.py                    配置与身份合同
  environment.py / maintenance.py             环境、更新、回滚
  inputs.py / metadata.py                     清单、Excel、严格关联
  geometry.py / voxels.py / labels.py          文件头、全体素与标签统计
  resources.py / vendor/                      v2 CPU资源机制适配
  engine/                                    执行、完整病例提交与恢复
  preprocessing/                             私有策略、隔离任务、CPU/RAM/IO队列
  training/                                  边际损失及逐病例监督Trainer接口
  observability/                             锁、状态与日志
  reporting.py                               本地报告和数组切面
tests/                                       仅合成行为测试
docs/                                        方法、运行、更新与验证说明
```

Xu 分割入口支持按作业获准的 GPU 池并行分支、拆分固定批次、按病例分发预测，并保留缺标边际监督和原网格评价。单设备也可运行。现场资格检查、逐卡实际工作量及恢复命令见[完整模型运行说明](docs/xu-model-run.md)。

独立预测入口为 `scripts/xu_predict.py`，沿用三组冻结留出名单，以训练计划准备测试 CT，使用已完成模型预测，再分别评价。模型冻结、单卡隔离、恢复和输出说明见[留出组预测](docs/xu-prediction.md)。

GPU 入口默认排除物理 0、1 号卡。并发训练先统一登记进程再放行；确认外部占用变化后，释放本组并重新准入，从最近完整 epoch 恢复。内存不足时缩小供数队列、减少 worker 并顺序执行分支；仍不足则以 `waiting_resources` 等待，定期复查。固定 batch 和研究参数保持不变，变化原因和分配历史分别保存在 `reallocations/`、`assignments/`。

管理进程和 CPU 阶段不开放 CUDA 设备；GPU 工作进程在创建前只开放分配给它的 UUID，启动时核验实际可见设备。各进程的卡号映射和隔离证据保存在 `resources/`。按输出目录安全停止旧任务的命令见[运行说明](docs/xu-model-run.md)。

更多内容见 [分析方法](docs/analysis-method.md)、[数据合同](docs/data-contract.md)、[环境部署](docs/environment.md)、[CPU与恢复](docs/running.md)、[代码更新](docs/update.md) 和 [验证记录](docs/validation.md)。

## 运行测试

使用项目专属环境运行 `python -B -m unittest discover -s tests -v`。测试只创建临时合成输入，实际数据不放入仓库。需要完整环境与足够 RAM 才能执行正式链路测试；资源不足会明确停止。

代码发布目标固定为 `tianwuasd/project` 的 `dataset1282_analysis` 子目录。真实影像、Excel、患者字段、映射、分析结果、环境和本机配置均不属于代码发布内容。
