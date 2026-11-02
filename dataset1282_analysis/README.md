# dataset1282_analysis

可在 Windows 本机或 Linux 服务器调用的 NIfTI + Excel 只读数据审计工程。首版提供独立环境部署、代码更新、数据统计和本地报告；适合先弄清数据来源与质量，再决定后续处理。

输入影像、标签、Excel 和历史记录不会被修改。程序不会自动改头、重采样、移动标签、剔除病例或开始训练。输出文件对数与不同 PatientID 数分别统计；标签用数值编号呈现，不猜测解剖含义。未知空间单位不按毫米解释，未经核实的灰度不标 HU。

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
| `run.py inventory` | 读取主清单，检查规范路径和文件存在情况 |
| `run.py analyze --stage metadata / headers / full` | 元数据、文件头或全体素审计，三个范围明确区分 |
| `run.py status --run NAME` | 只读取状态，不安装或启动计算 |
| `run.py resume --run NAME` | 从已完整提交且内容摘要匹配的病例恢复 |
| `run.py report --run NAME` | 生成 CSV、JSON、PNG 和 HTML/Markdown 报告 |
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
  observability/                             锁、状态与日志
  reporting.py                               本地报告和数组切面
tests/                                       仅合成行为测试
docs/                                        方法、运行、更新与验证说明
```

未来处理阶段可复用 `CaseInput`、配置、日志、病例队列和完成合同；新增方法须使用独立协议与输出位置。首版没有训练/GPU模块，不把 CPU 短测视为 GPU 加速验证。

更多内容见 [分析方法](docs/analysis-method.md)、[数据合同](docs/data-contract.md)、[环境部署](docs/environment.md)、[CPU与恢复](docs/running.md)、[代码更新](docs/update.md) 和 [验证记录](docs/validation.md)。

## 运行测试

使用项目专属环境运行 `python -B -m unittest discover -s tests -v`。测试只创建临时合成输入，实际数据不放入仓库。需要完整环境与足够 RAM 才能执行正式链路测试；资源不足会明确停止。

代码发布目标固定为 `tianwuasd/project` 的 `dataset1282_analysis` 子目录。真实影像、Excel、患者字段、映射、分析结果、环境和本机配置均不属于代码发布内容。
