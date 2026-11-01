# 一键启动说明

## 只需先做这几步

1. 打开项目目录，Windows 双击 **`start.bat`**。它优先使用项目 `.venv`，然后尝试 `py -3` 和 `python`。不需要修改 PowerShell 执行策略。
2. 出现中文菜单后选择合成演示、数据目录、CSV 清单或粘贴路径。目录可以是含清单的根目录，或下面有 `data/*.csv` 的项目目录。若发现多个清单，会让你明确选择。
3. 等待环境和数据检查，以及小规模试跑。完成后打开终端中显示的 `report.txt`。

CSV 列至少包括 `case_id,patient_id,split,image,label`，完整流程还需要 `initial_label`。不自动猜患者 ID、类别语义和训练/验证划分，也不会把普通全心标签当作初始血管标注。格式见 [数据规范](data-guide.md)。

## 自动检查的内容

| 环节 | 内容 |
|---|---|
| Python | 3.11 或以上，显示实际解释器路径 |
| 依赖 | numpy、scipy、torch、nibabel、PyYAML、scikit-image、networkx 的导入及最低版本 |
| CPU | 实际张量计算与反向传播 |
| GPU | CUDA 可用性、实际 CUDA 运算、设备名称及显存；失败时 auto 模式使用 CPU |
| 磁盘 | 输出目录可写、至少 1 GiB 空闲；不代表完整训练的数据存储需求 |
| 数据 | 文件存在、3D 影像、有限数值、标签编号、相同网格、患者不能跨集合、train/val 非空 |
| 空间 | affine/spacing 正交一致；带 shear 的数据需要先重采样 |

环境缺失时仍能进入启动器并写出失败报告，不会因为提前 import torch 而直接崩溃。启动器给出安装命令；如果没有合适的环境，先按 README 建立 `.venv`。它不会静默下载依赖或修改全局 Python。

## 试跑做了什么

默认 `smoke` 模式：

1. 自动生成几何体，训练全部 8 个阶段（6 个 3D 任务、血池 2D、血池 LSTM），每阶段仅一轮、每集合一步，然后执行完整融合推理、特征和规则输出。
2. 如果选择了自己的数据，先校验清单里的所有病例。试跑只抽取至多 2 个 train、1 个 val、1 个 test，保持原患者与集合归属；生成最大边长 32 的独立 NIfTI 副本，在副本上完成同样的缩小网络测试。
3. 没有 test 行时，在 val 副本上做推理连通性检查，`result.json` 明确记录 `inference_split=val`，不产生性能评估。
4. train/val 缺少任何 `initial_label` 时，所选数据仅训练 6 个可用阶段，并检查 all128 单模型推理；不写诊断结果，不宣称完整融合通过。

原影像和原清单保持不变。试跑权重和报告被 Git 忽略，仅保存在本地。缩小可能丢失小结构，所以本环节只检查软件连通性，不能判断训练收敛、临床精度或正式 paper 配置的显存需求。

## 看懂报告

| 总结果 | 含义 | 退出码 |
|---|---|---:|
| `passed` | 请求的小规模流程通过；合成演示仍只证明合成流程 | 0 |
| `checked` | 仅完成环境/数据检查，未训练推理 | 0 |
| `partial` | 所选数据基础环节通过，缺初始血管标签而未验证完整流程 | 2 |
| `failed` | 环境、数据或试跑失败，查看错误与 `run.log` | 1 |
| `interrupted` | 用户按 Ctrl+C 中止，保留已有输出 | 130 |

每次运行新建 `runs/quickstart/时间戳_随机后缀/`，也可指定一个尚不存在的输出目录。不会重用试跑 checkpoint，避免把旧权重当成本次验证结果。

## 命令行与服务器

```powershell
# 无需交互，完整合成测试
.\start.bat --demo --non-interactive

# 选择自己的清单并试跑（中文、空格路径均支持）
.\start.bat --dataset "D:\数据目录\manifest.csv" --non-interactive

# 只检查，不进行训练和推理
.\start.bat --dataset "D:\数据目录" --mode check --non-interactive

# 明确要求 CPU；默认 auto 在可用时使用 CUDA
.\start.bat --demo --device cpu --non-interactive
```

Linux/macOS 用 `python start.py` 替代 `.\start.bat`；可运行 `python start.py --help` 查看参数。没有图形桌面或 tkinter 时，可选择粘贴路径；非交互模式应明确提供 CSV，避免多个清单时自动选错。

## 常见问题

- **没有 Python**：启动窗口提示安装 Python 3.11+ 并加入 PATH。
- **找不到数据清单**：目录应有满足上述列名的 CSV；仅有 DICOM/NIfTI 文件不能直接训练。先按数据规范准备清单和标签。
- **标签应位于 [0,10]**：当前数据可能使用其他编号（例如 MM-WHS）；需正确映射，不能直接对编号取模或裁剪。
- **image/label 网格不一致**：先确认这对影像和标签属于同一病例，并正确完成医学影像重采样。
- **部分通过**：补充初始血管的七类标注，再重新运行。
- **小测试通过后如何正式训练**：确认真实数据、标签和计算资源，使用 README 中 `configs/paper.yaml` 的训练命令；本启动器默认只做快速检查。
