# AAOCA / TAPVC 病例与影像整理工具

在本地整理 CT、超声、PDF 和患者主表，逐文件/序列检查并转换可用影像，提供中文面板、进度、暂停、安全停止、增量处理和独立修复窗口。原始资料保留。

本目录仅发布程序、合成测试及公开说明。真实患者资料、姓名编号映射、修复政策、数据库、报告和研究影像不在 Git 中。

## AAOCA 独立粗分割

在本目录运行 `bash start_segmentation.sh --runtime /srv/private/aaoca-seg-terminal`，打开与 CT 任务管理方式一致的中文终端面板。支持环境/许可配置、多 GPU 病例并行、统一 CPU/内存/读写预算、后台运行、历史输入、续跑及另建任务重跑。输入必须是保留 HU/mm 的原图，不能直接使用下面预处理产生的 `[-1,1]` 图像。操作见 [独立分割面板](SEGMENTATION_WORKBENCH.md)。

## AAOCA 独立预处理

可安装的 CPU 预处理模块：对原始 HU NIfTI CT 执行线性重采样与 `clip(HU, -1000, 1000) / 1000`，输出 float32 `[-1,1]`。完整预处理入口要求显式选择自动间距或三个毫米数值；兼容的 `run` 入口仍可只归一化。支持资源限额、分块并行、阶段耗时、后台停止/续跑与哈希校验。**预处理环境支持 Python 3.10+，不需要下面桌面面板的 GUI 依赖。**

安装与 Linux 服务器运行见 [PREPROCESSING.md](PREPROCESSING.md)。主入口：`python preprocess_aaoca.py --help`；安装后为 `aaoca-preprocess`。这次发布的是代码，不能据此认定真实数据已全量处理。

1.1 工作流与 nnU-Net 自动间距的区别见 [PREPROCESSING_WORKFLOW.md](PREPROCESSING_WORKFLOW.md)；自动建议不等于完整 nnU-Net 规划，也不替代任务效果验证。

1.2 新增 [中文服务器工作台](SERVER_WORKBENCH.md)：`bash start_aaoca_server.sh` 或安装后的 `aaoca-server`。结构参考 CHD 任务面板，保留配置菜单；目标间距、CPU、内存默认自动估算，可手动调整，支持命名任务、进度、日志、历史、停止和续跑。该入口独立于下面的原始资料整理/转换面板，目前仅接入预处理。

## 安装和启动

使用带 Tcl/Tk 的 Python 3.12 或以上版本：

```powershell
python -m venv .venv
& './.venv/Scripts/python.exe' -m pip install -r requirements.txt
```

双击 **一键处理.cmd**。默认使用本目录 `.venv`；也可设置 `AAOCA_PYTHON` 指定 `pythonw.exe`。如果使用可移植 Python，可将与该解释器完全匹配的 Tcl/Tk 资源放在 `.venv/tcl`，程序会自动定位，不混用不同版本的资源。

## 四种操作

| 操作 | 行为 |
|---|---|
| 添加资料 / 继续处理 | 接收任意组合的 CT、超声、PDF、主表；登记新增/变化资料，复用有效成果 |
| 修复数据… | 独立窗口选择唯一姓名匹配、验证完整副本、明确期相拆分，显示进度和统计变化 |
| 仅更新报告 | 按已有登记刷新快照、分类及报告，不重新扫描和转换影像 |
| 只检查，不转换影像 | 登记与核对，不把未转换的影像计为成功 |

缺任一种资料不会阻止其他资料处理。单个失败不连带阻断其他独立单元。结果按八种齐全组合分类，问题按编号关联、数据本身、资料缺失分组。PDF 文字不足只备注，CT 原始序列和期相输出体积分开统计。

修复必须有唯一身份、完整像素或明确期相等证据。不会自动补缺失像素、猜测超声时间、OCR或作临床诊断。已有成功成果保留，失败重试不覆盖旧成果。详细行为见 [项目升级使用说明](项目升级使用说明.md)。

## 当前命令行入口

```powershell
python run_project.py --input 'D:/sample/new_batch' --output 'D:/sample/results'
python run_project.py --output 'D:/sample/results' --action report
python run_project.py --output 'D:/sample/results' --action retry --match-names --recover-copies --split-phases
```

最后一条会实际尝试选中的修复；不要将其当作查看状态。可用重复 `--input` 传入多批资料，`--patient` 限定转换编号。已有项目的数据库与成果应一起备份。旧 `run_pipeline.py` 是历史单批流程，不直接重跑已升级的正式结果。

报告标题可在结果目录 `project.json` 的 `report_title` 中配置；未配置时对 TAPVC_result 使用 TAPVC 名称，其余使用 AAOCA。只修改展示标题，不更改分类路径。

## 文件结构

- `pipeline_gui.py`：主面板及独立修复窗口。
- `run_project.py`：增量处理、重试、报告命令行。
- `patient_pipeline/project_*.py`：登记、身份、处理、修复、分类和报告。
- `convert_ct.py` / `convert_us.py`：独立转换器，见 [说明](独立转换器说明.md)。
- `repair_policy.py`：读取外部已确认政策，不包含真实患者映射。
- `tools/research_ct/`：可选的研究 CT 整理、格式核对和重采样，见 [说明](tools/research_ct/README.md)。
- `tools/coarse_segmentation/`：[AAOCA 11 类粗分割与独立服务器网页面板](tools/coarse_segmentation/README.md)，支持预检、批处理、复用核验和模型日志；输入为原始 HU CT，环境独立。
- `aaoca_preprocessing/`：独立预处理 Python 包；`preprocess_aaoca.py` 与 `preprocess_server.sh` 为源码入口。
- `tools/chd_dataset/`：加密标签解压、NIfTI 影像/标注完整检查及原字节整理，见 [说明](tools/chd_dataset/README.md)。空间网格不同的文件单独保留待复核，不自动移动标签或改头。
- `tests/`：仅使用合成资料的回归测试。
- [PROJECT_HISTORY.md](PROJECT_HISTORY.md)：公开开发历程及接手说明。

研究工具使用单独依赖环境和外部数据目录，不属于主面板自动处理步骤。无标注影像可做研究预处理和自监督学习准备；HU 保留、归一化、配准及模型训练应分别记录，不能混称已经完成。

## 测试

```powershell
python -m unittest discover -s tests -v
```

Windows 建议用短临时路径。测试包含真实 Tk 控件和合成影像流程，需要可用的本地窗口环境。源码与本地部署镜像按哈希核对后同步；临床文件始终排除。
