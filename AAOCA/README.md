# AAOCA / TAPVC 病例与影像整理工具

在本地整理 CT、超声、PDF 和患者主表，逐文件/序列检查并转换可用影像，提供中文面板、进度、暂停、安全停止、增量处理和独立修复窗口。原始资料保留。

本目录仅发布程序、合成测试及公开说明。真实患者资料、姓名编号映射、修复政策、数据库、报告和研究影像不在 Git 中。

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
- `tools/chd_dataset/`：加密标签解压、NIfTI 影像/标注完整检查及原字节整理，见 [说明](tools/chd_dataset/README.md)。空间网格不同的文件单独保留待复核，不自动移动标签或改头。
- `tests/`：仅使用合成资料的回归测试。
- [PROJECT_HISTORY.md](PROJECT_HISTORY.md)：公开开发历程及接手说明。

研究工具使用单独依赖环境和外部数据目录，不属于主面板自动处理步骤。无标注影像可做研究预处理和自监督学习准备；HU 保留、归一化、配准及模型训练应分别记录，不能混称已经完成。

## 测试

```powershell
python -m unittest discover -s tests -v
```

Windows 建议用短临时路径。测试包含真实 Tk 控件和合成影像流程，需要可用的本地窗口环境。源码与本地部署镜像按哈希核对后同步；临床文件始终排除。
