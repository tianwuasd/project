# Whole Heart 数据探索工具

这是面向初学者的只读数据探索代码，带中文注释。输入三维 CT/MRI 和对应分割标签，输出中文说明书、逐病例信息表、标签体积表、组间统计图和代表病例预览。

读取使用 NiBabel，统计使用 NumPy，绘图使用 Matplotlib。这个阶段无需安装 nnU-Net、MONAI，也不需要显卡。

## 1. 安装依赖

需要 Python 3.10 或更高版本。推荐在独立虚拟环境运行，避免影响其他项目。

本次实际验证环境为 Python 3.14.3、NumPy 2.4.2、NiBabel 5.4.2、Matplotlib 3.10.8。`requirements.txt` 提供最低版本范围；`requirements-tested.txt` 记录本次精确版本，较旧 Python 请使用前者让安装器选择兼容版本。

在这个目录打开终端后执行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

如果你使用本机 7897 代理访问 PyPI，可在安装命令末尾加入 `--proxy http://127.0.0.1:7897`。分析过程本身不联网、不下载病例、不训练模型。

## 2. 运行

工具放在 `whole_heart/数据探索工具` 时，默认会找到相邻的 `Wholeheart_Train_Dataset`：

```powershell
# 先快速试跑 2 例。此报告仅代表已扫描病例，不代表整个数据集。
.\.venv\Scripts\python.exe analyze.py --limit 2

# 全量运行：完整读取每份影像和标签，统计所有标签，计算文件哈希。
.\.venv\Scripts\python.exe analyze.py

# 也可以明确指定输入与新的输出目录；路径有空格时一定加引号。
.\.venv\Scripts\python.exe analyze.py --input "D:/code_project/whole_heart/Wholeheart_Train_Dataset" --output "D:/code_project/whole_heart/分析结果_新一轮"
```

结果默认在 `results/运行时间/`。双击 `数据说明书.html` 即可离线阅读，图片已内嵌；也提供 Markdown。已有非空输出目录不会被覆盖。

## 3. 输出说明

| 文件 | 用途 |
|---|---|
| 数据说明书.html / .md | 先读这一份：整体、单例、标签、组间差异、问题 |
| 病例信息.csv | 每个病例的模态、尺寸、间距、强度、SHA256 与问题 |
| 标签体积.csv | 每例每种标签的体素数、mL 体积和占比 |
| 分组概览.csv | 每组病例数量和间距摘要 |
| 问题清单.csv | 缺失文件、异常标签、空间不一致等 |
| 完整统计.json | 包括 qform/sform 的全部结构化结果 |
| 运行信息.json | 运行范围、抽样规则、版本、只读检查 |
| 逐例检查记录.jsonl | 每例完成即写入；不自动续跑。批次重复检查在之后进行，以完整统计.json 为准 |
| figures/ | 单独导出的 PNG 图 |

CSV 使用 UTF-8 BOM，可用 Excel 打开；比例为 0～1，不是百分数。

## 4. 看代码的顺序

1. `analyze.py`：主流程和命令行参数，负责把各步骤串起来。
2. `medical.py`：理解 NIfTI 文件头、实际缩放、标签计数和物理体积计算。
3. `charts.py`：统一类别配色、按物理比例显示三方向图像、绘制组间差异。
4. `reporting.py`：将统计写成 CSV/JSON 和中文说明书。
5. `tests/test_medical.py`：已知答案的合成影像，检查统计逻辑是否正确。

初学者可以先按 [代码阅读导览](代码阅读导览.md) 对照“输入 → 统计 → 绘图 → 报告”阅读。完整运行示例的统计结果保存在 `results/完整数据探索_已验证/`。

## 5. 重要约定

- 原始影像只读，不修改 421，不修复文件头，不删除任何病例。
- NIfTI 灰度缩放由 NiBabel 应用一次。不要再手动给 CT 减 1024。
- 标签计数、NaN/Inf 和 min/max 检查覆盖全部体素；分位数每例最多均匀抽样 200000 个体素，包括背景。
- 单病例依次读取，内存需求与最大病例大小有关；大幅三维影像建议预留数 GB 内存。
- 体积优先按标签有效空间矩阵的行列式计算；矩阵无效时输出间距推算的名义值，表格标出依据。空间单位不明确时留空。空间信息异常病例的体积需要核实。
- 三方向图只是轴交换/翻转到接近 RAS，保留斜位扫描的倾斜；不做插值、不冒称标准放射学视图。
- qform/sform 编码不同不一定错位；比较的是实际物理矩阵。首选矩阵无效时，仅诊断性单图显示可使用有效 qform，并禁止叠加。
- SHA256 只识别文件字节重复，不判断患者身份、近似重复或不同格式的相同影像。
- 预览默认每组一例，并额外显示最多 8 例问题病例；详细检查统计覆盖全部已扫描病例。
- 强度/尺寸/体积差异不等于疾病差异，不提供临床判断。
- C/D 目录作为一个已公开的数据组，不推断个体属于 C 还是 D。

## 6. 可选参数

```text
--limit N                 只处理前 N 例，便于调试
--samples N               每例强度分位数的抽样上限
--skip-hash               跳过文件字节重复检查，降低磁盘读取
--no-previews             不生成病例预览，仍生成统计图
--max-issue-previews N    额外问题病例图上限
```

读取失败时继续处理其他病例，最终返回退出码 2；标签异常等技术问题记录在报告，不代表程序崩溃。未发现问题也不证明医学标注完全准确。

## 7. 运行测试

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

官方标签来源：https://zmic.org.cn/care_2026/track_wholeheart/ 。更多学习资料在相邻 `参考资料` 文件夹。
