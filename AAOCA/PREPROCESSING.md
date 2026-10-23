# AAOCA CT 预处理

本模块把已经标定为 HU 的三维 NIfTI CT 转成可复用的 float32 研究影像。固定灰度规则：

```text
normalized = clip(HU, -1000, 1000) / 1000
-1500 → -1; -1000 → -1; -500 → -0.5; 0 → 0; 500 → 0.5; 1000 → 1; 1500 → 1
```

NIfTI 的 slope/intercept 由 NiBabel 读取时应用一次。输出灰度是无量纲的 `[-1,1]`，**不再是 HU**。输入必须来自已经正确标定 HU 的 CT 转换；不根据最小最大值猜测 HU 偏移，也不对每个体积独立做 min-max 或 z-score。没有分割标签也可使用。

## 安装

预处理支持 Python 3.10+，建议独立 Python 3.11 环境；仅需 NumPy、SciPy、NiBabel、psutil。无需桌面 GUI、PyTorch、CUDA、nnU-Net 或 GPU。

```bash
git clone https://github.com/tianwuasd/project.git
cd project/AAOCA
python3 -m venv /path/to/aaoca_env
/path/to/aaoca_env/bin/python -m pip install .
/path/to/aaoca_env/bin/aaoca-preprocess --help
```

Windows 对应环境解释器位于 `aaoca_env/Scripts/python.exe`。若只需从源码运行，可安装 `requirements-preprocess.txt`，随后使用 `python /path/to/project/AAOCA/preprocess_aaoca.py`。不需要安装主目录的桌面 `requirements.txt`。安装包只包含 `aaoca_preprocessing`，不会打包其他工程或研究数据。

## 推荐执行顺序

`--input` 指向原始 HU NIfTI 目录，`--output` 指向独立的新目录，两者不得相互包含。下面路径为示例，需要替换为自己的目录。

```bash
# 1. 只读检查当前执行机器的环境与明确路径，不处理影像
aaoca-preprocess doctor --path /path/to/AAOCA --path /path/to/results-parent

# 2. 只读头信息计划：输出体积数、输入/输出网格、内存估计与未压缩容量
aaoca-preprocess plan --input /path/to/AAOCA

# 3. 实际预处理：默认保留原分辨率，进行 RAS 轴重排和固定 HU 归一化
aaoca-preprocess run --input /path/to/AAOCA --output /path/to/AAOCA_normalized_v1

# 4. 读取最后保存的进度
aaoca-preprocess status --output /path/to/AAOCA_normalized_v1
```

默认 `--workers 1 --chunk-depth 8 --max-case-gib 8`。CPU 是主要计算资源，压缩读写也会影响速度；共享 NAS 建议从 1 个病例开始，按实际内存和 I/O 测量再提高 `--workers`，不要按整机逻辑线程数直接开满。运行前检查个人配额，程序检查到的文件系统空闲空间不代表个人配额。

`--max-case-gib` 限制保守估算的单病例数组工作内存，不是操作系统硬上限；源体积仍需读入内存，输出使用磁盘映射与分块插值。并发工作内存估计超过当前可用内存的 75% 会拒绝启动；能识别的 Linux cgroup 内存限额也参与检查。CPU affinity 只约束可见范围，不能证明共享服务器分配。调高限额前核实实际获准资源。

大体积的临时 float32 数组默认写入 `OUTPUT/.scratch`；可用 `--scratch /path/to/allocated-scratch` 显式选择已获准、容量足够的路径。不会自动把 NAS 数据搬到本地盘，也不假定本地盘更快。首次失败、参数冲突或容量不足均有明确错误信息。

## 可选空间重采样

默认不改变分辨率，避免在只有灰度需求时重复插值。明确需要近 0.5 mm 时，在 `plan` 与 `run` 上都加相同参数，并使用新的结果目录：

```bash
aaoca-preprocess plan --input /path/to/AAOCA --spacing 0.5 0.5 0.5
aaoca-preprocess run --input /path/to/AAOCA --output /path/to/AAOCA_05mm_normalized_v1 \
  --spacing 0.5 0.5 0.5 --workers 1
```

三个间距值对应 RAS 重排后的轴。尺寸使用 `round(native_shape * native_spacing / target_spacing)`；保持完整体素单元视野与中心，实际间距由整数输出尺寸决定，可能略偏离名义值。保留斜切和剪切，空间头记录真实 affine；存在剪切时只编码准确的 sform，qform code 设为 0，避免伪造一个不准确的正交变换。

处理顺序为：校验空间头 → RAS 轴重排 → 可选 HU 线性插值 → HU 截断 → 除以 1000 → float32 NIfTI。边界采用最近边缘值；重采样是线性插值，未添加下采样抗混叠滤波。RAS 轴顺序不等同于消除斜切，也不等同于患者间配准。没有裁剪、去噪、缺片修复、分割、训练或数据集划分。

空间单位未知、不含有效变换、qform/sform 冲突、非三维、NaN/Inf、截断 gzip 等输入会失败并记录，不能通过猜测坐标或 HU 来自动修复。

## 输入发现与原始清单

目录模式递归发现 `.nii` / `.nii.gz`；忽略隐藏目录、`研究版*`、`preprocessed`、`results`、`labels`、`masks` 等派生目录，以及名称中独立的 `label` / `mask` / `seg` 标记。它不是自动判断医学模态的分类器；只选明确存放原始 HU CT 的目录。输出目录不可放进输入树。

已有研究工作区的 `CT总清单.json` 放在输入根目录时会自动使用，并只选择 `type=AAOCA`。也可以通过 `--manifest` 指定严格清单：

```json
[
  {"type": "AAOCA", "input": "subject_a/ct_a.nii.gz"},
  {"type": "AAOCA", "input": "subject_b/ct_b.nii.gz"}
]
```

`input` 或旧格式的 `destination` 必须相对 `--input`；可选 `sha256` 在正式运行时强校验。跨目录、绝对路径、重复源文件及标签/派生文件条目会被拒绝。完整项目清单与单独 AAOCA 文件夹的相对根不同，不直接混用。缺少清单时，公开命令可直接处理服务器上的 AAOCA 目录。

## 结果与续跑

```text
OUTPUT/
  dataset.json                  固定算法、依赖和实现签名
  progress.json                 当前/最后一次运行的进度、资源与问题
  manifest.json                 最后一次完成的运行清单
  failures.json                 最后一次完成的运行失败明细
  cases/<relative-path-hash>/
    image.nii.gz                float32 [-1,1]，独立的最小空间头
    record.json                 输入/输出哈希、几何映射、统计、全像素回读结果
```

病例键来自相对路径的哈希，只用于稳定定位体积，不能当患者编号；一个患者的多个序列/期相分别处理。源相对路径保存在本地记录中，后续模型划分仍需可靠患者映射。

**续跑就是重复相同命令。** 输入 SHA256、算法/依赖/实现签名、输出 SHA256 都一致时才复用。源文件改变、结果损坏或参数改变时保留已有成果并拒绝覆盖，使用新输出目录重新处理；不提供悄悄覆盖旧成功结果的开关。修改代码或依赖版本也会改变签名。

输出影像和记录在同一临时目录完成，全像素解压回读、CRC、空间头校验通过后一起发布。单病例失败不影响其他病例。输出目录有操作系统互斥锁；锁文件留在磁盘不等于仍在运行，不要手动删除它。必须使用支持互斥锁及同文件系统原子重命名的文件系统。

Ctrl+C 取消未开始的病例，等待已经开始的病例结束，然后记为中断。强制终止或断电可能留下 `.staging` / `.scratch` 临时目录；它们不会被当作完成，也不会在续跑时自动删除历史残留。确认无运行进程后再人工检查清理。`status` 是最后保存的快照，进程被强杀时可能仍显示 running；它不是存活监控。

记录含源相对路径、几何和处理信息，应保存在私人结果目录。公开 Git 只发布代码、合成测试和说明，不上传输入 CT、真实运行清单、服务器笔记或输出影像。

## Linux 服务器启动

源码入口支持从任意当前目录调用：

```bash
export AAOCA_PREPROCESS_PYTHON=/path/to/aaoca_env/bin/python
bash /path/to/project/AAOCA/preprocess_server.sh plan --input /path/to/AAOCA
bash /path/to/project/AAOCA/preprocess_server.sh run \
  --input /path/to/AAOCA --output /path/to/AAOCA_normalized_v1
```

脚本不安装环境、不自动更新 Git、不占 GPU。若要断开 SSH 后继续运行，先选择独立的私人日志目录：

```bash
mkdir -p /path/to/private-logs
nohup bash /path/to/project/AAOCA/preprocess_server.sh run \
  --input /path/to/AAOCA --output /path/to/AAOCA_normalized_v1 \
  > /path/to/private-logs/aaoca-preprocess.log 2>&1 < /dev/null &
```

后台启动成功不等于处理成功，随后用 `status` 和日志检查。shell 入口为常用数值库默认设置 1 个线程，病例并行由 `--workers` 控制；尊重用户已有线程环境设置。

## Python 接口与验证

```python
from aaoca_preprocessing import Config, normalize_hu
from aaoca_preprocessing.pipeline import plan, run

config = Config()  # 或 Config(target_spacing=(0.5, 0.5, 0.5), workers=2)
report = plan('/path/to/AAOCA', config)
summary = run('/path/to/AAOCA', '/path/to/AAOCA_normalized_v1', config)
```

```bash
python -m unittest discover -s tests -p 'test_aaoca_preprocessing.py' -v
```

测试只生成合成影像，覆盖 HU 边界、缩放、空间方向、剪切、完整视野、独立采样对照、分块、输入发现、失败隔离、校验续跑、锁和命令行。数值网格复用 `tools/research_ct/research_preprocess.py` 的项目实现；CPU 分块内核使用 SciPy，不需要导入旧脚本的 nnU-Net 环境。原 HU 保留型研究脚本保持独立，旧的 0.5 mm 样例不自动当成新的归一化成果。

本地合成测试与打包验证不代表已在目标服务器部署或完成真实数据全量处理。服务器当前硬件、目录权限和可用空间须在目标机运行 `doctor` 后核对。
