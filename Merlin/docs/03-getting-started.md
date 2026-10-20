# Merlin 入门与阅读顺序

第一轮只完成一个目标：**知道一例 CT 经过哪些处理，最后怎样得到 2048 维特征。** 读代码不需要先下载权重；运行示例时才需要环境、模型文件和一个 CT。

## 从哪里看起

| 顺序 | 阅读材料 | 完成标准 |
| --- | --- | --- |
| 1，约 15 分钟 | [项目介绍](01-introduction.md) | 能说明 CT、报告、EHR 在训练和推理中的不同角色 |
| 2，约 20 分钟 | [demo.py](../upstream/documentation/demo.py) | 找到模型初始化、样本列表、DataLoader 和模型调用 |
| 3，约 25 分钟 | [预处理](../upstream/merlin/data/monai_transforms.py)＋[数据加载](../upstream/merlin/data/dataloaders.py) | 写出输入形状、体素间距、强度范围和裁剪规则 |
| 4，约 40 分钟 | [load.py](../upstream/merlin/models/load.py) → [build.py](../upstream/merlin/models/build.py) → [i3res.py](../upstream/merlin/models/i3res.py) | 从 `Merlin(ImageEmbedding=True)` 追到最终返回值 |
| 5，约 20 分钟 | [源码阅读笔记](02-code-reading.md) | 说清 2048／512 维、Tensor／tuple、logits／概率的差别 |
| 6，另安排实践 | 下面的第一例特征提取 | 得到一个含 2048 个有限数值的输出文件，并记录版本 |

这些时间只是安排阅读的参考。遇到不熟悉的 PyTorch 操作，先查明 `permute`、`squeeze`、`unsqueeze`、`eval` 和 `no_grad` 的含义，再继续追调用链。

## 阅读时只追一条路径

在编辑器中打开本目录，依次定位：

```text
from merlin import Merlin
  → merlin/__init__.py
  → merlin/models/__init__.py
  → load.py 的 Merlin
  → _load_model()
  → build.py 的 MerlinArchitecture
  → ImageEncoder
  → i3res.py 的 I3ResNet.forward()
```

先不要展开报告生成和 Docker。把纯图像分支读懂之后，再回到默认模式看文本分支，最后再看 RadLLaMA 与分割。

## 准备运行环境

以下命令从 **`project/Merlin` 目录**执行。需要独立 Python 环境，官方 README 使用 Python 3.10。不要直接复用其他医学影像项目的环境，因为 torch、torchvision、MONAI、Transformers 的组合可能不同。

使用 Conda 的示例：

```powershell
conda create -n merlin-study python=3.10
conda activate merlin-study
python -m pip install -e ./upstream
python -m pip install h5py
```

可编辑安装让 Python 使用本目录的源码。`h5py` 用于下面的特征输出脚本，不在主包的依赖列表中。

另一条选择是使用上游的锁文件：

```powershell
uv sync --project upstream --python 3.10
```

两条路线选一条即可。锁文件路线的环境在 `upstream/.venv`，后续命令要使用该环境的 Python；运行特征提取脚本仍需确认其中装有 `h5py`。普通 pip 路线遵循依赖下界，不能保证解析出作者当时测试的全部版本。

只检查环境，不加载模型：

```powershell
python -c "import torch, torchvision, monai, transformers; print('torch', torch.__version__); print('torchvision', torchvision.__version__); print('MONAI', monai.__version__); print('Transformers', transformers.__version__); print('CUDA available', torch.cuda.is_available())"
```

首次实例化 `Merlin` 会下载或读取 Merlin 权重，也可能下载 ResNet-152 和 Clinical Longformer。报告生成还涉及 RadLLaMA-7B，官方文档提示相关权重约 25 GB，因此第一轮先使用图像特征模式。

在本机访问 GitHub、默认 PyPI、Hugging Face 和 PyTorch 下载服务时，使用已配置的本地代理 `http://127.0.0.1:7897`；例如可在该 PowerShell 会话临时设置 `HTTP_PROXY`、`HTTPS_PROXY`。服务器上的 `127.0.0.1` 指服务器本身，不能直接照抄本机代理地址。国内镜像按其自身路由访问。

## 第一例使用已有特征提取脚本

本版本已有 [extract_feat_LP.py](../upstream/documentation/Merlin_CVPR26CTFM/extract_feat_LP.py)，可读取 NIfTI 文件并写出 HDF5 特征。用一例已获准用于研究、HU 和空间信息正确的 `.nii.gz`，放入本地 `data/smoke-input`。这些数据目录已在本项目忽略规则中排除。

```powershell
python upstream/documentation/Merlin_CVPR26CTFM/extract_feat_LP.py -i ./data/smoke-input -o ./outputs/smoke-features --batch_size 1 --num_workers 0
```

该脚本没有 mask 时使用中心裁剪，有 CUDA 时使用 GPU，否则选择 CPU。CPU 分支存在不等于本机已经验证可接受的耗时；先用一例和 batch size 1。

预期每例产生一个 `.h5`，内部键为 `y_hat`，形状为 `(2048,)`。把下面的 `CASE_ID` 换成实际文件名：

```powershell
python -c "import h5py, numpy as np; f=h5py.File('./outputs/smoke-features/CASE_ID.h5','r'); x=f['y_hat'][:]; print('shape', x.shape, 'finite', np.isfinite(x).all()); assert x.shape==(2048,); assert np.isfinite(x).all(); f.close()"
```

一个文件能生成，只能证明流程运行了。还应查看预处理后的切片，确认目标心脏在裁剪范围内、方向正确、HU 正常。否则模型可能处理的是错误视野。

### 如果使用 ROI mask

脚本支持 `--masks_path`，但 mask 必须与图像同名，且其空间信息与图像对应；该实现使用前景标签 `1`。指定 mask 后，找不到对应 mask 的图像会被过滤，而不是自动回退到全卷处理。必须核对控制台的 `Found ... cases` 与预期病例数。

ROI 和无 ROI 的代码在 [transforms.py](../upstream/documentation/Merlin_CVPR26CTFM/transforms.py)。ROI 裁剪是已有挑战代码的功能，不是论文已经验证对先心诊断有效的结论。

## 原始 demo 的注意点

[demo.py](../upstream/documentation/demo.py) 同时演示多个任务，并多次构造模型。它虽然设置了 `device`，仍直接调用 `.cuda()`，因此不能仅靠那行设备判断就让原始 demo 在 CPU 上运行。

第一轮使用上面的特征提取入口更容易检查输出。若要了解图文匹配，再阅读 demo 的默认 `Merlin()` 分支，并对照源码确认返回值。

## 常见问题先查哪里

| 现象 | 优先检查 |
| --- | --- |
| 初始化一直下载 | Merlin、ResNet 和 Clinical Longformer 是不同资源；逐项看日志和缓存 |
| 没有 CUDA | 当前 Python 中的 torch 构建、驱动和 GPU 可见性；不是单纯删除 `.cuda()` 就完成配置 |
| 图像模式仍下载文本模型 | 当前构造函数无条件建立 TextEncoder，属于这份版本的实现 |
| 输出形状多一维 | 纯图像原始输出为 `[1,B,2048]`，现有提取脚本使用 `[0]` |
| 更改预处理后结果不变 | 换一个缓存目录，避免旧图像缓存被重用 |
| 病例数变少 | 检查 `.nii.gz` 后缀筛选，以及启用 mask 后的同名配对 |
| 显存不足 | 保持 batch size 1，先不加载报告生成；记录实际峰值再决定配置 |
| torchvision 或 Transformers 报兼容性错误 | 对照 `pyproject.toml` 和 `uv.lock`，不要在其他项目环境里盲目升级 |

## 第一轮完成后再做什么

- 想做病例分类：冻结编码器，提取特征，建立患者级划分的轻量分类基线。
- 想做图文任务：读 512 维对齐分支，明确正负提示与评价方式。
- 想做分割：转到 [Merlin-nnUNet](https://github.com/ashwinkumargb/Merlin-nnUNet)，检查编码器与解码器连接。
- 想复现预训练：先列出公开仓库缺失的数据处理和训练模块，不把推理示例当完整训练程序。

本次归档未执行环境安装、权重下载或模型推理。完成上述实践后，应记录环境版本、输入形状、病例数量、输出形状及运行日志，作为实际运行证据。
