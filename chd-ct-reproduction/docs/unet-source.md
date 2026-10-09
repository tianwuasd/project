# U-Net 原作者来源、移植与 gate 开关

## 来源

本项目 0.3 使用 **按原作者 Caffe 网络定义移植的 PyTorch 主干**。它不是作者发布的官方 PyTorch 软件，也不是第三方 wolny/pytorch-3dunet，更不等于 CHD 论文完整系统。

- 二维：[Ronneberger / Fischer / Brox 官方页面](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)，2015-10-02 发布包中的 `phseg_v5-train.prototxt`。
- 三维：[Çiçek 等官方代码页](https://lmb.informatik.uni-freiburg.de/lmbsoft/unet.en.html)，`3dUnet_miccai2016_no_BN.prototxt`，明确选用作者提供的无 BatchNorm 版本。
- 原始定义、许可证、发布包及文件 SHA256 均保存在 [reference](../src/chd_ct/models/reference/sources.json)。不下载或加载官方细胞/肾组织权重作为心脏模型。

## 主干对齐了什么

| 项目 | 二维官方发布网络 | 三维官方 no-BN 网络 |
|---|---|---|
| 编码深度 | 5 层，4 次池化 | 4 层，3 次池化 |
| 卷积 | 无 padding 的 3×3 卷积 | 无 padding 的 3×3×3 卷积 |
| 激活 | ReLU，包括上采样后 | ReLU，包括上采样后 |
| 归一化 | 无 | 无；没有混入 InstanceNorm/GroupNorm |
| 跳跃连接 | 中心裁剪后，按上采样特征、编码特征顺序拼接 | 同左 |
| Dropout | 编码最后两层，概率 0.5，训练时启用 | 无 |
| 原始通道 | 每层 64/128/256/512/1024；最后上采样输出 **128**，按实际发布文件保留 | 各编码块先 32/64/128/256，再 64/128/256/512；上采样保持输入通道数 |
| 参数初始化 | Caffe Xavier 默认 FAN_IN 规则 | Caffe MSRA 默认 FAN_IN；第一卷积 bias=-0.1 |
| 原始尺寸例子 | 输入 572² → 输出 388² | 输入 116×132×132 → 输出 28×44×44 |

二维发布文件的最后上采样通道与常见简化示意不同，本项目以实际发布文件为准。主干只负责神经网络，不移植官方 Caffe 数据层、训练优化器、数据增强或推理二进制。

## 项目中的三个文件

1. [models/unet.py](../src/chd_ct/models/unet.py)：有效卷积主干 `UNet`，自身输出比输入小，不包含 gate。默认参数使用该维度的作者通道数和深度。
2. [models/grid.py](../src/chd_ct/models/grid.py)：`GridUNet` 对输入进行对称镜像扩展，把有效输出中心裁剪回目标网格；用于现有 CHD 标签、融合和空间还原。这里的边界适配是本项目实现，不能用插值把有效区域拉伸到整图。
3. [models/gate.py](../src/chd_ct/models/gate.py)：保留的空间概率门控，明确属于本项目扩展。

`features()` 返回完整编码、解码后的特征，blood_lstm 使用 blood2d 的冻结特征。训练/预测的独立脚本与数据缓存格式不变。

## 默认开启 gate，如何关闭

`configs/chd.yaml` 和作者通道配置中的 `crop64/crop128/all64/all128` 均默认 `spatial_gate: true`；blood2d 保持原来未接 gate 的设置，blood_lstm 结构不受此次修改影响。

若要关闭某一阶段，在该阶段配置中写：

```yaml
stages:
  crop64:
    spatial_gate: false
    # 其余 size/base/levels/epochs/batch 参数保留
```

完全关闭三维门控对照时，将上述四个阶段都改为 `false`。YAML 用布尔值，不要写字符串 `"false"`。开关会写入有效配置和模型记录。不能在预测时随意改变已训练模型的开关，需要按对应配置重新训练。

## 尺寸与显存

配置的 `size` 现在表示适配后与标签一致的目标网格，原始主干实际接收的镜像扩展网格更大：

| 当前配置 | 实际主干输入 | 主干有效输出 | 最终网格 |
|---|---|---|---|
| 3D、4层、size64 | 156³ | 68³ | 64³ |
| 3D、4层、size128 | 220³ | 132³ | 128³ |
| 2D、5层、size256 | 444² | 260² | 256² |

因此旧版同尺寸卷积的显存估算不适用；正式训练之前必须做 GPU preflight。当前未验证 RTX 3090 能否承载所有默认或作者通道配置。

- `chd.yaml`：实用宽度，3D base16/4层，2D base8/5层；符合作者通道排列但宽度缩小。3D base 指每块第一卷积的基础通道，其第一块输出是 2×base。
- `chd-author-unet.yaml`：作者主干宽度/深度，3D base32/4层，2D base64/5层。类别数仍适配 ImageCHD，三维 gate 默认开启，因此整个模型仍含本项目扩展。
- `chd-smoke.yaml` 或 `--mode smoke`：缩小深度和通道，仅验证软件流程，不等同作者原规模模型。
- 旧 `chd-paper-scale.yaml` 已移入 [历史目录](../back/custom-unet-v1/README.md)，防止旧八层配置造成误解。

## 权重与验证边界

新模型格式为 `imagechd7-author-unet-v2`，记录主干和网格适配版本。旧 `imagechd7-multistage-v1` 权重无法兼容，必须重新训练；原始数据和预处理缓存可继续复用。

测试逐项对照官方文件的卷积次序、核、通道、输出形状；另用独立解析官方网络图的 PyTorch 函数执行器，在相同随机权重下比较输出。官方宽度的形状检查采用 meta 张量，不代表真实显存测试。没有运行原始 Caffe 二进制进行跨框架数值一致性验证，也未复现官方原任务或 CHD 论文的训练结果。
