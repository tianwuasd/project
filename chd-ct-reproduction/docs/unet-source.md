# CHD 论文网络与历史 U-Net 来源

2026-10-15 起，当前六阶段工厂使用 [chd_unet.py](../src/chd_ct/models/chd_unet.py)，按 Xu 等 CHD 论文图 4 和表 3 实现。依据、未明确细节及参数量差异见 [论文对齐记录](paper-mapping.md)。这是根据论文实现的 PyTorch 网络，未取得原作者代码的逐层一致性证据。

## 当前网络

| 网络 | 输入 | 编码通道 | 结构 |
|---|---|---|---|
| crop64 / all64 | 64³ | 64、128、256、512、1024 | 5 个分辨率层级 |
| crop128 / all128 | 128³ | 48、96、192、384、768 | 5 个分辨率层级 |
| blood2d | 512² | 32、64、128、256、512、1024、2048、4096 | 8 个分辨率层级 |
| blood_lstm | 7×32×512² 特征 | 每个方向隐藏通道 32 | 2 层双向 ConvLSTM，中心切片输出 |

每个 U-Net 编码/解码块包含两次 `Conv3 → InstanceNorm → LeakyReLU`；下采样使用 MaxPool2，上采样使用 ConvTranspose2，跳跃连接拼接特征，最后用 Conv1 得到类别输出。3D 输出为背景和 ImageCHD 七结构，共 8 类；血池输出为背景、内部、边界，共 3 类。blood2d 的最后分类卷积之前有 32 通道特征，冻结后提供给循环模型。

位置关联仅在四个 3D 阶段默认开启：`MaxPool4 → Conv3+IN+LeakyReLU（8→64→64→8）→ ConvTranspose4`。U-Net 概率与位置先验逐点相乘；实现使用对数域，后续 softmax 归一化。该分支的拓扑来自图 4，概率激活选择见对齐记录。

```yaml
stages:
  crop64:
    spatial_gate: false  # 消融时关闭；其余字段保持原配置
```

四个 3D 阶段分别控制，开关记录在权重配置中；不能在预测时更改已训练模型的开关。2D 不接受该开关为 true，图 4(b) 没有画出对应分支。

## 网格、规模与权重

标准 64³、128³、512² 网格直接进入主干，不再为有效卷积扩展到 156³、220³ 或 444²。自定义奇数/极小网格会对称镜像补齐到下采样倍数，再中心裁剪回原网格；不通过插值拉伸输出。

`chd.yaml` 是论文明确参数规模；`chd-smoke.yaml` 和 `--mode smoke` 是 size16/base2/三层的软件短测。正式配置开启 `activation_checkpoint: true`：训练时重算编码/解码块和位置分支的中间结果，以减少保存的张量；不改变 batch4、参数量或损失的批次聚合。推理、冻结特征编码和 smoke 不启用重算。正式规模仍先做 GPU preflight，形状检查通过不代表显存足够。当前没有 full-size GPU 训练或论文精度结果。

模型格式为 `imagechd7-chd-paper-v3`，architecture 为 `chd-media2023-fig4-v1`，grid_adapter 为 `same-pad-symmetric-lattice-crop-v1`。旧权重必须重新训练；模型集合检查和恢复契约会拒绝结构混用。预处理缓存与 Excel 重采样不因本次主干变更而失效。

## 保留的历史参考

- [unet.py](../src/chd_ct/models/unet.py)：Freiburg 2015 二维与 2016 三维 no-BN Caffe 定义的 PyTorch 移植。
- [grid.py](../src/chd_ct/models/grid.py)、[gate.py](../src/chd_ct/models/gate.py)：旧有效卷积网格适配与旧空间概率门控。当前主干只复用 grid.py 的通用镜像工具。
- [reference/sources.json](../src/chd_ct/models/reference/sources.json)：原始网络定义、来源与 SHA-256；原始许可证继续保留。
- [修改前完整备份](../back/before-paper-alignment-20261013/SNAPSHOT.md)：含旧工厂、配置、训练/预测、Excel 重采样和服务器代码。旧 `chd-author-unet.yaml` 只在这个版本运行。

历史主干的独立网络图与数值一致性测试仍保留，但这些测试只证明历史移植，不证明当前 CHD 论文实现与原作者程序完全一致。
