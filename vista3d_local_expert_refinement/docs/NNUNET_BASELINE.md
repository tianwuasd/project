# wholeheart nnU-Net 监督基线

2026-10-25：用户确认先实现 nnU-Net baseline，再继续 VISTA3D／局部专家方案。本版代号 **B0**，只有 CT 监督学习。代码复用并隔离适配已有 `whole_heart/project1` 的计算引擎与任务面板，未修改旧项目。VISTA 权重、局部专家、半监督一致性、点击采纳器均未实现；研究目标仍保留。

## 数据与标签预算

原始目录显式传入，包含 `A ct_train`、`B ct_train`、`G ct_train`，各 20 对 `CaseXXXX_image.nii.gz` / `CaseXXXX_label.nii.gz`。沿用既有几何质控，隔离 B 的 Case2009、Case2017，合格数量 A20/B18/G20。当前发现阶段校验文件集合和影像 SHA256，不打开标注；原始影像永不修改。MRI 不进入本版。

默认 `ct_holdG`：

| 用途 | A | B | G | 标签是否被 B0 使用 |
|---|---:|---:|---:|---|
| fit／训练 | 4 | 4 | 0 | 是，梯度与指纹拟合 |
| cal／来源验证 | 2 | 2 | 0 | 是，逐 epoch 验证与检查点选择 |
| selector／研究预留 | 2 | 2 | 0 | 否 |
| U／标签隐藏 | 12 | 10 | 0 | 否，影像也未进入 B0 优化 |
| target／中心留出评价 | 0 | 0 | 20 | 仅正式模型冻结后计算指标 |

B0 实际读取 12 例来源标注。研究后续使用 selector 时，来源标注总预算是 16 例。不能称为“仅 8 例标注的完整方法”，也不能把本版称为半监督方法。

每个中心按 `SHA256("20260919:病例编号")` 排序：前 2 例 selector，接着 2 例 cal，再 4 例 fit，其余 U。影像摘要重复或已知患者编号重复时拒绝划分。当前没有患者映射，图像不重复不等于患者独立；本版是探索性中心留出。取得患者关联后须重新审计。

另支持 `ct_holdA`（target20、U22）与 `ct_holdB`（target18、U24），每折仍为 fit8/cal4/selector4。Dataset ID 分别 811/812/813，和旧工程隔离。默认仅 G；`all` 是三个单独训练的中心留出模型，不是 3 折模型融合。目标中心属于发布训练队列，不声称新增外部队列。

## 预处理

确实有预处理，入口为 `05_src/server_prepare.py`，调用固定版本 nnU-Net 的官方 API：

1. 校验 3D 形状、有限且可逆的 affine、影像与标签空间一致性。
2. 将官方标签 `0/205/420/500/550/600/820/850` 转为 `0..7`，依次为背景、Myo、LA、LV、RA、RV、AO、PA。出现 421、非整数、未知类别即停止。写出后检查往返映射。
3. 只将 fit+cal 共 12 例复制进该折 `imagesTr/labelsTr`。selector、U、target 不进入训练原始目录或预处理缓存。
4. **指纹和规划仅使用 8 例 fit**：获取前景强度统计、间距与形状，自动确定 spacing、patch、batch size、网络层数及各层通道。规划显存预算默认 5 GiB；它是网络规划目标，不是显存用量保证。
5. 使用官方非零区域裁剪、CT 强度归一化与重采样；没有手工指定心脏框、统一 1 mm 或固定 128³ patch。具体结果写入 `WholeheartBaselinePlans.json`。
6. 逐病例生成 Blosc2 影像／标签缓存与属性文件；完成标记放在缓存目录外。中断后继续未完成病例，不先删除完整缓存。

CT 强度处理可表示为

\[
x'=\frac{\operatorname{clip}(x,q_{0.005},q_{0.995})-\mu_{fit}}{\max(\sigma_{fit},10^{-8})}.
\]

阈值、均值和标准差由 fit 前景采样得到；cal 与目标数据不重新拟合。重采样遵循规划中的官方图像/分割插值方式，分割按类别处理，不能直接把整数类别当连续灰度插值。原始方向由 NibabelIO 的读写映射保存，推理后恢复原始 shape 与 affine。

运行清单、来源标签摘要、原始路径只写入独立结果目录。预处理、续跑检查来源标签摘要；任何来源图像、标签、协议、规划或关键实现不匹配时拒绝旧缓存。旧工程的全标注缓存不能迁入。

## 训练方法与公式

网络是 nnU-Net v2.8.1 `3d_fullres` 规划出的 U-Net，从随机初始化训练。自定义训练器仅固定数据划分、预算、有限损失检查和原子检查点；沿用官方结构构建、增强、损失与优化器。

对类别概率 \(p_{vc}\) 和 one-hot 标签 \(y_{vc}\)，优化目标为多尺度交叉熵加前景 soft Dice loss：

\[
L=\sum_s w_s\left[L_{CE}^{(s)}-\frac1{7}\sum_{c=1}^{7}
\frac{2\sum_v p_{vc}^{(s)}y_{vc}^{(s)}+\epsilon}
{\sum_v p_{vc}^{(s)}+\sum_v y_{vc}^{(s)}+\epsilon}\right].
\]

这里 \(L_{CE}=-\frac1N\sum_v\log p_{v,y_v}\)，背景参与 CE、不参与 Dice；\(\epsilon=10^{-5}\)。实现采用负 Dice，与 `1-Dice` 相差常数。Dice 是否跨 batch 聚合由官方规划的 `batch_dice` 决定。深监督按 \(2^{-s}\) 归一化赋权，单卡非编译模式最低分辨率输出权重为 0。

| 设置 | 本版值 |
|---|---|
| 优化器 | SGD，momentum 0.99，Nesterov，weight decay 3×10⁻⁵ |
| 初始学习率 | 0.01 |
| 调度 | \(\eta_e=0.01(1-e/250)^{0.9}\) |
| 训练预算 | 250 epochs × 250 train iterations；每 epoch 50 validation iterations |
| 随机种子 | seed 0；不宣称跨硬件完全逐位确定性 |
| 增强 | nnU-Net 官方空间、镜像、强度、噪声等增强；没有 B1 自定义外观增强 |
| 模型选择 | 来源 cal 的 patch 前景 EMA Dice 选择 `checkpoint_best.pth` |
| 续跑 | `checkpoint_final.pth` 或最新 `checkpoint_latest.pth`；不拿 best 回滚训练进度 |

**这是固定预算 nnU-Net 基线，不是默认 1000 epochs 的完整配方。** 正式对比应对所有方法保持数据、预算和来源验证协议一致，后续可另设 1000-epoch 敏感性实验，不能覆盖本版结果。每轮原子保存 checkpoint；中断恢复最近完整轮，不能恢复尚未落盘的半轮。长训练没有在本次交付中启动。

## 测试、推理与评价

这里区分软件测试、12 步工程短测和正式中心留出评价。工程短测只用 fit/cal，验证梯度、验证步、权重与优化器恢复、恢复后再训练一步及原图预测，短测 Dice 不能用于报告准确率。

正式训练全部完成后，冻结所选折／B0、best 和 final 的摘要、训练环境及源代码。target 不参与提前停止、选模型、调参或选择后处理。推理使用步长为 patch 的 0.5 的滑窗、Gaussian 加权及训练允许轴的镜像 TTA，输出恢复原始网格和官方标签编号。固定 raw 输出为主要结果；每类 26 邻域最大连通块 LCC 仅是预先指定的次要诊断，不按目标分数选择采用哪种输出。

每病例、每结构记录：

\[
Dice(P,G)=\frac{2|P\cap G|}{|P|+|G|},\qquad
HD95(P,G)=\max\{Q_{.95}(d(\partial P,\partial G)),Q_{.95}(d(\partial G,\partial P))\}.
\]

边界为 6 邻域腐蚀剩下的边界体素中心，完整 affine 转到物理空间后计算最近距离（mm）。单侧为空：Dice=0、HD/HD95=∞；双侧为空：未定义并报告数量，不能改记为完美分割。先汇总病例内有定义结构，再给逐中心和中心等权 Dice、最差中心；完整分母与缺失数量保留。没有官方血管截断，不与挑战榜单直接混称。仅 seed0，尚无多种子结论。

输出包括原图 NIfTI、病例 JSON、`case_metrics.csv`、`class_metrics.csv`、`summary.json` 和中文报告。评价遇到失败中止派发，已完成病例可按摘要续跑，不吞掉失败病例。

## 模块与接口

| 模块 | 职责 |
|---|---|
| `cohort.py` / `server_data.py` | 协议、运行清单、来源绑定及标签读取边界 |
| `data_pipeline.py` / `server_prepare.py` | 标签转换、几何校验、官方规划和预处理 |
| `project_trainers.py` / `server_train.py` | B0 训练、选模、原子保存和恢复 |
| `evaluation*.py` / `metrics.py` | 模型冻结、滑窗预测、原图空间指标及汇总 |
| `task_panel.py` / `task_runtime.py` | 交互面板、命名历史、快照、锁、续跑和停止 |
| `task_worker.py` / `task_inputs.py` | CPU/GPU 阶段、历史输入独立复制和缓存兼容 |
| `task_queries.py` / `task_update.py` | 无训练依赖的只读查询、私有仓库认证更新 |

入口与服务器命令见 [服务器面板](SERVER_PANEL.md)，实际验证范围见 [验证记录](BASELINE_VALIDATION.md)。后续接入局部专家时新增模型注册和独立协议，不能静默改变现有 B0 的训练数据或冒充本版已实现。

依据：实际检查本机 nnunetv2 2.8.1 的训练器、损失和预处理源码；官方软件说明见 [nnU-Net](https://github.com/MIC-DKFZ/nnUNet)。安装器版本与摘要核对 [Miniforge 26.7.2-0 官方发布](https://github.com/conda-forge/miniforge/releases/tag/26.7.2-0)。
