# CHD V2 2.2.2：统一流程面板

在原 `chd-ct-reproduction` 目录更新 main，运行 `bash start_server_v2.sh`。先选择 **预处理、训练、预测等步骤**，再选择 **STU-Net-S / 原六路**，始终返回同一个主菜单。V1 仍用 `bash start_server.sh`，原有文件不改。

[同项目入口](../V2_README.md) · [本次核查与模板矩阵](docs/V2_2.2.2_AUDIT.md) · [科学配方复核](docs/STUNET_SCIENTIFIC_AUDIT.md)

## 正常使用

1. `e` 选择方法，准备其独立环境。
2. `1` 选择方法，预处理默认数据目录。
3. `2` 选择方法，选择已完成的该方法预处理任务，启动训练。
4. `3` 选择方法，选择已完成的该方法模型，执行外折预测和评价。

默认路径来自各方法的 `configs/servers/zmic44.json`，正常操作不再输入路径。STU-Net 首次预处理会核验原始文件对、作者间距表、标签覆盖和重复影像，自动生成独立的私有 `imagechd_input.json`；再次使用核对冻结清单，不覆盖原始数据或已有描述。源文件变化须另建输入并核查。三折沿用原实验 seed42，缺七结构标注的病例隔离；病例编号尚不能证明患者级独立。

| 内容 | 服务器默认位置 |
|---|---|
| 原始 ImageCHD | `/data_nas/zhangruichen/baidu_import/ImageCHD_dataset` |
| STU 私有输入描述 | `/data_nas/zhangruichen/chd_ct_v2_stunet_inputs` |
| STU 环境/登记/作者权重 | `/data5/zhougaowei/zhangruichen_workspace/chd_ct_v2_stunet_runtime` |
| STU 任务结果/预处理 | `/data_nas/zhangruichen/chd_ct_v2_stunet_results` |
| 原六路 V2 环境/登记 | `/data5/zhougaowei/zhangruichen_workspace/chd_ct_v2_runtime` |
| 原六路 V2 结果/缓存 | `/data_nas/zhangruichen/chd_ct_v2_results`、`chd_ct_v2_prepared` |

STU S1 首次训练会在自己的 runtime 下载并核验固定作者权重；已有有效权重复用。直连失败按该 runtime 的 `c` 网络配置重试一次，下载失败或 SHA256 不符立即停止，不改为随机训练。也可显式设置 `IMAGECHD_STUNET_WEIGHTS`。环境、菜单和查询不会下载权重或启动训练。

## 方法与边界

- **STU-Net-S**：S1 作者预训练微调；S0 同架构随机对照。0背景、1 LV、2 RV、3 LA、4 RA、5 MYO、6 AO、7 PA；额外编号作为 ignore8，输出仍为8通道。作者间距派生副本、单例 Z-score、训练子集拟合规划、3D patch 和全体积滑窗。正式1000轮、S1 lr.001/S0 lr.01、默认固定全局 batch4、末轮主评价。
- **原六路**：保持 CT0.24 六网络/三折/预测ROI/投票与验证后处理合同。此次修正自动查找作者间距表的文件名，新预处理会正确识别 `imagechd_dataset_image_info.xlsx`；旧缓存及旧任务不改写，其原有间距解释也不会自动变为新解释。
- `4 全病例预测 / 5 诊断 / 6 重算评估` 当前属于原六路；STU 的 `3` 同时执行七结构外折预测与评价。未实现的新任务预测或诊断功能不会伪装为已支持。

任务、缓存、环境和结果按方法隔离；V1 缓存/模型不能接管。历史任务仍使用原冻结代码恢复。已审计的 **2.2.1 STU正式1000轮模型→2.2.2新预测** 精确代码映射保留训练签名；未知版本、训练语义/数据/计划/环境改变仍拒绝。旧250轮模型不能冒充当前正式1000轮结果。

`u/w` 统一维护整个 chd-ct-reproduction 项目的 main（兼容原V2分支），不拆两个项目。同一检出目录的更新顺序执行。`v` 单独打包获准产物，STU产物使用 `chd-v2-stunet-artifacts/<package_id>` 分支；原始影像、缓存、环境和凭据不进代码分支。

## 命令行与验证

```bash
bash start_server_v2.sh --help
bash start_server_v2.sh --engine stunet --stage prepare --server-defaults --non-interactive
bash start_server_v2.sh --engine stunet --stage train --input-job /completed_STU_prepare_job --models S1 --global-batch 4 --non-interactive
bash start_server_v2.sh --engine stunet --stage predict --input-job /completed_STU_train_job --models S1 --non-interactive
python -B v2/scripts/verify_v2.py
```

保留明确 `--engine` 的兼容接口；正常交互入口是统一面板。STU资源底座为skill VISTA0.7.0，并采用0.7.1精确预测兼容规则；原六路采用0.5.1工程适配。模型不是VISTA foundation，不能声称两路工程均完整等同0.7.1。测试、真实本机小规模验证及尚未完成的服务器Linux/CUDA/NCCL验收见核查报告。
