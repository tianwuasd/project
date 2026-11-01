# STU-Net-S：官方实现复核与2.2.0修正

2026-11-01。结论：网络结构和作者预训练权重已有独立数值一致性证据；旧适配的训练预算、随机对照学习率及若干工程衔接需要修正。不能把“结构正确”解释为真实七结构效果已验证。

用户选择“按照官方为主”，因此新任务采用作者公开代码配方，同时保留固定全局batch默认4和nnU-Net2.8.1兼容层。旧2.1.0/b31b4e1f的250轮任务保持原冻结源码，不原地改成1000轮任务。

## 论文与固定来源

论文是Ziyan Huang等的 **STU-Net: Scalable and Transferable Medical Image Segmentation Models Empowered by Large-Scale Supervised Pre-training**，2023-04-13。[论文](https://arxiv.org/abs/2304.06716)、[PDF](https://arxiv.org/pdf/2304.06716)、[作者实现](https://github.com/uni-medical/STU-Net)。S是模型系列的小型版本，不是另一篇论文。

论文在TotalSegmentator的1204个CT体积、104结构上进行有监督预训练，再研究下游微调。六级残差网络和最近邻上采样加1×1卷积便于扩展S/B/L/H。论文跨任务结果不能代替ImageCHD儿童先心数据上的实验。

本次固定作者提交`0652eaf5cafda6bc269942238c3695994f2dc4f0`，不跟随未经核验的新提交：

- [small与small_ft训练器](https://github.com/uni-medical/STU-Net/blob/0652eaf5cafda6bc269942238c3695994f2dc4f0/nnUNet-2.2/nnunetv2/training/nnUNetTrainer/STUNetTrainer.py)，文件blob85587aca5fdf1180dc266a138f3ae97c231875b2。
- [作者所附nnU-Net基础训练器](https://github.com/uni-medical/STU-Net/blob/0652eaf5cafda6bc269942238c3695994f2dc4f0/nnUNet-2.2/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py)，文件blobaa51c70066f4688b631ff2d4331c913c96c153e9。
- [官方微调权重载入](https://github.com/uni-medical/STU-Net/blob/0652eaf5cafda6bc269942238c3695994f2dc4f0/nnUNet-2.2/nnunetv2/run/run_finetuning_stunet.py)跳过分割输出头。
- [官方训练入口](https://github.com/uni-medical/STU-Net/blob/0652eaf5cafda6bc269942238c3695994f2dc4f0/nnUNet-2.2/nnunetv2/run/run_training.py)默认末轮验证，显式val_best才切best。

## 已独立验证的结构与权重

四个网络类Decoder、STUNet、BasicResBlock、Upsample_Layer_nearest的AST与官方对应类一致。small配置depth=[1]×6，dims=[16,32,64,128,256,256]；单通道、8类输出时14,548,344参数，官方105类输出约14.6M，差别来自输出头。

作者small_ep4k.model为117497661字节，SHA256为`f440f401bf4cac1d1d6f7c4635542ef057193a5f96c0b3c661a6179afeab7be6`。120个非输出头张量精确加载，10个目标输出头张量重新初始化。缺少或错误权重会停止S1，不静默变成随机训练。

独立构建官方类与项目类，同步状态，在CPU输入1×1×32×32×64：五级输出逐元素最大误差0，全部参数梯度最大误差0且有限。这是实现核对，不能证明分割精度。

## 新旧训练规则

| 项目 | 旧适配2.1.0 | 2.2.0新任务，作者代码优先 |
|---|---|---|
| S1官方预训练微调 | 250轮，lr0.001 | 1000轮，lr0.001 |
| S0随机初始化对照 | 250轮，lr0.001 | 1000轮，lr0.01 |
| 每轮预算 | 250训练、50验证尝试 | 同左；AMP溢出跳步仍计入尝试，不另补样本 |
| 优化器 | 继承nnU-Net | 显式冻结SGD、momentum0.99、Nesterov、weight_decay0.00003、poly指数0.9 |
| batch | 用户固定默认4 | 保留；是明确的用户覆盖项 |
| 主评价检查点 | 来源验证EMA Dice选best | 预先规定末轮final；best仅内部诊断 |
| 任务身份 | 源码/环境/计划/划分摘要 | 同时记录完整方法配方与所选global batch |

论文第5页写骨干学习率为输出头0.1倍、weight_decay1e-3、batch2；作者公开微调类使用统一lr1e-3并继承所附基础训练器。遵从用户选择，以公开代码为主，不拼接两套设置后声称完全复现论文。nnU-Net2.8.1兼容层、固定batch4与ImageCHD数据适配仍是明确差异。

预算由`stunet_recipe.py`统一提供；训练器、控制器守卫、任务计划、资源回执、进度和时间外推一致。旧250轮完整模型不接受为新1000轮配方输入；旧任务仍用自己的冻结代码续跑。预处理兼容性单独按实现摘要核验，兼容时可以复用，不要求重做影像预处理。

## 查明并修复的衔接问题

1. 原控制器硬编码250轮，单独把训练器改1000会被拒绝，且进度和预算显示错误；现统一配方。
2. 固定batch训练签名包含selected_global_batch_size，原冻结评估未包含该字段，会拒绝合法完成模型；现训练与评估共用同一身份构建函数，并校验完整配方和完成轮数。
3. 旧模板短测汇总要求各方法初始权重相同，与S1预训练/S0随机对照相矛盾；现检查记录完整和输入计划相同，允许权重按设计不同。
4. 结果面板仍读取“中心等权”的旧字段，而ImageCHD输出的是病例均值、折均值；现显示真实OOF字段，不把折称为中心。
5. 更新备份期间切换分支可能使选中的提交落到错误分支；现应用前重查，应用后核验分支和提交。
6. 同一检出并发更新可能使任务复制混合版本代码；现核验复制前源码、复制后源码和目标逐文件摘要，失败不登记或启动任务。

每项按可复现路径或实际输出字段核查；本机真实Git故障注入、配方与冻结评估、预训练/随机短测汇总均有回归记录。

## 数据和评价边界

保留七结构LV/RV/LA/RA/MYO/AO/PA，含背景为8输出；额外来源标注映射ignore8，不误当背景。作者Excel间距写入独立派生副本，原像素不变；不直接把源灰度当标准HU。指纹与规划只使用各折训练病例，预处理训练和验证病例，目标病例不参与规划。下游patch由nnU-Net规划，不套用预训练固定128³或1.5mm。

目前固定数据协议是97例可用、13例缺MYO隔离；三折train54/55/55、val各10、target33/32/32，每例恰好一次外折目标。尚无经过独立核验的患者身份映射，不宣称患者级无重叠；作者间距来源、预训练数据重叠等仍需研究记录支持。raw为主结果，逐类LCC仅诊断，不用目标分数选择后处理。

## 验证与实际限制

本轮重新完成真实合成CUDA流程：来源训练子集指纹/规划、预处理、官方权重加载、真实参数更新、检查点权重与128项动量精确恢复、恢复后更新、滑窗预测和CPU输出还原，最终网格48×48×64与原图一致。诊断batch1，正式默认batch4不变；没有执行1000轮真实训练。

运行环境为本机torch2.12.1+cu126、nnunetv2 2.8.1；不是目标Linux固定依赖安装验收。两rank Gloo损失数值验证和目标服务器CUDA/NCCL、正式batch4容量是不同验证层次，不混称。

项目测试入口是`foundation_engine/tests/run_panel_suite.py`，涵盖ImageCHD科学契约和可复用资源/恢复机制。原模板测试以`--legacy-template`保留：原b31b4e1f同环境运行422项已有19失败、37错误、56跳过，含旧病例规则、发布路径/身份、测试mock与平台假设。不是所有用例都过时；仍适用的缓存复制/源码完整性、评估队列、环境不改写和阶段恢复已纳入当前验收。另将真实导出测试改成ImageCHD合成角色与作者间距，继续检查并发和失败后的部分提交。保留旧失败记录，不把它们算作本项目通过证据。原CT六路的回归另在上级tests中执行。

发布测试摘要见[2.2.0交付记录](V2_2.2.0_MAIN.md)。没有新的真实三折LV/PA Dice，不能承诺难例已解决；需要新配方训练与外折评价才能回答。
