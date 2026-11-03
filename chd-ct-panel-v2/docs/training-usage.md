# STU-Net-S训练：完整监督与部分标注

当前实现CPU/单张可见CUDA卡的STU-Net-S微调。入口仍是`run.py`，不需要面板。多进程CPU供数属于加速；本版没有弹性多GPU训练，不能用torchrun多rank启动。

## 环境、作者权重与输入

```bash
python setup_env.py --plan --profile training
python setup_env.py --install --profile training --download-weights
python setup_env.py --check --profile training
```

training使用独立`.venv_training`，默认部署Torch2.8.0 CUDA128；CPU环境显式加`--cpu`。已有可信环境可以在`project.local.json`设置`training_python`。预处理环境不被替换；实际包版本写入训练签名，安装成功不等于已通过GPU正式尺寸容量检查。当前本机验收环境以私有报告实际记录为准，不把它说成新装服务器环境。

作者small_ep4k权重只在显式`--download-weights`时下载，117497661字节、SHA256 `f440f401bf4cac1d1d6f7c4635542ef057193a5f96c0b3c661a6179afeab7be6`；已有错误文件保留并拒绝，不静默随机初始化。可用`--weights`指定已有文件，或保存私有`weights_path`；原始预训练文件在检查点恢复时不再是必需输入。网络代理只作用本次`--proxy`，不改全局设置。

训练读取一份预处理run。默认优先使用私有`training_prepared_root`，其次使用成功prepare保存的`outputs/latest_preprocessing.json`；不会到当前终端目录猜测描述文件。Linux结果根和权重缓存默认沿用既有STU运行根，见`configs/paths.json`。首次没有可用预处理时，先运行prepare，或显式`--prepared`。

## 三个入口

```bash
# 只检查来源、角色、文件、配方；不申请GPU
python run.py train --prepared /预处理run --check-only

# 同一训练执行器的短测：64³、B1、2轮，每轮4训练迭代/1验证迭代
python smoke.py --stage train --prepared /预处理run --output outputs/train-smoke

# 正式默认：128³、B4、1000轮，每轮250训练迭代/50验证迭代
python run.py train --prepared /预处理run --fold 0 --output /训练结果/fold0
```

短测自动选择一例已有缓存的main train、一例aux（若该方法有辅助池）、一例main val；原角色不变，不借用外折test。需要这些角色的预处理缓存确实存在。也可在`--smoke`下明确给`--train-cases`/`--val-cases`。正式训练必须具备本折全部所选训练和验证产物，不能用短测子集伪装正式队列。

可显式选择`--device cpu`、`--workers 0`串行供数或限制CPU供数进程。`--epochs`、`--train-steps`、`--val-steps`、`--batch-size`、`--patch-size`属于显式配方变化，写入签名；不会因显存不足自动降batch/patch。全局batch在本版就是单卡batch。

## 科学配方与结构

官方小模型六级深度均1，通道16/32/64/128/256/256，六级3³卷积核。stride由锁定nnU-Net2.8.1的拓扑规则结合本折spacing和训练patch规划；不会把预处理压缩块提示冒充网络计划。加载作者骨干逐tensor核验，8类输出头重置；不迁移105类源输出头。

默认微调：SGD lr0.001、momentum0.99、Nesterov、weight_decay3e-5、poly指数0.9、梯度裁剪12，1000轮。每轮250训练**迭代**、50验证**迭代**；B4时分别消费1000和200个patch，AMP跳步仍消费一次尝试，不追加样本凑更新数。深监督五级权重为归一化`[1,.5,.25,.125,0]`。

作者v2微调入口依据：[官方STUNetTrainer](https://github.com/uni-medical/STU-Net/blob/main/nnUNet-2.2/nnunetv2/training/nnUNetTrainer/STUNetTrainer.py)。当前nnU-Net接口适配为2.8.1，不称逐字复现作者2.2工程。架构来自Apache-2.0上游，许可见[STUNET_LICENSE.txt](STUNET_LICENSE.txt)；派生文件来源blob记在`_vendor/stunet_architecture.py`头部。

```text
configs/training/stunet-s.json     正式配方与版本
src/chd_stunet/training/
  contracts.py                   科学身份、配方和依赖版本
  data.py                        完成缓存、角色、坐标、原生取块/增强
  model.py / weights.py           官方骨干、拓扑、作者权重校验
  losses.py                      部分监督与深监督
  engine.py                      train/validation状态机
  checkpoints.py                 模型/优化器/scaler/RNG原子检查点
  code_snapshot.py               运行代码固定与恢复核验
  feedback.py / resources.py      进度/内存监督/GPU真实尺寸准入
src/chd_stunet/acceleration/
  training_cpu.py                 有界供数、独立读取、已消费游标
```

## 部分标注监督与评价边界

main train与aux_train的并集按病例均匀抽样，不保证每个batch都含辅助病例。fold0默认池为53主训练＋13辅助，val仍10。源test绝不能进入训练或验证。辅助病例仍保留原六类标签与0的歧义，源标签0用`-log(P0+P5)`监督，其他六类保留真值。

Dice排除背景和ignore；每例先屏蔽未标类别，再汇总到batch的每类统计，MYO只由完整标注病例贡献。全batch都无MYO监督时，MYO不进入Dice类均值。完整且有有效监督时与原生nnU-Net CE+负Dice数值/梯度对照；全ignore的显式政策返回可微0，原生退化输入有-1常数，不能把两者的loss值说成完全相同。

调用原生nnU-Net取框、空间/灰度增强、镜像与深监督下采样。为跨新worker恢复可重放，关闭高斯模糊按墙钟选择FFT/卷积的benchmark，使用固定实现选择；增强参数不因机器改变。

验证只使用main val，记录每类patch Dice/损失、TP/FP/FN和缺失类别null。它不是全体积测试成绩，也不证明LV/PA或整体临床效果。文件对不等于已认证的独立患者，原标注存在也不等于专家逐层审核通过。

## 坐标接口修正与旧缓存

新预处理协议v2将前景坐标保存为`(channel,z,y,x)`四列，这是nnU-Net2.8.1 `get_bbox`的要求。此前0.2/0.3三列缓存只允许匹配两份已知精确生产者SHA时只读补通道0，并检查范围与坐标实际对应标签；记录在`input_adapters.json`。不改旧pkl、影像或标签。未知生产者三列缓存拒绝，而非猜测轴序。

新训练可直接消费已经验收的旧缓存及本轮坐标修正后的新缓存；必须通过文件hash、原冻结角色、本折预处理计划合同、标注元数据和几何数组检查。预处理自身的严格签名仍不允许用新版本热恢复旧运行，应使用旧快照或新输出目录。

## V2资源策略的适配范围

- CPU/RAM80%阶段预算、至少4GiB主机预留；按增强输入、输出队列、元数据和worker导入工作集保守准入。显式workers0也须有实际CPU/内存预算。
- 常驻CPU进程供数，所有native线程限1，派发/预取槽至少覆盖worker；NumPy通过有界pipe传输。读取独立限流，增强在闸门外；没有全局SHM清理。
- 批次种子由seed/阶段/epoch/序号确定。队列peek不算消费，完成训练尝试或验证后才commit；未消费预取重放，worker数变化不改变样本。
- 每秒记录主机/cgroup余量和本任务进程组RSS。持续压力只缩减本组供数，保留安全检查点，最多3次；缩减预算进入恢复状态。极端外部瞬时OOM不能由采样保证拦住。
- CUDA只使用可见设备；核对物理UUID和外部compute PID，证据不可解析则拒绝。保留至少4GiB或总显存10%，并以真实固定B/patch前反向及SGD动量预留做容量确认，不更新或消费probe批次。
- 本版单卡，不包含V2弹性扩卡/NCCL能力或速度提升承诺。CPU供数/异步复制/混合精度与数值训练分别验收。

## 状态、停止和恢复

```bash
python run.py status --output /训练run
python smoke.py --stage train --train-steps 8 --stop-after-steps 2 --output outputs/resume-smoke
python smoke.py --stage train --train-steps 8 --resume --output outputs/resume-smoke
```

`--stop-after-steps N`指到达**总训练尝试数N**后安全暂停，恢复时去掉该参数。Ctrl+C由主进程请求安全停止，供数worker忽略SIGINT，父进程只终止自己创建的进程。不可捕获强杀仅保留此前已提交检查点。

`status.json`每秒反馈阶段、epoch/step、尝试/实际更新和内存；`logs/`有完整事件、病例、取框、学习率、loss、AMP跳步、worker PID和耗时。代码固定在`code/`，供数worker从该快照启动，环境版本启动时复核，目录更新不热换运行算法。

`checkpoints/latest.json`原子指向独立、校验过的完整代次，保留模型、优化器动量、scaler、RNG、阶段和已消费游标；另有best/final指针，最终轮仍可独立使用。最多保留近期两个及best/final引用的代次。恢复前先核验科学身份、数据和代码快照，拒绝时不改变旧状态或日志。完成最后一轮但收尾写出中断时，可以只恢复final/summary元数据，不申请GPU。

阶段结果保存到`summary.json`、`history.json`和`capacity_probe.json`。本机真实短测、服务器正式128³/B4容量、新装环境和全量模型性能分别报告，不互相替代。
