# STU-Net-S 整幅预测

预测入口为 `run.py predict`，短测入口为 `smoke.py --stage predict`。复用训练环境、旧微调检查点和对应折的预处理计划。旧训练目录保持只读；每次预测独立保存代码、模型副本、任务参数和来源摘要。

## 服务器运行

在 `chd-ct-panel-v2` 目录使用已经安装训练依赖的解释器。下面的 `py` 使用此前服务器环境。

```bash
py=.venv_preprocess/bin/python

# 只核对来源及测试病例，无 GPU 容量测试。
"$py" run.py predict --fold 0 --check-only

# 两例完整体积短测；沿用实际训练 patch，不缩小图像。
"$py" smoke.py --stage predict --fold 0 --background

# 对第 0 折原留出测试病例进行正式预测。
"$py" run.py predict --fold 0 --background

# 每例只使用其对应 test 折，汇总所找到各折的独立测试结果。
"$py" run.py predict --fold all --background
```

默认预处理路径：`/data_nas/zhangruichen/chd_ct_v2_stunet_results/preprocessing/stunet-s-partial/server_run_v2_20261104`。

默认训练根目录：`/data_nas/zhangruichen/chd_ct_v2_stunet_results/training`。自动发现同折同实验中进度最大的来源。存在不同实验或同进度候选时明确报告路径，使用 `--training-run /完整训练目录` 指定。`--training-run` 与 `--fold` 分别用于直接指定来源和自动选择，不能同时使用。

默认预测输出：`/data_nas/zhangruichen/chd_ct_v2_stunet_results/prediction/{smoke,formal}/时间`。启动成功会打印实际 `output` 和 `log`。用 `--output` 固定目录时，重复运行须加 `--resume`。

默认选 `best`；可显式选 `--checkpoint epoch` 或 `latest`。来源训练必须完成；安全停止的中间模型需加 `--allow-intermediate`，结果保留实际训练状态。短测权重只能用于 `--smoke` 预测。新预测会复制固定代次权重，运行期间不跟随来源 best 指针改变。

## 日志、后台和恢复

```bash
# 将此变量改为启动时打印的实际目录。
pred_run=/data_nas/zhangruichen/chd_ct_v2_stunet_results/prediction/formal/实际时间

tail -n 80 -F "$pred_run/launcher.log"
"$py" run.py status --output "$pred_run"

# 请求停止本次预测；其他用户进程不受影响。
"$py" run.py predict --stop --output "$pred_run"

# 停止完成后恢复，使用此任务已冻结的模型和配置。
"$py" run.py predict --resume --output "$pred_run" --background
```

`--background` 将监督进程与终端会话分开，关闭日志查看终端不会发送训练/预测停止请求。服务器重启、作业管理器结束整个任务或系统 OOM 仍会终止进程。恢复按病例完成标记及文件 SHA256 复用，未提交病例重新计算。旧预测使用冻结代码；更新代码后需要新建预测目录使用新版实现。

工作进程每2秒核对监督进程的 PID 和出生时间；监督进程意外退出后，工作进程自行释放资源。恢复命令还会核对遗留进程的 PID、出生时间、解释器和任务目录，再清理本任务遗留进程；记录保存在 `control/orphan-cleanup.json`。无法确认归属时直接报告，不终止其他进程。

`launcher.log` 与 `logs/run.log` 每约 10 秒给出病例、阶段和窗口数量。`status.json` 包含实时资源和各 worker 状态。GPU 被占用时显示外部 PID、可用内存和病例内存预估。默认持续等待；`--wait-seconds 300` 可限制资源等待时间。未知数据或模型错误立即记录 `failure.json` 并终止。

每个 worker 位于 `control/gpu-*/` 或 `control/export-*/`。其中 `worker.log`、`status.json`、`admission.json` 和 `capacity-fold-*.json` 记录设备、显存余量及真实批量探测。每例 `result.json` 记录准备、计算/传输、融合、导出时间与峰值显存。

## 预测与评价

输入复用原训练的 spacing、CT 归一化统计、转置、裁剪及 patch。缓存命中需要完成记录、文件摘要和来源计划全部一致；没有预处理缓存时，对原 CT 做图像独立的同方法预处理。标签只在结果导出之后用于评分。

采用步长 0.5 的重叠滑窗和 Gaussian 加权。TTA 默认开启，使用三轴镜像；`--no-tta` 明确关闭。计算采用 CUDA 混合精度，整幅 logits 在 CPU 以 FP32 累积。输出按原病例属性执行 nnU-Net 官方概率插值与裁剪还原，再恢复 NIfTI 轴顺序。空间尺寸、物理 spacing 和标签范围均检查。

每例产物保存于 `cases/fold_N/病例/`：

| 文件 | 内容 |
| --- | --- |
| `病例.nii.gz` | 0背景；1 LV、2 RV、3 LA、4 RA、5 MYO、6 AO、7 PA |
| `病例_reference.nii.gz` | 原头信息与审计后物理几何不一致时，提供匹配分割的 CT 展示副本 |
| `result.json` | 几何、模型身份、耗时、七结构 Dice/HD95、体积及文件摘要 |
| `complete.json` | 最后原子写入的完成标记，恢复时核验 |

整体结果在 `cases.csv`、`summary.json`。HD95 使用真实 affine 毫米坐标，定义为两个方向距离第95百分位中的较大值。标签 ignore=8 不参与评分。真实结构存在且预测为空时 Dice=0、HD95=null，单独统计缺预测；双方为空时两项为 null，均值同时报告有效病例数。不会把空预测记为距离零。

原 ImageCHD 划分提供文件对层面的隔离；无法仅凭文件名证明患者独立性。正式结论仍需检查患者标识和来源关系。短测报告验证程序链路，模型性能需使用正式训练权重和完整测试集。

## CPU/GPU 调度

默认 `--max-gpus 2`，各 GPU 独立领取病例并保留模型，不进行 DDP 同步。`--gpus 0,1` 限制为 CUDA 可见编号；设置 `CUDA_VISIBLE_DEVICES` 后按其重新映射的编号解释。默认不使用有外部计算进程的设备。

每张卡预算为 `当前空闲-max(4 GiB,总显存10%)`，可用 `--gpu-memory-gib 16` 再设上限。真实训练 patch 上测试 1/2/4/8 等滑窗批量，要求保留余量，选择达到最佳吞吐 95% 的最小批量。`--max-batch` 限制探测上限；不改变训练 batch 或 patch。24 GiB 卡空闲时一般最多约20 GiB可用，实际值以当次探测日志为准。

主机资源沿用 V2 阶段 CPU80%/RAM80% 与新采集余量；至少保留4 GiB主机内存。已核实 USS 才抵扣已兑现的内存预留。每 GPU 最多预取下一例，同时为计算、预取及独立 CPU 导出计入线程和内存。外部占用、显存不足或主机内存压力允许有限重新分配与缩减并发；已完成病例保持有效。

计算和导出均空闲时释放驻留模型，重新采集 CPU/RAM 阶段预算。资源恢复后可以启动较大的病例。任务线程数同时受当前 CPU 余量和阶段剩余额度限制；资源不足时降低的并发上限保留。致命错误优先报告，完整结果优先确认；资源变化不会触发已完成病例重写。

`--no-prefetch --max-batch 1` 可用于同条件基准。`--device cpu` 明确使用 CPU。单卡本机测试不能替代服务器双卡和 Linux 驱动环境验收。

Windows 本机短测可在用户明确许可后，用 `--smoke --share-desktop-pid <PID>` 与 Codex 桌面共享。此选项仅接受当前 `ChatGPT.exe` 的 PID、出生时间及完整程序路径，持续校验身份，继续保留显存余量；正式任务和 Linux 拒绝该例外。

## 新病例与多折

```bash
"$py" run.py predict \
  --training-run /训练/fold0 /训练/fold1 /训练/fold2 \
  --images /新CT/case001.nii.gz /新CT/case002.nii.gz \
  --background
```

新 CT 要求三维 NIfTI、明确 mm 单位、有效几何，文件名只含英文、数字、下划线或连字符。与来源训练/验证文件摘要相同的输入拒绝作为新病例。每折分别按原统计预测，恢复共同原图网格后平均类别概率，最后 argmax 输出 `ensemble/病例.nii.gz`；此入口无需标签。留出测试集按对应折评价，不对接触过该病例的其他折做集成。

新 CT 的 header spacing 必须与 affine 轴向长度一致；存在冲突时停止并报告，输出保留已核实的原 affine。ImageCHD 使用原队列已有的 spacing 审计证据。

## 兼容边界

`configs/prediction/compatibility.json` 记录已审计训练生产者的精确科学源码摘要。默认校验训练源码摘要是否在名单中。短测和正式预测都可显式传入 `--skip-producer-audit`，仅跳过名单检查。仍逐项核验冻结源码、训练签名、权重文件、网络源码与结构、输入计划和病例角色。任务日志和 `summary.json` 记录是否使用该开关，恢复任务沿用冻结设置。

推理依赖校验包含 PyTorch、nnU-Net、NumPy、SciPy、Blosc2、acvl-utils 和 scikit-image；预测恢复另核对 NiBabel、threadpoolctl。环境变化需要重新核实兼容性。
