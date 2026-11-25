# Dataset1282 留出组预测

预测使用冻结名单的 `source_test`、`target_primary`（iCT 256）、`target_secondary`（Brilliance iCT）。`included` 与 `index_scan` 必须同时为真，每位患者只使用一例合格索引扫描。训练与 development 患者不能进入预测名单。设备来源沿用 Excel；医院中心身份需要独立的来源资料支持。

默认选择全部三组。`--groups source_test` 可选择其中一组，重复 `--case CASE_ID` 可选择该组内的指定病例。正式效果报告应包含完整冻结分母，并明确记录未成功预测的病例。

## 模型与预处理

现有 HU-marginal 输出包含训练与 development。留出 CT 从原 manifest 和 Excel 读取，应用同一训练 `nnUNetPlans.json` 的间距、转置、裁剪与 CTNormalization。该步骤调用 nnU-Net 官方预处理器，标签参数为空，不重新提取训练指纹或拟合参数。原图按已确认的 HU 假设处理。

`--training` 指向完整训练任务目录；所用阶段必须具有完成收据，权重使用 development 选择的 `best.pt`。保留既定预处理数据和 20% 框设置；模型和来源记录复制到独立输出的 `source/`。正在运行且未完成的血池阶段不能提供完整方法预测。

兼容既有 Xu 训练结果和拆分保存的血池最佳组件。预测不要求包版本文字、环境路径及源码摘要与历史记录相同，也不重复扫描输入和模型文件的完整摘要。PyTorch 按模型结构严格加载参数；依赖缺失、结构不兼容或读取失败在实际调用处报告。原训练记录、收据和检查点保持只读。详见 [预测运行与续跑](prediction-runtime.md)。

定位模型在留出 CT 上重新预测，严格投票后将框的各边总长度增加 20%，两侧分摊。空框、整图及超过现有 95% 体积限制的框记录为 `roi_rejected`。后续复用既有四路/六路投票、血池修正和血管处理。预测过程不读取留出标签。近端初始化分支沿用训练任务实际完成或因监督不足跳过的记录。

模式：`four-way` 使用四个三维模型；`blood` 加入已训练血池模型和可用近端分支；默认 `vessels` 继续执行血管处理。各模式只加载其所需且已完成的权重。

## 执行

使用已有研究 Python，需具备 PyTorch、nnU-Net、影像处理依赖及读取 Excel 的 openpyxl。预测直接读取指定训练计划，不比较安装包的精确版本文字。

下列变量分别指向现有预处理、训练结果、原本地配置、原预处理政策，以及新的预测输出目录。配置和政策使用创建训练预处理时的同一份私有文件。

```bash
"$PY" -B -u scripts/xu_predict.py run \
  --prepared "$D" --training "$O" --config "$CFG" --policy "$POLICY" \
  --output "$PRED" --devices all-visible --threads 4
```

默认排除物理 0、1 号 GPU。管理进程与准备阶段不开放 CUDA；预测子进程只开放一张获准 GPU。CPU 预取使用一个 worker、深度一，内存预算由新鲜资源检查决定。有外部任务时按实时剩余显存判断是否准入；资源不足时释放本任务子进程并等待，重新准入后复用已提交的病例和分支结果。单批仍放不下时保持等待，直到资源恢复。

本地 CPU 验证可用 `--device cpu`，技术数据必须明确指定 `--technical-subset`。恢复使用原参数和输出目录并添加 `--resume`，直接复用保存的名单、模型快照和逐病例完成记录。允许更新执行代码；更换训练目录、病例名单或预测模式需要新建实验。

```bash
"$PY" scripts/xu_predict.py status --output "$PRED"
tail -F "$PRED"/logs/predict-*.log
"$PY" scripts/xu_stop.py --output "$PRED"
```

`status.json` 记录运行、等待或终止状态；`progress.json` 包含阶段、病例进度、耗时及当前阶段预计剩余时间；`resources/` 保存预算、进程可见 GPU 和编号映射，`reallocations/` 保留重新分配原因。

## 输出和独立评价

- `source/`：冻结权重与原训练身份。
- `prepared/CASE/`：只含 CT 的测试预处理缓存、几何和完成摘要。
- `branches/`：各模型逐例预测及完成摘要，供中断恢复复用。
- `cases/CASE.nii.gz`：原 CT 数组尺寸上的类别分割，使用 Excel 毫米间距。
- `cases/CASE.json`：成功或 ROI 拒绝状态、质量信息及摘要。
- `complete.json`：全部所选病例的处理结果；其中 `rejected` 可大于零，必须保留在报告中。

坐标约定为 `local_array_excel_mm`，保留原数组轴顺序，不能据此推断未经提供的原始患者方位。

全部病例预测提交后，单独执行：

```bash
"$PY" scripts/xu_predict.py evaluate --output "$PRED"
```

此时才读取参考标签，按既有占位头约定使用数组对应关系。`evaluation.json` 按冻结组及设备分别统计 Dice、HD95、ASD、有效样本数与 ROI 拒绝数。缺标结构不计作真实背景；完整标注病例另列汇总。拒绝病例不参与成功预测病例的指标均值，其数量单独列出，不能把成功病例均值解释为全部入组患者结果。

## 本地技术验收

```bash
PYTHONPATH=src:tests "$PY" -B -m unittest test_prediction_selection test_prediction_preparation -v
"$PY" -B scripts/prediction_acceptance.py --training "$TECHNICAL_TRAINING" --output "$CHECK"
```

验收脚本接收已经完成的合成技术训练目录，分别运行官方预处理、三组选择、旧产物读取、CPU 子进程预测和恢复检查。正式留出病例保持未打开状态。`--models` 可传入既有四分支短测目录，补充真实已训练权重的原尺寸导出和独立评分检查。验收日志逐项保存，全部检查通过后才生成 `acceptance.json`。技术短测不能用于推断正式模型精度或服务器 GPU 性能。
