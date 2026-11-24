# 服务器代码更新与既有输入复用

源码对应GitHub的`aaoca-cta-reproduction`子目录。更新只替换本项目受管理的源码文件，保留既有data、course中的运行配置、预处理、模型检查点、结果、日志和其他项目文件。共享Git的HEAD和索引会随版本更新；更新锁、网络设置及course/updates/latest.json沿用既有管理流程，不迁移目录。

## 既有材料与少量私有文件

用户提供的现状为“除Excel外，服务器似乎已经具备其他材料”。默认复用现有CT、分割/定位标签、作者权重、预处理和Linux环境；本轮没有登录服务器独立核实。实际数据根及权重位置沿用服务器已有私有配置，不要求全量扫描、文件摘要、重新预处理或批量迁移数据。

本批集中目录为`private/datasets/<batch>/`，包含两个任务的相对路径清单、配置、主登记与说明；与诊断工作簿一次传到本项目`course/inputs/`。不另产Excel副本、CSV或压缩包。Windows配置记录本地组织，服务器使用bind-dataset读取已有服务器配置，生成Linux路径配置；不覆盖已有服务器配置，不将旧20人清单作为新批次输入。

先心助理复用原表及逐条证据交付英文data/evidence/definitions工作簿，生产直接读取data，无需另存CSV。配置diagnoses指向该原件，文书研究标签使用document_explicit，组合明确用户阴性确认时使用research_sources；原confirmed接口继续兼容。逐任务明确标签与含糊、冲突、未知及CT未匹配分别处理，不填造医生确认。代码更新和已有权重探索预测独立于正式训练条件，详见`CLASSIFICATION.md`。

## 代码发布与更新

update.py自带完整更新核心，仅依赖Python标准库和Git，读取main分支，没有`--branch`参数。更新命令不需要相邻CHD目录或`--shared-updater`。根目录许可清单包含`install_classification_dependencies.py`；首次引导可在内存执行目标提交中的独立update.py。

`CODE`、`COURSE`、`PY`沿用现有项目路径，`AAOCA_RELEASE_SHA`须设置为从官方main核实的完整40位发布提交，后续安装始终固定该SHA。未提供真实SHA时命令立即停止。更新时本项目不得有正在执行的计算任务。

```bash
set -euo pipefail
: "${AAOCA_RELEASE_SHA:?先完成审核发布并设置真实的完整提交SHA}"
curl -fsSL --connect-timeout 10 --max-time 60 \
  "https://raw.githubusercontent.com/tianwuasd/project/$AAOCA_RELEASE_SHA/aaoca-cta-reproduction/update.py" |
python3 -B - --repo "$(dirname "$CODE")" --runtime "$COURSE" \
  --force --mirror auto --sha "$AAOCA_RELEASE_SHA"
"$PY" -B "$CODE/run.py" --code-info
git -C "$(dirname "$CODE")" ls-remote https://github.com/tianwuasd/project.git refs/heads/main
```

更新器核验本项目受管理源码与目标Git提交；输出的安装SHA、实际入口`--code-info`及更新后官方main查询结果须一致，入口不得有本项目源码修改。共享HEAD或成功提示不能单独证明实际运行版本一致；没有服务器返回结果时，状态为服务器待核验。

独立入口已安装后，日常更新只需`python3 -B "$CODE/update.py" --force --mirror auto --sha "$AAOCA_RELEASE_SHA"`，默认从入口位置推导代码根和既有course位置；自定义部署继续显式提供`--repo`及`--runtime`。

## 更新后运行

PY须为现有可运行解释器，不能仅凭目录名判断依赖已具备。绑定及XLSX读取需要现成psutil、NumPy、SciPy、nibabel、SimpleITK和openpyxl；当前CLI会导入定位入口。只对这些实际入口依赖做一次真实导入，不安装或扫描整个环境；模型依赖在实际模型任务前另行核对。

```bash
"$PY" -B -c 'import psutil, numpy, scipy, nibabel, SimpleITK, openpyxl; print("binding imports passed")'
```

设置BATCH为已集中传入的正式目录名，BASE指向已有真实服务器配置。新批次通过已有配置中的实际数据根绑定，两份Linux配置自动生成：

```bash
BASE="$COURSE/inputs/project.server.json"
INBOX="$COURSE/inputs/$BATCH"
BOUND="$COURSE/datasets/$BATCH"
"$PY" -B "$CODE/run.py" bind-dataset --config "$BASE" \
  --inputs "$INBOX" --diagnoses "$COURSE/inputs/AAOCA_diagnoses.xlsx" \
  --runtime-root "$COURSE" --data-root "$DATA" --output "$BOUND"
"$PY" -B "$CODE/run.py" check-training --config "$BOUND/project.aaoca_present.server.json"
"$PY" -B "$CODE/run.py" check-training --config "$BOUND/project.origin_left.server.json"
```

bind-dataset可加`--check`进行零文件写入检查。正式绑定仅写5个文件（两个任务清单、两份Linux配置、一份binding.json回执），限制5MiB；同一内容复用，目标已有不同内容时停止。它读取文件路径、大小、stat与诊断来源，不读取像素、不验证内容摘要、不安装或运行模型，不复制影像、权重或预处理。现有prepared_inputs显式引用继续保留，其他兼容预处理由已有data_root定位；缺少匹配回执时不会自动重做预处理。

DATA须为本项目长期模型预处理目录。`--runtime-root`和`--data-root`仅决定新生成任务配置，处理旧BASE未显式提供course_root/data_root的情况，不修改BASE。`--aaoca-root`/`--imagechd-root`仅在已核实原配置源根不适用时显式指定，绑定仍核对实际文件路径与大小。无现有BASE时可明确以公开configs/zmic.example.json为模板，传入真实运行根、数据输出根及两套实际原始影像根完成首次绑定；示例规划路径不自动作为真实来源。

本批清单保持已经组织的患者角色及逐例用户确认来源；不重新audit、重置分组或覆盖旧manifest。check-training.ready仅表示对应任务训练/验证的标签与CT绑定条件。classification保留BASE已有的公共参数（例如device、batch_size、epochs、学习率配方、seed和资源预算），更新task与预测候选编号；任务改变时移除strategy、pretrained_detection、weights，避免将另一个任务的权重或初始化方式传入。相同任务保留这些声明时核对任务适用性，真实权重文件仍由运行入口严格加载。旧run_directory和prediction_directory均不带入新批次，须显式选择新任务输出目录。保留/移除字段列入绑定结果和回执；源配置不修改。完整训练/预测条件及方向、覆盖、定位几何、权重来源仍需单独确认，新增计算遵守具体范围。

两例三任务单成员CPU预测及零更新检查点同进程保存加载已验收；有更新的训练恢复、CUDA、五成员集成和诊断性能尚未验收，见`CAPABILITIES.md`。

## 独立分类预处理

`prepare-dataset`串行调用现有prepare_classification，使用一个清单的全部病例。检测清单覆盖当前两任务的病例并集时只运行一次；两份绑定配置使用共同DATA，相同病例、源和方法的结果共用，不按任务或fold重复生成。

```bash
CFG="$BOUND/project.aaoca_present.server.json"
"$PY" -B "$CODE/run.py" prepare-dataset --config "$CFG" --check
nohup "$PY" -u -B "$CODE/run.py" prepare-dataset --config "$CFG" \
  >> "$COURSE/logs/classification-preprocessing.log" 2>&1 </dev/null &
tail -f "$COURSE/logs/classification-preprocessing.log"
```

实际执行会产生正式预处理及进度，须具备对应计算与产物许可；--check仅计算清单与容量，不读数组或创建目录。每例保存normalized.nii.gz、training_xyz.npy、classification.json。进度位于对应任务course_root/preprocessing/progress.json，记录当前病例、完成/复用/剩余、耗时、进程出生信息及在做病例身份。资源事件沿用原入口100GB/整机80%保护，RAM峰值与运行速度仍须实测。

停止原进程后重新运行同一命令即可核对并复用完成病例；共享输出根下的.batch.lock防止不同任务对共同病例重复启动。未完成目录只有在进度记录证明同一病例、源/方法身份与完整路径且实际源stat未变时才清理本批已知文件再重做，原完成回执和未知文件保持保护；Windows遗留标签入口还须samefile确认。未知归属、未知文件、源变化或旧进程仍在时停止并报告完整路径，先人工核对，不能删除完整classification.json来强行重做。不要同时启动同一数据目录的批次和单例生产者。清理与接续实际写出范围须以真实短测记录为准。

STU-Net fold0离线图像是百分位裁剪/均值标准差归一化、nnU-Net空间重采样结果，不能直接当作作者分类输入；裁剪外区域和截断HU不可精确恢复。可以继续使用原CT、原LV/AO标签、间距及来源记录。本工程仍从原HU定位80×80×60mm ROI，按源索引偏移，经TorchIO0.19.6 B-spline到XYZ215×215×85，再做HU[-1024,1024]的256级量化；网络BCZXY，训练float16存储、推理float32。拒绝直接使用STU图像的依据是强度和几何处理不同，不以fold名称或缺回执本身代替方法判断。

公开测试入口使用显式AAOCA_TEST_ROOT指定已批准工作目录；依赖安装辅助入口通过--work-root指定目录。仅语法、帮助及导入检查不会运行这些测试或安装；真实病例计算仍需具体许可。
