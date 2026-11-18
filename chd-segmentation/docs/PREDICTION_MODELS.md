# TotalSegmentator 与 MedSAM3 预测

本轮两个模型只开放预测。原Q、MedSAM2、STU命令保持可用。普通正式配置由现有代码生成到course/chd-segmentation/inputs，无需上传新增配置包。

## 官方来源及环境

TotalSegmentator固定PyPI 2.18.0；官方源码[版本v2.18.0](https://github.com/wasserth/TotalSegmentator/tree/v2.18.0)。仅heartchambers_highres，fast=false。包代码Apache-2.0；心腔权重使用官方科研许可证，需用户自行取得许可。TOTALSEG_HOME_DIR配置集中权重和许可证位置；同时准备主模型301和3mm裁剪297。官方totalseg_download_weights -t heartchambers_highres只下载301；297须额外通过官方download_pretrained_weights(297)取得。运行代码不自动下载任何资源。

MedSAM3仅使用[Liu等论文v2](https://arxiv.org/abs/2511.19046v2)及[作者源码84f43118b5b67b50abad81d7c946681ebb97dcda](https://github.com/Joey-S-Liu/MedSAM3/tree/84f43118b5b67b50abad81d7c946681ebb97dcda)。使用该提交内sam3及lora_layers，避免导入不同同名包。[医学LoRA v1](https://huggingface.co/lal-Joey/MedSAM3_v1/tree/f579e37e4db9f2c7e85a133b7970d59907e36fec) best_lora_weights.pt约74.4MB；[facebook/sam3](https://huggingface.co/facebook/sam3/tree/main) sam3.pt约3.45GB，需要Meta授权访问，遵守[SAM许可](https://github.com/facebookresearch/sam3/blob/main/LICENSE)。作者源码含1,356,917字节BPE资源，配置指定本地路径。基础模型和LoRA均实际加载state，基础/LoRA缺失参数直接报错；无自动HF联网。此入口没有训练或optimizer恢复功能。

两个后端采用独立解释器环境，建议分别置于course/chd-segmentation/env/totalsegmentator和env/medsam3。主3.11环境保持原状。项目依赖声明用于STU；独立后端安装本项目时使用--no-deps，分别读取configs/models/requirements-totalsegmentator.txt或requirements-medsam3.txt。Med还需安装作者固定源码及其requirements，包含torch>=2.7、torchvision>=0.19、训练数据collator与transforms所需依赖；以安装解析结果及真实短流程确认兼容性，不宣称已安装验证。MONAI/nnU-Net计划依赖在实际调用旧训练/预处理入口时加载，独立Med预测没有借此触发STU预处理。两个环境各自直接运行本项目run.py，ResourceSession与模型处于同一解释器进程，公共manager使用真实共享句柄。

## 服务器命令

代码目录：/data4/zhangruichen/code/chd-segmentation。以下命令在获得资源安装/下载及运行许可后执行；当前仅交付源码。

```bash
python -B run.py models configure --model totalsegmentator --prepared /data4/zhangruichen/course/chd-segmentation/inputs/imagechd/prepared.json
python -B run.py models configure --model medsam3 --prepared /data4/zhangruichen/course/chd-segmentation/inputs/imagechd/prepared.json
```

configure使用当前解释器即可，生成inputs/<dataset>/<model>.yaml。使用各模型独立环境的python运行：

```bash
python -B run.py models predict --config /data4/zhangruichen/course/chd-segmentation/inputs/imagechd/totalsegmentator.yaml --prepared /data4/zhangruichen/course/chd-segmentation/inputs/imagechd/prepared.json --case ct_1001
python -B run.py models predict --config /data4/zhangruichen/course/chd-segmentation/inputs/imagechd/medsam3.yaml --prepared /data4/zhangruichen/course/chd-segmentation/inputs/imagechd/prepared.json --case ct_1001
```

--output可覆盖配置的output；新模型不接受旧checkpoint、official-weights、label-box或标签提示。--device、--gpu-ids、--wait-seconds沿公共资源接口使用。每任务一张GPU/一个工作进程，支持CPU。独立任务可经公共登记协调，不自动增加卡数。源prepared中训练/验证/测试身份保留；--test-set/--test-group按实际划分选择，不读取预测真值。

## 输入、类别与科学处理

支持显式input_source.path引用既有预处理目录或描述，包括跨fold；只定位当前消费病例与小记录。Total从manifest读取原始HU和空间，STU标准化数组不能当作HU。工作毫米仿射遵从matching_affine_author_spacing或shared_index_excel及manifest.spacing_xyz；结果恢复原始shape/affine。

Total缓存297/301两个nnUNetPredictor，一次任务内各初始化一次。使用v2.18.0的as_closest_canonical、change_spacing、crop_to_mask、undo_canonical、undo_crop、remove_auxiliary_labels、remove_outside_of_mask；297为3mm/order1输入和order0恢复，step=.8；心脏裁剪按官方高层实际20mm addon，301保持原采样并用官方nnU-Net内部预处理，step=.5；均不镜像。10mm remove_outside按官方平均spacing换算。采用官方predict_single_npy_array，XYZ→ZYX及spacing反序，不创建预处理/export子进程和临时NIfTI。固定perform_everything_on_device=false，网络仍在获准设备运行，logits在CPU累加；避免nnU-Net的设备累加失败后重试CPU分支。固定nnUNet_compile=false，避免编译缓存。官方内部其他工具行为以其源码为准，未改写外部依赖。该组合的真实模型及数值等价验证尚待批准。

Med按明确HU窗（默认[-160,240]）逐片转固定线性uint8/RGB，closest-canonical轴位，结果逆方向还原。依作者1008方形Resize、mean/std=.5、逐概念forward、sigmoid检测阈值、box NMS、mask先二值化再双线性恢复。重叠像素取最高实例检测置信度，同分保留dataset_json顺序先到类别。每次仅保留当前概念结果及当前切片分数，GPU不积累整个体积概率。零检出真实记录。兼容原生uint8源的fixed_linear_uint8转换标记、相同CT窗、完整ROI、原shape/affine及ZYX存储时直接复用单例mmap；现有MedSAM2按裁剪体积minmax生成的uint8与固定线性窗不同，读取HU重新转换并记录实际原因。

类别严格按dataset_json名称映射。ImageCHD通常LV/RV/LA/RA/Myo/AO/PA=1..7，CHD1282通常LV/RV/LA/RA/AO/PA/Myo/SVC/IVC/PV=1..10。Total覆盖七类，CHD1282额外SVC/IVC/PV记录uncovered_classes并在真实评价跳过；未知标注继续按unknown_classes跳过。Med为相应7/10解剖概念，无真值提示。

## 资源、保存与验证

直接复用acceleration的ResourceSession/ResourceClient/SafeBoundary。保持整机80%与100GB，单病例头计算动态工作空间并在安全边界replan；Med逐片逐概念、Total阶段边界核对。资源不足等待并记录原因，显式等待时限停止时保留已完成病例。官方一次forward/滑窗调用内部无法在任意算子处抢占，压力在下一个共同安全边界处理。程序错误直接抛出。

CaseCommit维护原子NIfTI及predictions.json；writing阶段仅清理本病例pending并重算，ready阶段补交。相同固定设置下可以扩展当前--case子集；已完成病例通过收据和存在性跳过，无模型初始化；未完成病例整例重算，无切片断点。

零文件检查：python -B tests/check_prediction_models.py；可附--prepared读取最多两例头和几何，不读取CT/标签体素。真实短流程测试源码同文件的--run-real --config ... --prepared ... --case ... --output ...，限定1–2例、0更新，真实运行一次后再次调用检查完成病例跳过。尚无新运行授权，不能自动执行该模式。尚未测量真实推理耗时、峰值RAM/VRAM或分割效果；静态检查不证明这些性能。
