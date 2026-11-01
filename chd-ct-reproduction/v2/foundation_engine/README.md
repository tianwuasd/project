> 当前2.2.2统一流程与默认输入生成见 [上级README](../README.md)。下面的手工冻结输入方式只供已有审计/划分的兼容参考；普通服务器预处理不再要求填写models.json或audit.json。当前标注核查只据标签覆盖隔离，不自动修复异常灰度。

> 本目录现作为原CHD V2 2.2.1的STU-Net方法引擎。统一从上级start_server_v2.sh进入；以下独立工程说明保留输入、权重与科学合同参考。

# ImageCHD：STU-Net-S 七结构微调

独立工程：用官方TotalSegmentator预训练STU-Net-S微调LV、RV、LA、RA、MYO、AO、PA。面板引擎来自skill v2固定的VISTA 0.7.0提交 `21ad9e539cb7dbcc6937b7e09064a7261cd2b913`，原模板212个文件在适配前已核验。本工程没有实现VISTA3D foundation，不覆写旧多路CHD实验。继承的docs和Wholeheart测试只作上游工程参考，数据/模型协议以当前源码和本文件为准。

| 方法 | 初始化 |
|---|---|
| S1（默认） | 官方STU-Net-S编码器/解码器，重新初始化8类输出头 |
| S0 | 同架构随机初始化对照 |

标签：0背景、1 LV、2 RV、3 LA、4 RA、5 MYO、6 AO、7 PA。额外源编号统一为ignore=8，不当作背景监督或计入评价；ignore不是新增解剖类别，网络仍为8通道。13例缺MYO排除；手工冻结描述保留来源QC标记，自动输入只核验标签覆盖/几何，不推断灰度修复。身份尚未经过患者级去重，不能声称患者层面独立。

默认三折：fold0为54训练/10验证/33外折，fold1、2各55/10/32；每例仅由未训练该例的外折模型评价。独立的单例Z-score、训练子集指纹/间距规划、3D patch训练及全体积滑窗预测，不假设未知HU偏移，不把全图压为64³。作者Excel间距只写派生副本，原像素、源文件不改；方向和原点沿用来源，不能额外宣称扫描仪坐标认证。结果还原原图网格及源头信息，距离计算使用作者间距。

## 输入与权重

先从冻结实验的审计和三折生成私有输入目录：

```bash
python tools/build_imagechd_input.py --raw /path/ImageCHD_dataset \
  --models /path/frozen/train/models.json --audit /path/data_audit.json \
  --output /path/private_input_descriptor
```

其中`imagechd_input.json`引用源文件，不复制体积。迁到服务器时应按服务器真实路径重新生成；Windows绝对路径不能直接在Linux使用。

下载[作者固定权重](https://huggingface.co/ziyanhuang/STU-Net/commit/75d046881d0273899b65bf3ef1ee601e9973e7c8)后设置：

```bash
export IMAGECHD_STUNET_WEIGHTS=/path/weights/small_ep4k.model
```

117497661字节，SHA256 `f440f401bf4cac1d1d6f7c4635542ef057193a5f96c0b3c661a6179afeab7be6`。加载使用限制类型的weights_only=True；120个骨干张量逐值核验，只重置10个输出头张量。缺失、错误或层不匹配均停止，不退成随机训练。首次训练会保存本地权重位置以便恢复；已训练检查点推理无需再次下载预训练文件。

## 面板

用`bash start_server.sh`准备独立Linux环境。面板数据路径填写私有输入描述目录；使用独立ImageCHD runtime/results。

1. 准备环境和独立预处理。
2. 新建训练任务，选择完成的预处理，默认S1；S0可另选作对照。
3. 新建预测任务，选择完成的训练；验证集选best，冻结后评价外折，禁止用外折分数回选模型。

```bash
bash start_server.sh --stage prepare --data-dir /path/private_input_descriptor --runtime /path/new_runtime --results /path/new_results --models S1 --non-interactive
bash start_server.sh --stage train --input-job /path/completed_prepare_job --runtime /path/new_runtime --models S1 --global-batch 4 --non-interactive
bash start_server.sh --stage predict --input-job /path/completed_train_job --runtime /path/new_runtime --models S1 --non-interactive
```

默认三折；明确选择`--scope ct-first`仅运行fold0。新正式任务以作者公开代码为主：1000轮×250次训练/50次验证，S1微调初始学习率1e-3、S0随机对照1e-2；SGD、momentum0.99、Nesterov、weight_decay3e-5、poly学习率、deep supervision Dice+CE。末轮checkpoint_final为主评价，best仅诊断；旧250轮冻结任务仍按旧代码续跑，不直接冒充新配方模型。用户选全局batch，默认4，扩卡不增加总batch。复用v2实时内存保护、完整epoch恢复与输入身份校验；不兼容方法/代码/缓存须新任务。

nnU-Net规划器提供间距和patch，实际训练及尺寸预检使用STU-Net。规划估计不能替代真实显存测量。本机诊断batch1不能代表正式batch4或服务器20GiB预算验收。

预测主结果为原始输出；LCC仅作预先定义的诊断对照。当前面板预测服务冻结外折评价。真正独立新病例可用三折原网格类别概率平均再argmax；该新病例融合入口尚未交付，不能改旧病例ID绕过划分。

## 验证

```bash
python tools/quick_check.py
python -m unittest discover -s tests -p test_imagechd_contract.py
python tools/verify_stunet_integration.py --weights /path/small_ep4k.model --output /path/new_diagnostic_output
python tools/verify_ignore_ddp.py --output /path/ignore_ddp_result.json
```

端到端诊断使用合成影像运行真实预处理、预训练加载、参数更新、验证、检查点恢复、滑窗预测和CPU还原，保留证据，不启动正式训练。忽略标签下CE按全组有效体素归一化，Dice保留全批/逐patch定义；两rank Gloo检查包含2+1不等分及整rank被忽略。Linux CUDA/NCCL、服务器容量、真实外折精度仍待实测。

源码可入Git；源影像、标签、MRB、私有描述/审计、缓存、权重、日志和本地路径不提交。项目验收入口为`python tests/run_panel_suite.py`，需要计算环境；`--legacy-template`保留原模板测试，涉及旧Wholeheart数据/路径/发布机制，不能作为本任务数据验证证据。

[官方代码](https://github.com/uni-medical/STU-Net)、[论文](https://arxiv.org/abs/2304.06716)、[Apache-2.0许可](docs/STUNET_LICENSE.txt)。S约14.6M参数，不是14亿参数的H。四个网络模块按官方文件blob `85587aca5fdf1180dc266a138f3ae97c231875b2`提取，trainer适配nnU-Net2.8 API。来源见[TEMPLATE_ADAPTATION.json](TEMPLATE_ADAPTATION.json)；原模板manifest记录适配前文件。
