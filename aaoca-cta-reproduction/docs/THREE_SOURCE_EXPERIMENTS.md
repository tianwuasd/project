# 作者检测模型的多来源研究实验

复用本项目run.py、统一分类预处理、完整检查点和逐成员概率融合。诊断表、清单、空间来源及病例统计保存在私有输入包，公开源码不包含患者资料。

## 输入与来源审核

`audit-sources --config BASE`读取既有清单的影像摘要及患者登记，报告精确文件交集。配置demographic_excel时，使用作者两种实际Excel布局，规范化出生日期、检查日期及性别，记录候选关联。候选关联不能证明同人；清单生成保守排除关联CHD患者的全部扫描，保留ImageCHD对应病例。没有共同UID时，不宣称患者独立性已经证实。

`build-cohort --config BASE --diagnoses NEW.csv --output INPUTS [--check]`读取患者级研究CSV，按明确诊断、合并症纳入条件、CT文书状态及实际登记对应选择阳性。阴性须有用户明确确认。CHD患者多次扫描只选一次；缺少LV或AO标注者不进入该定位方案。旧保留患者继续保留，旧运行不改。

患者级标签单独使用patient_research策略：保留CSV字段与患者匹配来源，ct_matched=false和same_study_unverified持续保存，不填造医生确认或同次检查依据。旧confirmed、文书标签和旧ImageCHD确认接口保持。

CHD1282方向、作者间距、数组索引配对与强度依据通过spatial_binding保存。局部零点可用于裁剪；解剖方向与相对数组配对须有来源。记录用户接受的抽查方向、同索引研究政策以及HU未校准状态。现成标准化分割缓存不能替代原始强度图像。本CHD标注使用LV=1、AO=5；ImageCHD使用LV=1、AO=6。

## 绑定与统一预处理

`bind-cohort --config BASE --inputs INPUTS --output RUN --chd1282-root CHD_ROOT --data-root DATA [--check]`绑定原有影像及标签，不复制CT。BASE提供现有AAOCA和ImageCHD根，可用对应根参数覆盖。输入包含新CSV，无需另建英文表。

```bash
python -B run.py prepare-dataset --config RUN/project.dataset.json --check
python -u -B run.py prepare-dataset --config RUN/project.dataset.json
python -B run.py prepare-author-weights --config RUN/project.dataset.json --author-archive AUTHOR_ZIP --check
python -B run.py prepare-author-weights --config RUN/project.dataset.json --author-archive AUTHOR_ZIP
python -B run.py generate-experiments --config RUN/project.dataset.json --output CONFIGS --device cuda:0
```

预处理共用一份：80×80×60mm定位区域、RAS输出、215×215×85、B-spline、固定HU256级量化。完成项复用，已归属的部分写入按既有批次规则接续；仅缺失或为空的定位可跳过，其他错误直接停止。跳过病例从训练、验证、预测和来源审查共同排除。每例保存均值、标准差、分位数及零值比例，不改变强度数组。

作者ZIP只提取检测任务5个部署成员，逐成员严格加载；提取记录绑定ZIP条目、CRC、长度和输出状态。原包保留，不下载或改写。来源不明的已有成员不覆盖。

新生成的成员描述使用PyTorch内存映射读取权重，减少完整检查点的主存副本；严格参数加载保持。旧成员描述维持原读取方式。

## 三组实验

生成project.author.json、project.finetune.json、project.finetune_robust.json。共享患者分组和预处理。作者原始世界方向未知，结果属于作者权重在本地研究输入上的评估。

- author：不更新作者5成员权重，直接预测。
- finetune：每个作者成员独立初始化对应本地成员，保留原增强。
- finetune_robust：同样微调，增加轻度gamma、噪声、模糊及模拟低分辨率。三来源采用同一增强分布，无左右翻转、来源特定强度模板或测试数据拟合。

微调采用5轮分类层暖机，随后最后一个阶段与分类层训练；固定BatchNorm统计，分类层学习率1e-4、最后阶段1e-5，AdamW权重衰减0.01，最多30轮。验证AP选优，并列时选择较低验证损失，连续5轮无改善停止。保留原BCE。完整latest保存参数组、阶段位置、随机状态及优化器。单卡依次执行成员，沿用资源等待机制。

```bash
python -u -B run.py finetune-ensemble --config CONFIGS/project.finetune.json --resume
python -u -B run.py finetune-ensemble --config CONFIGS/project.finetune_robust.json --resume
```

五成员入口的resume允许尚未开始的成员首次启动；已有成员从自己的latest继续，不增加更新预算。底层train的单成员恢复约束保持。
已完成成员核对真实latest中的模型、输入、配置身份及更新次数记录后直接复用，不再创建模型与优化器。

## 融合验证与测试

每组先验证，再测试。五个sigmoid概率等权平均，不平均参数文件，不根据测试挑选成员。验证确定最大Youden J阈值，并绑定实际成员文件；成员变化后须重新验证。结果分别保留，非空目录不覆盖。

```bash
python -B run.py predict-ensemble --config CONFIGS/project.author.json --split validation --device cpu --check
python -u -B run.py predict-ensemble --config CONFIGS/project.author.json --split validation --device cpu
python -u -B run.py predict-ensemble --config CONFIGS/project.author.json --split test --device cpu
```

微调组换相应配置执行同样两条命令。check仅核对名单、描述和输出计划，不运行模型；实际严格加载核对成员、任务及训练/验证患者。独立测试拒绝与成员训练或模型选择患者重叠。

结果包含整体指标、按来源指标及患者bootstrap区间。单一阴性来源AUROC/AP为空，特异度和假阳性照实报告。`audit-shortcuts --config RUN/project.dataset.json`生成仅在训练患者拟合的灰度统计基线及阴性来源识别报告。来源识别变难不能单独证明来源混杂消失；阳性仅来自一个来源的限制持续保留。研究抽样比例不解释为临床患病率。

CPU小输入功能检查不证明完整尺寸训练容量、CUDA性能或临床效果。旧配置、已完成预处理、模型和结果保留，新实验使用独立data/course目录。本项目update.py继续只更新受管理源码。
