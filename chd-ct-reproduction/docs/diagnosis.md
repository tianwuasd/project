# 独立诊断：解剖特征、决策树与规则对照

这一层接在已有分割预测之后，但每一步需要单独启动。不会隐式预处理 CT、训练 U-Net 或重做分割。主分类器是**每个疾病一棵浅层决策树**，允许一例同时有多个阳性标签；规则结果另列，不把两者概率相加。

论文 §4.4 通过解剖结构与专家规则诊断。这里的可训练树是项目扩展，候选规则是可审阅的研究实现，均不等于作者完整临床系统或已验证诊断性能。

## 先跑一次，查看“为什么这样分”

```bash
python start.py --task diagnosis-demo --device cpu --non-interactive --output runs/diagnosis_demo
```

这个入口会检查环境，再用 20 个合成分割病例完成独立的诊断训练、预测和评估。打开 `runs/diagnosis_demo/diagnosis-demo/prediction/diagnosis-report.html`：展开每种疾病，可看见实际访问的特征、数值、分支阈值、最终叶节点病例数及阳性比例。合成标签由演示程序构造，分数只证明程序能运行。

若环境已检查，也可以直接 `python diagnosis.py demo --output runs/diagnosis_demo_direct`。安装包提供等价的 `chd-diagnosis` 命令。

## 真实数据：六个独立步骤

以下路径是示例，每个输出目录都必须尚不存在。先安装当前项目依赖：`python -m pip install -e ".[dev]"`，其中增加了 scikit-learn。

```bash
# 1. 导入真实疾病标签，默认空白为 0；原表不修改
python diagnosis.py labels --input data/imagechd_raw/ImageCHD_dataset/imageCHD_dataset_info.xlsx --output runs/dx/labels

# 2. 单独生成分割预测；不传 --split，覆盖原固定划分的全部病例
# 这里使用已完成正式训练的模型，不能拿短测精度代表正式性能
python predict.py --prepared data/imagechd7_native --models runs/chd_train/models --output runs/dx/segmentations --device cuda

# 3. 只读取上一步预测分割，提取解剖描述
python diagnosis.py features --prepared data/imagechd7_native --predictions runs/dx/segmentations --output runs/dx/features

# 4. train 拟合，val 选浅树复杂度，test 不参与
python diagnosis.py train --features runs/dx/features --labels runs/dx/labels --output runs/dx/classifier

# 5. 测试集诊断和真实路径解释（同时输出规则对照）
python diagnosis.py predict --features runs/dx/features --classifier runs/dx/classifier --output runs/dx/prediction --split test

# 6. 独立评估诊断，不重新训练或预测
python diagnosis.py evaluate --features runs/dx/features --labels runs/dx/labels --predictions runs/dx/prediction --output runs/dx/evaluation --split test
```

第 2 步属于原分割流程；若已经有当前版本生成的全病例预测报告，直接从第 3 步开始。只有 test 预测不能训练诊断树，训练至少需要非空 train/val 特征。新版本预测报告记录模型模式及分割训练清单哈希；旧报告来源不足时请重新运行预测，不手动伪造来源字段。

诊断训练必须复用分割训练的**同一预处理清单和患者划分**。既有缓存可复用，不能重新划分后把参与过分割开发的患者放入诊断 test。对于新 CT，可独立做无标签预处理、分割和特征提取，再调用 `predict --split all`；需使用与分类器训练时一致的分割模型。

## 空白标签的约定

按用户约定，已存在疾病列中的空白默认作为 `0`。原表明确 `0`、明确 `1` 都保持原值，`unknown`/`?`/`na`/`nan` 及整列不存在则保留未知。`labels.json` 的每例 `assumed_negative` 记录哪些阴性来自空白，不改变原始 XLSX；`labels-review.csv` 只包含病例编号和疾病标签，不复制出生、检查日期等无关信息。

本机 110 例原表的实际导入汇总：158 个明确阳性、1 个明确阴性、1601 个空白。转换后 158 个阳性、1602 个阴性；89 例至少有一个阳性、21 例在这 16 列下均阴性。这是本项目的标签处理约定。实际训练数量还取决于分割预处理保留的病例和 train/val/test 划分。

16 个代码原样保留：ASD、VSD、AVSD、ToF、TGA、DORV、CAT、CA、AAH、DAA、IAA、PA、APVC、DSVC、PDA、PAS。不会凭缩写自动把 CA/PA/PAS/CAT 映射为候选规则里的其他疾病代码。可另运行 `labels --blank-policy unknown` 做敏感性分析；训练和评估报告分别记录采用的约定。

AVSD 在原表没有阳性；某疾病训练集中不足 2 个阳性或 2 个阴性时，会标记 `not_trainable`，预测显示“证据不足”。若所有疾病均不满足条件，命令返回退出码 2、保存 `trainability.json`，不会生成空的成功模型。

## 决策树实际看什么

`features.py` 从七结构预测标签提取 56 个描述：

| 描述 | 含义 |
|---|---|
| 每结构是否存在、占前景体素比例 | 模型预测出的七结构形状与相对大小 |
| 每结构连通部分数、最大连通部分占比 | 预测碎片和完整性 |
| 每结构在三条网格轴的跨度比例 | 跨度除以相应网格长度，无毫米单位 |
| 七组结构的标签邻接 | 一体素六邻域相邻，是连接的代理特征 |

空前景或非法标签拒绝提取；某结构缺失时，相关几何/邻接为未知。网格轴不是临床左右，标签相邻也不能直接证明室间交通、主动脉起源等。这些描述可能受分割错误、裁剪和扫描范围影响。

训练按疾病排除未知标签；缺失描述只用该疾病训练病例的中位数辅助拟合。默认 `--max-depth 4 --min-leaf 2`，候选深度 2/3/4（受上限约束），验证集有正负两类才选优，否则固定深度不超过 2。叶节点最少病例数保持为用户设定值。验证计分与部署一致：遇到路径必需特征缺失会弃判，并在选树分数中计为未正确判断，避免填补值带来虚高评分。

部署不把填补值冒充观测证据：沿树走到某个缺失特征就停止，输出“证据不足”。一片叶子内**训练阳性数 / 训练病例数**是输出的 `probability`，大于 0.5 判断阳性，等于 0.5 判断阴性；未经校准。树若没有分支，仅显示训练先验，不能宣称找到个体解剖依据。HTML 解释的是模型如何判断，不是疾病病因。

`classifier.json` 保存完整树、训练中位数、各疾病是否可训练、开发病例/患者编号和来源哈希。`diagnosis.json` 保留逐步路径；`diagnosis-report.html` 供人阅读；`diagnosis-metrics.json` 给出每病及微平均混淆计数、准确率、敏感度、特异度、F1、AUROC、覆盖率与弃判数量。准确率等只统计有结论病例，同时提供 `correct_fraction_all_known` 以所有已知标签为分母；分母为零或 AUROC 无两类时为 null，不伪造零分。

测试评估拒绝与诊断开发数据重叠的患者或病例。当前是固定划分上的研究基线，未实现嵌套交叉验证、分割折外训练特征或临床风险校准，也不采用原论文特殊的 top-two 判对约定。

## 规则如何保留

默认 `--method both`：树负责可训练分类，候选规则单独显示。也可 `--method tree` 或 `--method rules`，后者无需分类器。规则需要经复核的临床解剖证据，不会自动将体素邻接当作真实解剖连接；证据不足输出 indeterminate。

提供 `--evidence reviewed-anatomy.json` 可逐例输入经复核的事实及来源，例如：

```json
{"cases": [{"case_id": "ct_1001", "source": "人工复核记录编号", "features": {"conn_LA_RA": true, "single_atrium_confirmed": false}}]}
```

这只是输入格式，不能把示例当实际病例事实。规则文件在 `configs/diagnosis-rules.yaml`；尚未核实的临床阈值保留 null。规则原始疾病代码另列，只有与诊断表同名的疾病进入对应评价，不做含混别名转换。

## 服务器按步骤运行

进入服务器的 `project/chd-ct-reproduction`：

```bash
# 先检查环境和独立诊断链路，不占用 GPU
bash diagnosis_server.sh --task diagnosis-demo --non-interactive

# 默认标签表位于配置中的 dataset/imageCHD_dataset_info.xlsx
bash diagnosis_server.sh --task diagnosis-labels --non-interactive
# 覆盖来源可用 --dataset /新数据目录，或 --diagnosis-source /具体诊断表.xlsx

# 如尚无全病例预测，先单独使用原分割入口；不要指定 --split
bash predict_server.sh --models /完整路径/某次训练/train --non-interactive

# 后续成功步骤会记住各自结果目录
bash diagnosis_server.sh --task diagnosis-features --non-interactive
bash diagnosis_server.sh --task diagnosis-train --non-interactive
bash diagnosis_server.sh --task diagnose --diagnosis-split test --non-interactive
bash diagnosis_server.sh --task diagnosis-evaluate --diagnosis-split test --non-interactive
```

诊断步骤一律 CPU，不改变原分割的显卡设置。每次只派发选中的一步，先做环境检查。省略 `--non-interactive` 可以逐项输入路径；也可用 `--prepared`、`--predictions`（分割结果）、`--features`、`--diagnosis-labels`、`--classifier`、`--diagnoses`（诊断结果）覆盖相应输入。`--tree-depth` 与 `--min-leaf` 调整浅树大小；统一菜单的诊断集合参数为 `--diagnosis-split`，原分割集合仍为 `--split`。

输出在配置 `results` 的新时间戳目录，子目录分别是 `diagnosis-labels/`、`diagnosis-features/`、`diagnosis-train/`、`diagnose/`、`diagnosis-evaluate/`。短测或来源未确认的特征仅能用 `--allow-smoke` 做软件验证。不存在有效真实分类器时不会假装完成真实诊断。

## 从哪里读代码

依次读 `diagnosis/labels.py` → `features.py` → `classifier.py` 的 `train_classifier` / `explain_tree` → `predict.py` → `evaluate.py`。规则单独看 `rules.py` 和 YAML。入口和参数在 `cli.py`；最小完整例子是 `demo.py`，与原 `imagechd/` 六阶段模型代码分目录。
