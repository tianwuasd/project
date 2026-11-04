# 当前阶段：可选择方法的预处理

当前注册`stunet-s`（96例完整标注）和`stunet-s-partial`（同96例＋13例缺MYO辅助池）。`--method`选择的是预处理协议，不启动模型、训练或面板。部分标注的角色、输出元数据和训练消费边界见[部分标注预处理](partial-preprocessing.md)。以后增加方法时，在`configs/preprocessing/`增加配置、在registry注册，并提供相应的审计/拟合/转换实现与测试；已有方法的科学协议不被新方法覆盖。

## 上传到服务器后

把本目录整体放在 CT 的 `project/chd-ct-panel-v2/`。不要套在 `chd-ct-reproduction` 中，不需要旧工程、旧描述文件或旧面板。传代码时不传 `outputs/`、`project.local.json`、虚拟环境、原始影像。

```bash
cd /你的CT目录/project/chd-ct-panel-v2
python setup_env.py --plan --profile preprocess
python setup_env.py --install --profile preprocess
python run.py prepare --list-methods
python smoke.py --stage prepare --method stunet-s --output outputs/server-smoke
```

安装脚本自身只使用标准库，在 `.venv_preprocess` 部署 CPU 环境并写入本机解释器配置。它不下载权重、不申请GPU、不修改系统Python。`--check` 验证现有配置解释器及实际接口；安装步骤、退出码、完整错误写到 `outputs/environment/`。本机验收用既有研究环境，没有重复安装；服务器新装环境与Linux真实负载需在服务器验证。

Linux 默认原始数据、保存位置、运行根分别在 `configs/paths.json`，沿用此前确认的路径。描述文件可以不提供，程序会扫描文件对及作者间距表，自动生成自己的清单，避免再到代码工作目录猜测 `imagechd_input.json`。`project.local.json` 只覆盖机器路径和解释器；科学参数在方法配置中。

正式处理三个折：

```bash
python run.py prepare --method stunet-s --fold all
```

也可先运行一折。输出目录应固定，重跑时显式加 `--resume`：

```bash
python run.py prepare --method stunet-s --fold 0 --output outputs/fold0-run
python run.py prepare --method stunet-s --fold 0 --output outputs/fold0-run --resume
python run.py status --output outputs/fold0-run
```

相对输出目录以项目根为准，与当前终端目录无关。不提供 `--output` 时自动在结果根按 方法/协议/时间戳 分目录。`--cases`、`--limit` 仅限制最终输出病例，全部源数据仍接受审计，各折始终使用完整训练角色拟合；短测不会缩小原图尺寸。`--workers`是可选上限，不填时由资源策略决定；CLI默认读取上限2，`--readers`可明确覆盖，CPU计算不受这个读取数限制。

## 完整标注基线stunet-s的科学规则

代码目录更新可运行 `python update.py --check`、`python update.py --apply`。私有仓库使用官方GitHub API，并复用已有Git HTTPS/gh认证或环境中的GH_TOKEN/GITHUB_TOKEN；令牌不写入配置、命令参数或日志，也不跟随重定向。若服务器仅有SSH授权，先通过现有Git方式同步一份代码，然后用 `python update.py --apply --source /已同步的/chd-ct-panel-v2` 更新目标目录。`--proxy`仅作用本次请求，不改全局网络设置。

1. 审计原始110对，13对缺必需标注不纳入；另排除已确认源灰度质量问题的病例，获得96对。保留原始文件及历史97对角色，不重新打乱。清单按文件对计数，患者级独立性未得到确认。
2. 保留0..7；仅已知辅助编号映射到 ignore=8，未知编号阻止当前方法。ignore不增加网络输出类别。不会用最大连通域或形态学删改真值。
3. 作者表 hash 必须与已审核来源相同；仅将其报告的 J/K/M 应用于衍生网格的 x/y/z，保存原 affine 和衍生 affine，不声称完成DICOM几何认证。
4. 每个训练病例对1..7标签均匀抽取20万个强度值，按病例等权拟合CT统计。显式排除背景与ignore；验证/测试病例不参与。与官方按数据集大小分配共约一千万样本的统计并非逐位相同，这项采样设置纳入协议。
5. 使用影像非零区域裁剪，并检查目标标注没有被裁掉；不使用真值框或旧定位模型取框。按照锁定 nnU-Net 2.8.1 的顺序先归一化、后重采样。目标spacing用该版本官方的训练集median/各向异性规则。
6. 影像用官方3阶插值；标签用官方按类别的1阶安全重采样逻辑，不把标签值作为连续灰度处理。输出允许 nnU-Net 的裁剪区域标记 -1、背景0、目标1..7、ignore8；目标类别消失时不提交完成项。
7. 使用有界前景位置采样，避免为ignore合同中的背景一次建立并排序全体三维坐标。保留原生前景/annotated-area键与采样比例，随机抽样序列与上游不同。协议v2按nnU-Net2.8.1取块接口保存`(channel,z,y,x)`四列，并写入`class_location_coordinate_format=channel_zyx`；单通道的channel列为0。此前三列格式不能直接交给原生加载器。Blosc2采用1线程无损LZ4写入；不会使用上游写入器中固定的4线程。

## V2加速的实际接入范围

- 复用skill 0.7.2 `live_resources.py` 中 Linux/cgroup事实采集与 `case_policy`，Windows另有psutil适配。
- 每阶段按可用CPU与RAM的80%预算，RAM另保留至少4GiB。按原始及重采样后预计体积准入，实际病例进程和内层线程均受限。
- 开文件、校验hash、解压经过独立读入闸门，CPU计算在闸门外；出现I/O压力时减少新读入。
- 原始阶段预算和实时新增余量分开计算；只抵扣出生身份匹配的直接子进程USS，不用共享RSS抵扣，也不重复扣除活动线程。
- 父进程每秒采集内存，持续低于阶段预留3次后只回收本次自建进程，降低并发并重做未完成项，最多3轮。完整病例不重算；极端外部瞬时抢占仍可能先于采样。

这些是资源控制机制；实际并发、RSS/USS峰值与耗时见运行证据。未作串行对照，不报告加速倍数。Linux采集有合成cgroup测试，本轮真实运行在Windows；不代替服务器实际验收。

## 输出和恢复

0.3.1将两种方法的坐标协议均升级为v2。升级后使用新的输出目录，旧运行保留原代码快照；不要用新代码强行恢复旧协议。恢复仍核验完整源码、配置和输入身份，不删签名或按版本号放宽检查。此main版本只发布预处理，训练入口和旧缓存的训练读取适配在独立开发分支。

```text
run/
  signature.json             方法、输入stat、版本、代码hash
  effective_config.json      本次生效路径与输出范围
  code/                      当次实际代码快照
  manifests/                 主/辅纳入、排除理由、历史角色、分折、消费合同
  audit/cases/               全部源病例检查和源SHA256
  audit/samples/             强度候选样本，拟合时仅取train
  fold_0/                    fold_1/、fold_2/同理
    dataset_fingerprint.json 拟合train IDs、统计和原生空间信息
    preprocessing_plan.json  预处理计划，未冻结网络/训练patch
    splits.json              train/val/test及独立aux_train角色
    training_contract.json   本折拟合IDs与下游消费要求；尚未实现训练
    data/                    .b2nd、_seg.b2nd、.pkl
    completion/              完整产物hash，独立于data扫描目录
    results/                 逐例结果或完整错误
    staging/                 未提交项；无完成标记不可复用
  resources/                 阶段准入与实际峰值
  logs/                      总日志、逐例日志、事件、每秒资源
  status.json                当前阶段/心跳/进度/错误
  summary.json               本次处理/复用/失败及明确分母
  report.md
```

写完后逐块重新读取全部影像及标签并比较每个值，文件hash与完成标记最后提交。中断可保留没有完成标记的暂存文件，恢复时会重新计算该病例。恢复前核对合同、源数据与候选样本，已完成病例还需核对三个产物文件hash；不兼容恢复应在改变旧状态/日志/代码前拒绝。

交互停止用Ctrl+C并等待进程收尾。若从外部强杀父进程，需确认本次残留计算进程已退出再恢复；本轮没有实现对不可捕获强杀的自动孤儿进程接管。新病例入场会在上一个进程退出后重新采集资源，避免沿用退出前的内存读数误判容量不足。

后续训练需要单独建立STU-Net模型和训练计划，显式读取角色并隐藏外折test标签。本次 `preprocessing_plan.json` 不可直接冒充完整训练网络计划，128³只用于压缩块访问提示。


## 0.3.2启动与环境反馈

`python3 setup_env.py --check --profile preprocess`显示当前解释器、实际包版本及与默认部署的差异。已有环境检查不自动重装；对服务器已有Torch2.15等差异先完成两例预处理验收，不把导入/API通过解释成默认部署合同一致。

旧0.3.1业务入口在Linux上可能将虚拟环境符号链接解析成系统Python，从而跳过切换。0.3.2已修正调用路径判断并保留防循环保护；也可始终显式用`.venv_preprocess/bin/python -u smoke.py --stage prepare --method stunet-s-partial ...`。

安装/检查日志自动实时显示并保存在`outputs/environment/`。安装其他依赖时也锁定Torch版本，安装结束核对CPU构建，偏离时不写入配置。更新阶段也有逐文件及等待反馈，日志见`outputs/updates/<时间>/update.log`。本补丁不改预处理科学生产者与配置，实际数据和包版本相同时允许复用0.3.1运行。
