# CHD：数据分析、预处理、训练与预测

这是独立的命令行工程，沿用 `chd-ct-panel-v2` 目录名。0.6.0 提供 STU-Net-S 整幅预测、原测试集评价和新病例多折概率集成，保留多 GPU 训练、换卡或减卡与自适应 CPU 供数。已停止的 0.5.2 和 0.5.3 训练可以将检查点导入新运行目录。[多卡运行与继续训练](docs/multigpu.md)、[预测指南](docs/prediction-usage.md)。

MedSAM2 使用独立环境、缓存与训练配置，提供框提示式七结构分割。正式训练默认 200 个 epoch。见 [MedSAM2 使用说明](docs/medsam2.md)。

## 开始使用

```bash
python setup_env.py --plan --profile preprocess
python setup_env.py --check --profile preprocess
python run.py check
python run.py prepare --list-methods
python smoke.py --stage prepare --method stunet-s --output outputs/prepare-smoke
```

新环境用 `python setup_env.py --install --profile preprocess` 部署到独立 `.venv_preprocess`，后续入口自动选择它。不下载权重、不申请GPU、不修改系统Python。Linux 数据集从 `/data_nas` 读取，预处理、训练和预测默认输出到 workspace；本机路径保存在不入Git的project.local.json。无需历史描述文件，程序自己生成输入清单。

环境部署/检查会立即显示当前配置、解释器和完整日志路径，逐步输出创建环境、安装依赖与API核验的开始、结束、耗时和退出码；子进程输出同时刷新到终端和`outputs/environment/*.log`。连续10秒没有新输出时显示当前步骤及已运行时间，不代表测得了下载进度。失败后停止后续部署步骤，完整日志与失败记录保留，成功核验后才保存解释器配置。

Linux虚拟环境的Python可能链接到系统同一二进制，但环境和依赖不同。CLI按未解符号链接的执行路径切换环境，并保留重复切换保护；不能用`Path.resolve()`后的二进制路径认定两个环境相同。旧版遇到“环境检查通过、业务却缺包”时，可直接用`.venv_preprocess/bin/python smoke.py --stage prepare ...`启动。

部署时Torch版本约束同时进入后续依赖解析，安装结束再核对版本与CPU/CUDA构建，偏离计划则不保存环境配置。对用户已有环境执行`--check`仍检查实际API并明确报告版本差异，不自动重装可信环境；API通过不等于与默认部署版本一致。

## 预处理与恢复

```bash
python run.py prepare --method stunet-s --fold all
python run.py prepare --method stunet-s --fold 0 --output outputs/fold0
python run.py prepare --method stunet-s --fold 0 --output outputs/fold0 --resume
python run.py status --output outputs/fold0
```

`stunet-s`审计110对，按确认规则纳入96对；保留原97对角色。辅助标签仅在衍生数组中映射ignore8，各折只用train拟合统计。短测限制输出病例数，不缩小原图或缩减拟合训练集。方法、协议、运行输出各自分目录。科学规则、部署和完整日志说明见[预处理使用说明](docs/preprocessing-usage.md)。

新增`stunet-s-partial`：保留同一96例主队列，另处理13例缺MYO病例为`aux_train`；源灰度问题病例继续排除。它保存已标类别、缺标类别和背景歧义，供后续部分监督训练消费。新增病例不参与主训练集统计拟合，也不作为验证/测试真值。详见[部分标注预处理](docs/partial-preprocessing.md)。

预处理成功后自动精简已完成任务的描述、结果副本、最新进度及遗留暂存，训练数据、恢复合同和详细日志保留。旧结果可用 `python -B run.py compact --prepared PATH` 只读查看，加 `--apply` 应用。实际移除量与占用写入 `cleanup.json`，见[产物保留与精简](docs/preprocessing-artifacts.md)。

STU-Net 新预处理默认保留三折计划，共用一个活动折缓存目录。`--fold all --cache-fold 0` 只生成第0折数组；第0折训练结束后，在同一预处理输出目录运行 `prepare --resume --fold all --cache-fold 1` 切换，再训练第1折，第2折同理。切换按各折原参数重新计算，模型与检查点分别保存，同一缓存同时只供一折训练使用。旧缓存继续使用原布局，详见[共享缓存](docs/preprocessing-shared-cache.md)。

```bash
python smoke.py --stage prepare --method stunet-s-partial --output outputs/partial-smoke
python run.py prepare --method stunet-s-partial --cohort auxiliary --fold 0
python run.py prepare --method stunet-s-partial --fold all
```

训练入口已消费这些标注合同，支持完整监督及缺MYO的部分监督；不能把辅助产物交给忽略该合同的普通损失。见[训练使用说明与结构](docs/training-usage.md)。

```bash
python setup_env.py --check --profile training
python smoke.py --stage train --output outputs/train-smoke
python run.py train --fold 0 --gpus auto --output outputs/train-fold0
```

训练默认使用已保存的预处理run和权重路径；正式1000轮/B4/128³，短测另记64³/B1。默认从main train与aux_train按病例均匀抽样，验证仅main val；正式运行要求完整本折缓存。未通过容量准入不暗降规格。

多卡遵循固定全局batch：B4的1/2/3/4卡份额为[4]/[2,2]/[2,1,1]/[1,1,1,1]。`--gpus auto`在获准可见范围内选择空闲卡，`--workers`是全组CPU供数上限。详见[多卡使用、恢复与验收边界](docs/multigpu.md)。

CPU调度使用skill V2 0.7.2：CPU/RAM80%、至少4GiB主机余量、按重采样体积准入、有界独立病例进程、内层1线程、独立读入闸门、每秒内存监督和逐例恢复。来源与边界见[V2适配](docs/v2-acceleration-provenance.md)。预算不是实际CPU使用量，也没有加速倍数结论。

## 只读分析

```bash
python smoke.py
python analyze.py
python analyze.py --output outputs/analysis/one-run
python analyze.py --output outputs/analysis/one-run --resume
```

分析仅提供统计和质量标记，不改写原始数据或执行方法纳入规则。图表、CSV、逐例SHA和状态在本次输出目录；详见[分析口径](docs/analysis-method.md)。既有分析使用本地独立调度；Linux预处理使用本轮新增的V2资源适配。

## 代码结构

```text
run.py / analyze.py / smoke.py   共用业务实现的薄入口
setup_env.py                    标准库环境计划/检查/隔离部署
update.py                       仅更新本目录代码，保留路径与产物
preprocess_worker.py            调度器启动的CPU病例进程
configs/
  analysis.json                 分析策略
  paths.json                    已确认的服务器默认路径
  preprocessing/*.json         完整标注/部分标注方法协议与科学参数
  training/stunet-s.json         版本化微调配方与短测覆盖项
src/chd_stunet/
  cli.py / config.py            阶段、方法选择、路径解析
  data/                         只读统计、配对、作者表解析
  preprocessing/
    registry.py                 已实现方法列表
    annotations.py              已标/未标类别与背景语义合同
    cohort.py                   方法的标签与历史角色规则
    fingerprint.py              仅train拟合及官方spacing规则
    stunet_s.py                 审计、变换、写出验证
    run.py                      正式与短测共用监督流程
  training/                     模型、部分监督loss、取块、训练、原子恢复
  acceleration/
    cpu.py                      既有本地分析策略
    preprocess_cpu.py           V2加权队列、实时准入和恢复
    training_cpu.py             有界常驻供数、读取限流和确定性消费
  _vendor/                      审计后的V2策略依赖闭包
  observability/                日志、状态、OS锁、读取闸门
  analysis_run.py / reporting.py 分析执行和报告
scripts/package_code.py          白名单打包服务器代码
tests/                          合成科学合同、I/O、恢复测试
docs/                           方法、运行、来源、验收范围
```

业务与CLI/调度分离，科学方法与机器路径分离。新增方法通过registry和相应适配器接入，不覆盖当前方法的缓存。新预处理v2修正原生取块所需的四列前景坐标，训练对两份已审计旧生产者提供只读适配。结构依据见[架构](docs/architecture.md)。`prediction/` 提供旧模型整幅预测、测试集评价和新病例多折概率集成，资源调度位于 `acceleration/prediction.py`；命令与结果说明见[预测指南](docs/prediction-usage.md)。

## 更新与交付

main和训练分支同步包含预处理优化与训练入口。已完成的旧版预处理仍可只读用于新训练，不需要重算或改签名；源合同、每折计划、角色、文件SHA和坐标仍严格核查。预处理的 `--resume` 使用经过摘要核验的原运行冻结生产者，严格保留原方法、版本与源码合同。

```bash
python update.py --plan
python update.py --check
python update.py --apply
python scripts/package_code.py
```

更新器只接受本目录代码白名单和SHA清单，先暂存核验，再替换并保留旧文件，不删除未知文件、不更新环境或相邻CHD工程。远程更新从GitHub main读取，私有仓库复用已有GH_TOKEN/GITHUB_TOKEN、Git HTTPS或gh认证，不保存凭据；拒绝携带认证跟随重定向。只有SSH授权时，可先按已有Git方式同步代码目录，再使用 `--source` 离线更新。代码发布与服务器部署、全量预处理分别验证。

`update.py --check/--apply`自动显示授权读取、清单版本、逐文件一致/下载/SHA校验及备份替换进度，长请求每10秒反馈等待状态。完整过程写入`outputs/updates/<时间>/update.log`，成功或失败均有`update-result.json`；终端不显示访问凭据。

训练开发分支保留为`codex/chd-stunet-training-20261103`。本次验收后与main同步，无需再为取得训练功能手动切换分支；后续开发提交仍须独立验收后才合入main。

代码包不包含outputs、影像、权重、虚拟环境、私有配置；真实清单、日志、报告和分析图留在结果目录。

## 检查

```bash
python -m unittest discover -s tests -v
python -m compileall -q src run.py smoke.py setup_env.py preprocess_worker.py update.py
```

本轮交付源码目录，已支持从不同终端目录调用，未承诺普通wheel安装资源定位。Windows真实数据验证不代替Linux实际负载、服务器新环境、正式训练或模型精度验收。
