# CHD：数据分析与可选择方法的预处理

这是独立的代码工程，沿用 `chd-ct-panel-v2` 目录名。按engineering-project-reference逐模块实施，当前提供数据分析、环境部署及可选STU-Net-S预处理；没有面板、训练、预测。

## 开始使用

```bash
python setup_env.py --plan --profile preprocess
python setup_env.py --check --profile preprocess
python run.py check
python run.py prepare --list-methods
python smoke.py --stage prepare --method stunet-s --output outputs/prepare-smoke
```

新环境用 `python setup_env.py --install --profile preprocess` 部署到独立 `.venv_preprocess`，后续入口自动选择它。不下载权重、不申请GPU、不修改系统Python。Linux路径默认沿用此前服务器目录；本机路径保存在不入Git的project.local.json。无需历史描述文件，程序自己生成输入清单。

## 预处理与恢复

```bash
python run.py prepare --method stunet-s --fold all
python run.py prepare --method stunet-s --fold 0 --output outputs/fold0
python run.py prepare --method stunet-s --fold 0 --output outputs/fold0 --resume
python run.py status --output outputs/fold0
```

`stunet-s`审计110对，按确认规则纳入96对；保留原97对角色。辅助标签仅在衍生数组中映射ignore8，各折只用train拟合统计。短测限制输出病例数，不缩小原图或缩减拟合训练集。方法、协议、运行输出各自分目录。科学规则、部署和完整日志说明见[预处理使用说明](docs/preprocessing-usage.md)。

新增`stunet-s-partial`：保留同一96例主队列，另处理13例缺MYO病例为`aux_train`；源灰度问题病例继续排除。它保存已标类别、缺标类别和背景歧义，供后续部分监督训练消费。新增病例不参与主训练集统计拟合，也不作为验证/测试真值。详见[部分标注预处理](docs/partial-preprocessing.md)。

```bash
python smoke.py --stage prepare --method stunet-s-partial --output outputs/partial-smoke
python run.py prepare --method stunet-s-partial --cohort auxiliary --fold 0
python run.py prepare --method stunet-s-partial --fold all
```

当前仅完成预处理，不能直接把部分标注产物交给普通完整监督损失。每折`training_contract.json`明确后续加载器和损失必须消费的标注合同。

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
  acceleration/
    cpu.py                      既有本地分析策略
    preprocess_cpu.py           V2加权队列、实时准入和恢复
  _vendor/                      审计后的V2策略依赖闭包
  observability/                日志、状态、OS锁、读取闸门
  analysis_run.py / reporting.py 分析执行和报告
scripts/package_code.py          白名单打包服务器代码
tests/                          合成科学合同、I/O、恢复测试
docs/                           方法、运行、来源、验收范围
```

业务与CLI/调度分离，科学方法与机器路径分离。新增方法通过registry和相应适配器接入，不覆盖当前方法的协议和缓存。结构依据见[架构](docs/architecture.md)。后续训练与预测等预处理验收后再实现。

## 更新与交付

当前main预处理版为0.3.1，包含完整标注96例和main96＋缺MYO辅助13例两种方法，以及原生四列前景坐标修复。训练模块在独立开发分支，未随本版发布。

新预处理协议为v2，前景坐标按nnU-Net2.8.1要求保存为`(channel,z,y,x)`。旧版运行保留；升级后新建输出目录，不用新代码对旧协议运行执行`--resume`。同版本、同合同的运行仍可正常恢复和复用。

```bash
python update.py --plan
python update.py --check
python update.py --apply
python scripts/package_code.py
```

更新器只接受本目录代码白名单和SHA清单，先暂存核验，再替换并保留旧文件，不删除未知文件、不更新环境或相邻CHD工程。远程更新从GitHub main读取，私有仓库复用已有GH_TOKEN/GITHUB_TOKEN、Git HTTPS或gh认证，不保存凭据；拒绝携带认证跟随重定向。只有SSH授权时，可先按已有Git方式同步代码目录，再使用 `--source` 离线更新。代码发布与服务器部署、全量预处理分别验证。

代码包不包含outputs、影像、权重、虚拟环境、私有配置；真实清单、日志、报告和分析图留在结果目录。

## 检查

```bash
python -m unittest discover -s tests -v
python -m compileall -q src run.py smoke.py setup_env.py preprocess_worker.py update.py
```

本轮交付源码目录，已支持从不同终端目录调用，未承诺普通wheel安装资源定位。Windows真实数据验证不代替Linux实际负载、服务器新环境、正式训练或模型精度验收。
