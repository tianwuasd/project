# 结构依据与当前实现

采用engineering-project-reference可裁剪源码结构。业务目前为ImageCHD分析和可选择方法的预处理，当前方法为stunet-s、stunet-s-partial。main只发布预处理，训练/预测单独实施和验收。目录名称保留用户指定的chd-ct-panel-v2，目前没有面板。

## 依据

- `src/`隔离业务包与根目录工具，参照[PyPA src layout](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/)；本次交付源码目录，配置仍按源码根定位，不虚称wheel安装已验证。
- `pyproject.toml`记录包与可选依赖，参照[PyPA项目配置](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/)；环境部署独立于业务导入。
- CLI/配置→监督器→业务叶函数→输出验证，参照[nnU-Net公开源码](https://github.com/MIC-DKFZ/nnUNet)的experiment_planning、preprocessing和training职责分离，不改安装包。
- STU-Net相关框架参照[官方代码](https://github.com/uni-medical/STU-Net)，当前预处理使用锁定nnU-Net2.8.1接口，未将其旧版训练入口直接套用后宣称验证。
- CPU资源与恢复适配server-engineering skill V2已捆绑源码；不复制Wholeheart研究合同、网络、batch或面板。

这些是依赖方向与职责的依据，没有所有科研工程必须同名同层级的单一标准。复杂度按实际任务裁剪，不预建不可用训练/预测空壳。

## 调用关系

```text
run.py / smoke.py
  → cli.py / config.py
    → preprocessing/registry.py                   方法配置
    → preprocessing/run.py                        合同/审计/分折/执行/恢复
      → acceleration/preprocess_cpu.py            阶段与实时准入
      → preprocess_worker.py                      独立病例进程
        → preprocessing/stunet_s.py                方法变换和验证
        → nnU-Net2.8.1原生裁剪/归一化/重采样
      → fingerprint.py                            train统计与官方spacing
      → observability                             日志/状态/锁/读取准入
```

已有分析走analysis_run与data/analysis，不因预处理增加而修改其只读语义。短测与正式流程共用业务实现，根入口只转发参数。查询、方法列表和环境计划不启动训练。

## 配置、身份与边界

方法参数在configs/preprocessing，默认机器路径在configs/paths和私有project.local。CLI显式路径优先，相对路径按源码根，空路径报错。源清单、方法参数、软件版本和生产代码hash约束复用；资源上限不改变科学统计。

分析与预处理输出分开，方法/协议/运行分目录，历史97角色只过滤不重新划分。每折独立拟合train；后续训练还必须单独限定train/val/test标签访问。预处理计划没有STU网络结构或正式训练patch；128³只是压缩块提示。

父进程单写总日志、事件与状态；病例进程写独立日志和结果。每秒资源采样、阶段预算分别记录，不能用CPU槽预算替代实际活动进程。源SHA前后不变，产物逐块回读后才提交完成标记。拒绝不兼容恢复时先保护旧证据。

验收包括合成科学合同、cgroup事实注入、OS锁、真实nnU-Net/Blosc2接口及本地真实输入。实际病例分母见各次输出，不能把几例通过当成全96对三折完成。Linux正式负载、新装服务器环境和GPU训练尚需后续独立验收。
