# 验证记录 · 2026-10-07

本记录区分软件验证与论文结果复现。

## 已运行

* Windows，Python 3.14.3，PyTorch 2.10.0+cpu；CUDA 不可用。其他直接依赖见 `requirements-tested.txt`。
* editable 安装成功；`pip check` 输出 `No broken requirements found.`。
* `python -m pytest -q --basetemp runs/test-tmp-05`：16 passed；包含八阶段训练、推理、原方向输出、错误 stage/配置/数据折拒绝、缺失标签、未知诊断、物理距离和规则冲突。
* `python -m ruff check src tests scripts`：通过；`ruff format --check`：30 files already formatted；`compileall -q src`：通过。
* `python scripts/run_demo.py --output runs/demo`：成功生成 4 个合成几何体、验证 manifest、训练八阶段并输出 `runs/demo/prediction/`。
* 独立只读代码审查：未发现阻止交付的重要问题；额外核查了空证据三值结果、init 类别映射和右侧坐标约定。
* 未执行真实 CT 训练、GPU 训练、临床测试、三折临床实验或论文指标对齐。

首次测试的临时目录被环境权限阻止，之后将临时目录限定在项目下。首次 pip 缓存写入也被限制，已通过 `--no-cache-dir` 和可写 TEMP/TMP 解决，不修改系统代理或全局 Python。

## 纸面配置的参数量检查

使用 PyTorch meta device 只建立参数形状，不分配真实权重内存，得到：

| 模型 | 本项目参数数 |
|---|---:|
| crop64 / all64 | 90,435,542（每个） |
| crop128 / all128 | 50,877,670（每个） |
| init64 | 90,428,110 |
| init128 | 50,872,094 |
| blood2d | 497,780,707 |
| blood_lstm（不重复计算冻结编码器） | 369,347 |

论文相应报告约 84.9M/51.2M、2D 452.9M、BiConvLSTM 0.2M。**本项目参数量不完全一致**；因此准确称谓为架构/流程复现，不能称逐层精确复刻。正式模型需要较大显存；并未证实它在论文 4×11GB 配置下的训练可行性。

## 合成输出的解释

所有示例输入均为规则几何体，随机初始化后只训练少量步数。有限 loss、能写出 NIfTI、能输出规则结果仅证明流程连通；无论输出何种候选疾病，都不反映医学能力。仓库只提交代码、配置与文档，示例影像/权重/逐体积输出留在本地忽略目录。
