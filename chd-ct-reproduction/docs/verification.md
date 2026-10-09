# 验证记录 · 2026-10-07

本记录区分软件验证与论文结果复现。

## 一键启动增量验证

2026-10-07 新增 `start.bat` / `start.py`、数据选择、环境探测、完整数据校验和小样本试跑。

* 完整测试：`python -m pytest -q --basetemp runs/wizard-all-tests`，**25 passed**。
* 新测试覆盖中文及空格路径、目录/CSV 选择、多清单拒绝猜测、患者划分、原数据不变、副本 spacing、缺初始标签的 partial/退出码 2、缺依赖失败报告、check 模式不训练、拒绝覆盖已有输出。
* 使用子进程实际完成“选择数据目录 → 环境检查 → 合成完整试跑 → 所选数据完整试跑”；测试输入是合成几何体，不是患者影像。
* Windows 实际运行 `start.bat --demo --mode check --non-interactive --output runs/batch-launch-check` 成功。
* 独立审查用 `python -B -S start.py --help` 验证没有科学依赖仍可进入启动器；未发现阻止交付的重要问题。
* 未打开系统文件选择对话框进行人工 UI 操作，也未运行 CUDA 或真实临床数据测试；这些范围不列为已验证。

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

## 2026-10-09：服务器入口与 archive 检查

- 完整回归：`python -m pytest -q -o cache_dir=runs/pytestcache_server`，**37 passed**。
- `ruff check .`、新增 Python 文件格式检查、`bash -n start_server.sh`、`git diff --check` 通过。
- `python -B -S start_server.py --help` 通过：服务器引导模块在尚未安装科学计算库时也能载入。
- 实际运行 `python start.py --demo --device cpu --non-interactive --output runs/server-cpu-verification-20261009`，八阶段训练与完整推理通过，report.json 为 passed。
- 新增测试覆盖 GPU 分配范围、忙卡/显存不足、原始 ImageCHD/分卷拒绝、预检保持尺寸但限制 epoch/batch、partial/失败阻止正式训练、pip 安装目标隔离、中断进程组清理和并发保存设置。
- 独立只读审查提出的 pip 目标继承、中断残留子进程、并发设置临时文件竞争已修复并加入回归测试。并发测试用两个写入完成后顺序原子 rename 的交错复现问题，避开 Windows 同时替换同一文件的访问限制。
- 本地完整解压 13 卷 ImageCHD 成功；核对 110 对影像/标签的网格、全部标签值及两份说明表。见 [archive 检查报告](imagechd-archive.md)。
- 服务器的 Linux 自动安装、真实文件锁和 CUDA 全尺寸预检尚未实机执行；CPU 合成测试不能证明 3090 显存足够或真实诊断有效。
