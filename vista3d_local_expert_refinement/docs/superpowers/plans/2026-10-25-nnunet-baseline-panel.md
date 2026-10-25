# wholeheart nnU-Net 基线与服务器面板实施计划

> **For agentic workers:** Use superpowers:executing-plans to implement task-by-task. 用户已明确要求开始实施和上传，无需再次询问执行方式。

**Goal:** 交付可运行的 CT nnU-Net 监督基线、独立实验面板、验证记录并上传既有 project 仓库。

**Architecture:** 复用 whole_heart/project1 的 nnU-Net 官方规划、训练/评估以及标准库任务面板，独立复制并适配本项目。运行时从用户指定数据目录创建私有清单，默认 A+B→G，8 fit/22 U/4 selector/4 cal；基线只使用 fit 与 cal，U/selector 保留给后续研究。所有训练与评价输出在独立运行目录，不进入源码归档。

**Tech Stack:** Python 3.11+、nnUNet v2 2.8.1、PyTorch 2.8.0/CUDA 12.8、标准库控制面板、Linux Bash。

**Spec:** ../../wholeheart_initial_plan_2026-10-25.md 与用户本轮“先以 NNUNET 为 baseline，以 skill 形成面板，上传 GitHub，开始干”。本轮不实现 VISTA/专家/半监督/采纳器，不将监督基线称作完整研究方法。

## Global Constraints

- 仅 CT，三个中心留出方向；默认 ct_holdG，只有 B0 模型。不得自动启动真实长训练。
- 内部 0..7 与原始 0/205/420/500/550/600/820/850 一一往返；原始影像不修改。
- 指纹只拟合 8 例来源 fit，验证只用 4 cal；U、selector、target 标签不进入训练/预处理。
- 同源协议中的 selector 4 例是预留数据，B0 实际读取标注为 12 例；不得将 U 说成已用于半监督。
- 数据目录必须显式指定；运行清单及结果不公开。已核实 project 目前为私有仓库，更新只能经认证原站，禁止公共镜像。
- 任务命名/历史/源码快照/检查点续跑/正确进程中断/只读查询/1秒显卡视图/独立更新/缓存复用需实际存在。
- 不改变旧 whole_heart 工程；原研究归档保持不变。

## Review Focus

1. U/selector/target 标签误入 nnU-Net 指纹或训练缓存：数据契约测试验证读集和清单。
2. 老项目缓存或全标注模型被当作本项目少标注输入：缓存签名与工程标识测试必须拒绝。
3. CPU 查询或预处理等待 GPU：控制面板/工作进程测试检查依赖与执行顺序。
4. 退出面板误停后台、恢复使用新配置、PID重用：复用生命周期回归测试并尽可能进行 Linux 合成验证。
5. 私有仓库更新把认证信息送公共镜像：线路白名单测试，仅允许已验证 GitHub owner/repo HTTPS或SSH。

## Task 1: 数据协议与隔离清单

**Files:** Create 05_src/cohort.py, tests/test_cohort.py, 06_configs/baseline.json, 04_data/manifests/README.md; adapt 05_src/data_pipeline.py and server_data.py.

**Interfaces:** discover_cases(data_root) -> list; make_splits(cases, seed=20260919) -> dict; bind_workspace(code_root, work_root, data_root, gpu_memory) preserves existing worker contract. Each split contains train/val/selector/unlabeled/target and total annotation budget.

- [ ] Write tests for 58 eligible CT, 8/4/4/22/20 default split, deterministic stratification, duplicate identities, label access, mismatched data and cache identities; run to see missing behavior fail.
- [ ] Implement runtime-only inventory, source-only label conversion and strict split validation; run data tests to green.
- [ ] Commit task and record actual verification.

## Task 2: nnU-Net compute pipeline

**Files:** Adapt prepare_nnunet.py, server_prepare.py, project_trainers.py, server_train.py, smoke_train.py, server_launch.py, evaluation*.py, metrics.py, gpu*.py, model_catalog.py; add focused tests.

**Interfaces:** Preserve prepare/train/predict/summarize stages and frozen run.json used by panel; only BaselineNNUNet/B0 is registered. Checkpoints are atomic and source split is mandatory. Environment and training parameters are frozen per run.

- [ ] Add tests for B0-only registration, source split enforcement, private output boundaries, label roundtrip, raw-grid evaluation and no implicit random split.
- [ ] Adapt existing tested engine, retain official nnU-Net transforms/optimizer/loss; remove B1 and unrelated prior publication assumptions.
- [ ] Run unit tests and an isolated synthetic API smoke where dependencies are available; no patient training. Commit and ledger.

## Task 3: 面板、交付与发布

**Files:** Adapt task_panel/runtime/worker/inputs/queries/update.py, runtime setup, start_server.sh and bootstrap_server.sh, server docs, README and .codex handoff. Add lifecycle/privacy tests.

**Interfaces:** bash start_server.sh and --query status|gpu|models|logs|processes|history|results; --name/--tasks/--input-job/--resume-job with existing semantics. Runtime/results/data roots independent; credentials never in saved plans or logs.

- [ ] Test panel query without torch, naming/navigation, historical restore and cache mismatch, private authenticated updater, source snapshot and process locks.
- [ ] Adapt control plane to project identity and generic explicit paths; no zmic44 absolute defaults. Keep source snapshots and read-only results.
- [ ] Run complete suite, help/menu checks, Bash validation and Linux process tests if available; mark any untested server/GPU boundary explicitly.
- [ ] Request fresh code review, fix material defects with regression tests, update docs and handoff.
- [ ] Publish only this project’s source/docs changes through connected GitHub account; verify remote commit and changed paths, copy reviewed code back to working project. Mark goal complete only after remote verification.
