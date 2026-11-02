# V2资源策略来源

本项目按工程参考skill逐模块实施，没有复制面板或启动训练。`src/chd_stunet/_vendor/vista_live_resources.py` 是server-engineering skill VISTA 0.7.2中 `live_resources.py` 在 `_rank_setting` 之前的完整前缀，仅含标准库导入、CPU/cgroup/内存采集、预算和病例策略；无训练/DDP部分。

来源固定提交：`47016371e8d56a19630585f49c2cc0299259fa34`。本机源资产：`server-engineering/assets/vista-v2/source/05_src/live_resources.py`。实际父级队列借鉴同版 `cpu_case_queue.py` 的阶段总预算/实时新增余量分离，以及只抵扣已核实直接子进程USS的修正；内存预留原则参考 `memory_guard.py`。这些策略在 `acceleration/preprocess_cpu.py` 适配本项目独立病例任务、Windows与日志接口。

几何表解析器 `data/author_excel.py` 复用此前CHD工程已审核的geometry-only解析器，不读取表中个人元数据。源文件SHA256及裁剪后的资源模块SHA256见交付包 `code-manifest.json` 的provenance。运行时还在 signature 和 code/ 保留当次实际版本。

适配新增的真实边界：一病例一独立子进程、无训练池/共享内存、单线程SciPy/Blosc2/Torch、最大三轮压力恢复。最小4GiB预留比单纯RAM80%更保守。资源数据仍是采样和预估，不能保证避免所有瞬间OOM。

Windows虚拟环境可能以小型redirector再启动真正Python。项目通过CPython的 `__PYVENV_LAUNCHER__` 机制直接启动基础解释器，同时保留原虚拟环境的sys.executable/sys.prefix；测试核对Popen PID与计算PID相同，确保直接子进程USS、实际RSS和退出清理对应真正计算进程。执行中的worker从本次输出code/快照导入，后续源码更新不混入当前已冻结病例任务。

Linux/cgroup测试使用可注入的proc和祖先cgroup证据，不将Windows成功描述为Linux正式验收。模板的GPU扩卡、固定batch、编译、训练预取机制不属于本次CPU预处理范围。
