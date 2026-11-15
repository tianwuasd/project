# 快速检查

`chd_segmentation.preflight.quick_check(config, stage, prepared=None)` 提供非训练型 preflight，并返回可 JSON 序列化的小型字典。`stage` 可为 `prepare`、`preprocess` 或 `train`；`train` 必须提供 prepared 描述。调用者应先用 `load_config` 读取和验证配置。

检查包括配置中的数据集、范围、折数和统计来源；清单及划分映射、病例分组计数、训练与验证标签和正式患者身份；病例影像与标签路径可读；预训练权重 SHA256；关键 Python 包版本、插值与映射 API 和 CUDA 可用性；目标路径现有父目录可写以及对应磁盘剩余容量。

提供 `prepared` 时，函数复用 `load_prepared`、`validate_prepared_config` 和生产环境的 `validate_storage(full=False)`。这会核实 prepared 内容摘要、清单和计划来源身份、折与划分、离线配方及离线文件大小和修改时间 stamp。它不会重新计算离线数组 SHA256。需要离线数据完整性深度复核时，应显式调用生产预处理核验流程的完整模式；快速检查不替代该流程。

检查不会读取 CT 或标签体素，不会创建探测文件，不会初始化网络或执行训练步骤。路径权限基于 `os.access` 和最近存在的父目录判断；Windows ACL、网络盘及挂载状态可能使该结果与实际写入能力不同。预处理容量依据 NIfTI 头和冻结计划计算未复用病例的裁剪前输出上界、分块中间文件和读盘并发；容量不足直接停止。已有病例仅核实其元数据和持久文件标记，不在快速检查阶段重算大数组摘要。

依赖缺失、路径错误、身份不符或权限不足会直接抛出异常。成功结果中 `ok` 为 `true`；失败时调用方应记录异常，不能把失败当作通过。
