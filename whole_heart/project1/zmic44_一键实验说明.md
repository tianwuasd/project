# zmic44：Whole-heart 新任务面板

2026-10-25更新。两个入口已统一：

```bash
bash run_zmic44.sh
```

面板默认查看任务；新建后依次选择步骤、历史输入、模型及实验范围。模型支持 **B0、B1或两者多选**，训练和预测采用同一份冻结选择。完整操作与命令见 [服务器启动说明](服务器启动说明.md)。

| 用途 | 既有服务器候选路径，启动前需核实 |
|---|---|
| 运行环境/下载缓存 | `/data5/zhougaowei/zhangruichen_workspace/project1_runtime` |
| 原始数据 | `/data_nas/zhangruichen/baidu_import/Wholeheart_Train_Dataset` |
| 结果根目录 | `/data_nas/zhangruichen/project1_results` |

每次新任务在结果根目录建立独立子目录，不直接覆盖旧结果。可用`--runtime`、`--results`、`--data-dir`覆盖位置；原`--base-dir`、`--work-dir`保留为别名，其中新面板的`--work-dir`表示结果根目录。

旧`.zmic44_settings.json`和runtime日志中的工作区可在历史列表只读发现；也可以`--query results --job /旧工作区`。旧记录不自动成为可续跑的新任务；原来代码签名不一致时不能强行续训。需要继续使用旧已完成模型时，通过新任务选择历史输入，保留来源与独立输出。

默认后台，提交后通过`s/m/l/g`查询。退出面板或关闭SSH不会主动停止任务；中断需选定任务，保留最后完整检查点。模型训练仍为CT250轮、MRI300轮，seed0；B0/B1同初始化比较规则保持。单模型结果不报告两组差值。

本次未连接zmic44、未安装服务器环境、未提交真实训练。代码同步、服务器部署和真实运行是不同状态。
