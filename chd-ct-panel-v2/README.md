# V2入口已移入原项目

V2的维护源码现位于[../chd-ct-reproduction/v2](../chd-ct-reproduction/v2)。本目录只保留旧启动命令的转接，不保存第二份模型或面板实现。

在原 `chd-ct-reproduction` 目录更新main后直接运行：

```bash
bash start_server.sh     # V1
bash start_server_v2.sh  # V2
```

两版环境、任务和结果保持隔离。旧版V2尚未更新时，首次目录迁移请从V1的u→1完成；旧V2的维护范围保护会拒绝跨目录迁移。

完整说明见[同目录使用说明](../chd-ct-reproduction/V2_README.md)。
