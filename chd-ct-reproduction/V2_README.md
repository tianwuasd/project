# 在原目录使用V1和V2

V2 2.2.1直接随main更新到 `chd-ct-reproduction`，无需另建服务器项目或进入平级目录。

在原面板用u→1更新main并退出，然后仍在原目录启动：

```bash
bash start_server.sh     # 原V1
bash start_server_v2.sh  # V2：选择STU-Net或原六路方法
```

```text
chd-ct-reproduction/
├── start_server.sh       原V1入口，文件不改
├── src/chd_ct/           原V1实现，文件不改
├── start_server_v2.sh    新V2入口
├── start_server_v2.py    V2的Python入口
├── start_server_v2.cmd   V2的Windows入口
└── v2/                  完整V2代码、配置和测试
```

V2使用独立包 `chd_ct_v2`；原V1仍使用 `chd_ct`。V2的运行目录、环境、任务、缓存和结果沿用其既有独立身份，不接管V1的目录，也不迁移或修改旧冻结任务。STU-Net与V2六路方法同样各有独立根目录。

V1和V2属于同一个项目，统一更新main。V2的u更新整个chd-ct-reproduction目录，备份也包含整个项目；w允许两版的项目源码，继续排除影像、权重、私有状态和其它项目。启动入口和运行状态分开，更新不按版本拆分。同一Git检出上的代码更新应顺序执行；这与独立运行状态是两个层面。

如果仓库只稀疏检出了 `chd-ct-reproduction`，普通cone模式会同时包含其v2子目录，不再要求补平级目录；手写排除v2的规则仍需调整。V2启动控制Python要求3.11+，可沿用CHD_V2_CONTROL_PYTHON设置。

旧平级 `chd-ct-panel-v2` 仅保留启动转接；当前完整实现只有本目录的v2这一份。首次从旧布局迁移，请用V1的u→1；旧V2更新器会按旧范围拒绝跨目录变更。

训练规则保持2.2.0：官方1000轮、S1 lr0.001/S0 lr0.01、默认global batch4、末轮主评价。数据、预处理、标签、划分、损失和科学训练代码均未因目录整合而改变。旧250轮任务继续使用其冻结版本。

版本核查与测试见[v2/docs/V2_2.2.1_INPLACE.md](v2/docs/V2_2.2.1_INPLACE.md)，科学复核见[v2/docs/STUNET_SCIENTIFIC_AUDIT.md](v2/docs/STUNET_SCIENTIFIC_AUDIT.md)。没有服务器部署或正式训练完成的声明。
