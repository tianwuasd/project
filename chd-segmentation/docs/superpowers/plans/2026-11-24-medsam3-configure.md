# MedSAM3 Server Configuration Implementation Plan

**Goal:** 在服务器读取已有绑定清单和冻结划分，生成基础与微调YAML，无需上传两份配置。

**Architecture:** 在既有模型CLI加入configure，沿用main模板及服务器目录约定。自动读取清单data_root及训练/development病例；保留included/index_scan与患者分离。材料缺失明确报告；不执行模型、环境安装或训练，已有不同配置拒绝覆盖。

**Tech Stack:** 既有Python CLI、PyYAML、冻结JSON清单。

- [x] 新增configure模块和CLI；只读元数据，排除测试角色。
- [x] 同一两例实际绑定清单检查真实CLI、重复不写、用户配置保护及未绑定清单拒绝；零训练更新，自动清理两份测试YAML，保留报告。
- [x] 更新服务器命令说明，保存并发布本任务代码；服务器执行状态以实际回执为准。

本地真实检查通过；报告为server-config-check/report.json。服务器SSH返回Permission denied，未取得会话、未写入配置，未安装环境、上传权重或启动训练。
