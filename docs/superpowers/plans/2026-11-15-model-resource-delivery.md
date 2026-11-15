# Q与MedSAM2资源及交付 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** 保持两模型科学方法，继承确定公共资源合同，并以一条安装命令复用已有材料，尽量实现零手动上传。

**Architecture:** 公开交付代码生成机器配置并从已有冻结输入重建Q，官方源码与权重按固定身份自动取得。模型只继承已发布公共组件，资源和日志沿用统一入口。

**Tech Stack:** Python3.11、hashlib、PyYAML、现有Git及模型依赖。

**Spec:** docs/superpowers/specs/2026-11-15-model-resource-delivery-design.md

## Global Constraints

- Q统计仅A393；保持r2和40000/5000/20000预算、全局4样本。
- MedSAM2原生配方、八帧512、官方权重、75epoch不变。
- 先等待助手1确定main提交；不编辑其工作树，不覆盖其他项目。
- 服务器路径按数据集和模型，共用course/chd-segmentation资源登记；旧目录不自动迁移。
- 两例、每阶段两次更新、200验证文件/8GiB，报告总根10MiB，结束自动清理；不登录服务器。
- 所有编辑使用文件编辑工具；无捕获错误或隐式替代路径。

## Review Focus

- 已绑定清单与原包仅机器路径不同：保留绑定和mtime。
- 同名目标内容不同或目标指向原始数据：安装前拒绝，既有文件不变。
- 官方下载摘要错误、源码目标冲突或路径越界：保留已有内容，清理本次pending。
- 官方源码/权重在旧服务器路径：核对身份后直接引用，保留原位置。
- 两模型在实时共同预算下降及恢复时：保留全局batch/RNG与同数量进程，不冒充已验证CUDA。

### Task 1: 模型配置生成

**Files:** 新增src/chd_segmentation/delivery/{__init__,configs}.py，tests/test_delivery.py；复用公开configs/models模板。

**Interfaces:** model_configs(root, code, assets)返回五份实际模型配置及目标路径，供Task2补充三个准备配置。

- [x] 写配置行为检查，先观察缺少实现失败。
- [x] 实现数据集模型路径生成及各模型固定科学预算。
- [x] 实际检查统一资源根及局部模型的上游检查点路径。
- [x] 提交并记录结果和产物清理。

### Task 2: 单命令安装及实际正式交付

**Files:** 新增delivery/{install,assets,__main__}.py，scripts/CHD1282-Q-MedSAM2-20261115_install.sh，tests/verify_delivery_real.py；修改模型运行脚本的默认数据集模型路径。

**Interfaces:** install(root, code, source, assets, check=False, dataset_roots=None)返回创建/缺失记录及实际资产位置；inspect_inputs与ensure_q_inputs核对并精确重建Q；复用bind_inputs和manifest_identity，生成Task1的配置。

- [x] 写真实冻结元数据及已有真实权重引用的安装检查，观察未实现失败。
- [x] 实现只读缺失核对、资产身份、全部目标预检查、安装/路径绑定和重复复用。
- [x] 实际检查已有绑定保护、冲突无写入、生成配置、实际CLI与Bash入口；模型生产解析及运行在Task3继续验证。
- [x] 核对服务器已有源数据精确重建完整Q组合，明确零新增手动上传，不打包已上传材料、公开大文件或ct_1126数组。
- [x] 提交并清理测试目录，保留正式安装说明与报告。

### Task 3: 确定上游合同与模型适配

**Files:** 继承助手1确定提交的公共文件；修改模型memory/execution/offline及预测预算、模型脚本、ASSISTANT2_UPSTREAM和精确兼容映射。

**Interfaces:** 以助手1实际确定提交的接口为准，在继承时逐项记入私有ledger；消费共同实时预算、单例上限、快速检查及统一日志。

- [ ] 等待并读取确定发布，记录公开/本地源码提交及文件映射。
- [ ] 针对模型边界的实际失败写检查，保留既有数值和科学合同。
- [ ] 继承共享实现，完整登记Q融合与MedSAM2传播内存，保持恢复语义。
- [ ] 通过生产入口检查两例以内预处理、复用、Q更新/恢复及MedSAM2官方供数/预测，输出统一日志证据。
- [ ] 提交并清理已无必要的验证副产物。

### Task 4: 最终验收、发布和交付

**Files:** docs模型部署与交付说明、私有ledger和集中交付清单。

- [ ] 运行相关单元检查，完成一次新上下文全分支审阅，修复必要问题并实际验证。
- [ ] 核对本次全部副产物、进程、清理与报告容量；正式包内容/数量/容量/来源完整。
- [ ] 在最新main上按项目路径整合，以旧SHA非强制更新并重查实际树，保护助手1及其他项目。
- [ ] 交付正式包位置、上传目的地、固定SHA的--force命令、一条安装命令和已验/未验范围。
