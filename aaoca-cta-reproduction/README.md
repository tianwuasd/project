# Shiri AAOCA CCTA 研究准备

研究对象：[Shiri等，Nature Communications 16:3095 (2025)](https://doi.org/10.1038/s41467-025-58362-9)。作者发布版本固定为 `b134e621943f60f20caf6a9272aa85ceeeb441f8`，来源与非商业许可见 [THIRD_PARTY.md](THIRD_PARTY.md)。

已验收输入冻结、患者隔离、私有输入迁移，以及r2两例作者原索引裁剪/真实TorchIO0.19.6分类重采样/量化/轴序/几何/复用。r2与固定作者直接计算一致，28.031秒完成并清理。真实MONAI SENet154三任务、作者增强、完整latest/best、五成员预测和明确确认标签接口已编写；分类模型、训练恢复及实际预测尚未数值验收。现有分割只用于定位，未知诊断保持为空。

## 入口

```bash
python -B setup_env.py
python -B run.py audit --config project.local.json
python -B run.py check-training --config project.local.json
python -B run.py export-inputs --config project.local.json
python -B verify_inputs.py --config project.local.json
python -B run.py --code-info
```

`audit`显式生成或核对分组，已有不同清单时停止。`prepare --case`和`check-training`直接读取冻结清单，不重新遍历全部病例。`prepare`只核对所选病例的状态与几何，同源同方法且输出状态相符时复用。`smoke.py`限私有配置中的两例，零训练更新，结束自动清理本轮产物。

`export-inputs`生成私有相对路径清单及服务器配置。`bind-inputs`在目标机器显式核对路径与大小、登记目标机器状态，不改变分组；已有不同配置或清单时停止。大小检查不等于文件内容哈希校验。私有配置和清单不公开。

## 资源与验收

定位使用整例解剖标签，距离计算缩到结构区域，原CT只读取裁剪范围。阶段估算依据标签头/实际bbox/ROI及重采样数组，源码保留100GB/整机80%约束。各数据保留源数组方向，索引偏移的物理方向不同，统一分类输入方向尚未验证。实际已验和未验边界见[能力审核](docs/CAPABILITIES.md)。

新增路径集中管理线程、数据供给及单个显式CPU/CUDA设备，分阶段主存准入和50ms实际主存超限终止代码已接入，仍需真实运行验收。没有自动扩卡、分布式分类或等价大块处理验收。作者参考方向未确认时正式分类入口拒绝启动，不自动把输入转成RAS。新增接口与运行前条件见[分类模块](docs/CLASSIFICATION.md)。

两例真实定位裁剪及复用短流程总14.13秒，零训练更新；4份临时文件约16.1MiB已清理。计时来自Windows本地，不能作为全量或服务器速度证据。训练/预测速度尚无实测。

## 文件与服务器

- `src/aaoca_reproduction/`：输入冻结、定位裁剪、迁移引用、资源准入与时序记录。
- `configs/zmic.example.json`：公开服务器配置示例，不含真实病例编号或诊断。
- `update.py`：复用相邻CHD工程的现有更新核心，只同步本独立目录，固定Git提交并核验内容。
- `private/`、`work/`、`reports/`：私有运行内容，排除Git；`state_root`可放course，相对路径按配置文件目录解释。
- [服务器部署与上传](docs/SERVER.md)：最小上传清单、现有环境及入口。

当前缺少已绑定确认监督、经审核方向和实际分类权重；现有MONAI1.5.2按必要源码兼容依据复用。完整论文复现与正式分类任务仍未验收，模型计算、训练及权重下载须取得各自具体许可。
