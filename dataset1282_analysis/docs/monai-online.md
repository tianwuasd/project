# MONAI 在线读取

分支 `chd_1282_monai` 与 `main` 独立维护。三个模型入口保持各自独立：Xu 多分支复现、STU-Net-S、SAM-Med3D Turbo。

## 准备

使用原来的研究 Python 和 nnU-Net 2.8.1，增加 `configs/monai-requirements.txt` 中的依赖。不要升级产生原计划的 NumPy、SciPy、PyTorch 等数值依赖；读取器检查原运行记录中的版本和官方预处理模块摘要。

```bash
"$PY" -m pip install --no-deps -r configs/monai-requirements.txt
"$PY" -B scripts/monai_prepare.py --prepared "$D" --output "$W/prepared-online"
```

`D` 指向已完成 HU 预处理目录，`W` 指向 workspace 内的结果目录。紧凑输出保存原有计划、来源摘要、空间信息、部分标签监督和患者划分；不复制 `.nii`、`.b2nd`、模型权重。原 CT、标签和已有结果保持完整。此命令复用已拟合的训练计划；新的科学预处理方案仍需独立拟合和验证。

把后续训练或预测命令的 `--prepared` 指向紧凑目录，即自动启用在线处理。原路径继续使用旧缓存读取。不要把指向新目录直接当作原实验的续跑许可；续跑要求原任务路径或已有经过验证的路径迁移记录。

## 加速和容量

- MONAI Dataset/Compose 按病例读取，直接调用 nnU-Net 内存处理函数。训练、开发及预测都应用原计划，Excel 空间修正不产生整例磁盘副本。
- `CHD_RAM_CACHE_BYTES` 为每进程缓存上限，默认 2 GiB，可设为 `0` 禁止保留空闲病例。实际准入同时检查当前主机/cgroup 余量和调度器给该进程的内存份额；模型、队列和正在处理的数组占用会影响准入。
- 每个进程同时准备一例，保留现有有界 CPU 预取和 GPU 运算重叠。Blood 借用病例期间固定数组，多个切片共用一次读取和重采样。
- 内存不足时释放空闲缓存，最小工作量仍无法满足时触发现有资源等待流程。不会修改 batch、标签、采样顺序或图像尺寸来继续执行。
- GPU 隔离、外部进程检查和各模型原有的选卡规则继续生效。本分支不增加分配给进程之外的可见显卡。
- 源文件、元数据或数值环境发生变化时立即报错。

首次在线处理包括解压和重采样，会比直接读取已完成的压缩缓存耗时。内存命中可省去这部分运算。超过缓存容量的病例会在下一次访问时重算；不能把单例缓存命中的速度推算为整轮速度。

紧凑目录只包含元数据及 `preparation.log`。训练沿用最新续跑状态、选定最佳权重、必要日志与结果的保留规则。整体占用还包括这些正式权重和预测，不能仅用紧凑预处理目录大小估计全部空间。

## STU-Net 数据边界

本分支支持 Dataset1282 已审核数据的 11 类 STU-Net 训练和预测，包括本项目旧缓存与在线输入。`stu_imagechd.py` 另外读取 ImageCHD 的冻结三折描述文件，按原工程的规则把额外源标签编号映射为忽略区域，提供 8 类网络和独立损失；作者间距只应用于内存中的工作空间信息。两种数据使用各自的输出类别和标签定义。这个原始 NIfTI 入口没有替代 ImageCHD 工程已拟合的 nnU-Net 训练计划，正式复现仍应使用该计划及原工程的活跃折读取锁。

ImageCHD 的一例真实前景块和一例含额外标签编号的病例通过只读检查；8 类预训练骨干完成一次 32³ CPU 更新。短测入口：

```bash
"$PY" -B scripts/monai_stu_imagechd_acceptance.py --descriptor "$IMAGECHD_DESCRIPTOR" --weights "$STUNET_WEIGHTS" --log "$CHECK/acceptance.log"
```

## Q 路线一

Q 模型保留独立工程与三个训练阶段。其 `q_model/prepared.py` 通过来源校验后的 `q_route1_bridge.open_q_cache` 读取旧缓存或 MONAI 紧凑目录，原有 11 类重排、A/B/C/D 患者划分及模型来源核验继续生效。`q_model/local_data.py` 的主机准入同时预留尚未分配的 CPU 激活池容量；已占用容量由实时可用内存统计承担。两个修改位于现有 Q 工程中，Q 工程的代码副本与旧检查点摘要必须分别验证。

本机一例真实病例完成旧缓存与在线输入的 Q 类别重排及 32³ 图像、标签读取。旧缓存读取约 2.10 秒；在线冷读取含紧凑元数据生成约 69.91 秒。缺少来源核验通过的 Q 上游检查点，尚未执行 Q 的冻结上游前向或训练更新。

## 真实短测

```bash
"$PY" -B scripts/monai_acceptance.py --prepared "$TWO_CASE_PREPARED" --output "$CHECK"
```

入口固定选择技术子集的一例训练和一例开发数据。可增加 `--device cuda:0 --sam-weights PATH --stunet-weights PATH`，逐个模型完成有限更新及保存重载。检查日志保留输入一致性、耗时、缓存字节、实际产物容量与清理记录；短测副产物自动删除。技术短测不启动完整训练，也不证明正式训练效果或多卡速度。

## 代码与数据

公开分支只保存代码、依赖声明及通用文档。CT、标签、Excel、真实划分、病例摘要、日志、模型权重和预测始终保存于私人工作目录。本分支的更新入口固定跟随 `chd_1282_monai`，支持已有镜像设置；Git 工作目录也可以执行 `git pull --ff-only origin chd_1282_monai`，启动新进程后生效。
