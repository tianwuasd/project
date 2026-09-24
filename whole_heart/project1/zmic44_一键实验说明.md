# zmic44：从启动到拿到 B0/B1 结果

本次仍然只有两种方法：**B0 基线、B1 外观增强**，没有加入 CaberNet。CT 与 MRI 分开建模，五个中心留出方向各跑两种方法，共 **10 次正式训练**，随机种子先用 0。

这是启动工具的交付说明；尚未在 zmic44 上实际安装或训练，不能把代码测试通过理解为服务器实验已完成。

## 1. 默认位置已经按你的服务器调整

| 内容 | 默认目录 |
|---|---|
| 你提供的本地个人工作目录 | `/data5/zhougaowei/zhangruichen_workspace` |
| Conda、训练环境、下载缓存、临时文件、后台日志 | `/data5/zhougaowei/zhangruichen_workspace/project1_runtime` |
| 原始数据候选位置，启动时需要核实 | `/data_nas/zhangruichen/baidu_import/Wholeheart_Train_Dataset` |
| 预处理数据、模型、预测与最终报告 | `/data_nas/zhangruichen/project1_results` |

只在你提供的个人子目录内创建本项目内容，不动 `/data5/zhougaowei` 下的其他文件。预处理大文件仍放 NAS，避免本地盘约 115 GB 的剩余空间被五套缓存和模型挤满；NAS 读写可能拖慢训练。环境盘最低检查 20 GiB，结果盘最低检查 100 GiB，这不是容量保证。

服务器信息依据你提供的 9月22日更新文档：Ubuntu 24.04、8 张 RTX 3090 24 GB、驱动 580.173.02、尚无 Conda。启动器自动安装个人 Miniforge，不要求 sudo，不修改驱动，不需要 Docker 或 Hugging Face。

## 2. 第一次复制这些命令

登录服务器后，把代码下载到你的个人目录。若该位置已经有 `project` 仓库，跳过 clone，进入仓库后用 `git pull --ff-only` 更新；正在运行实验时不要更新源码或环境。

```bash
mkdir -p /data5/zhougaowei/zhangruichen_workspace
cd /data5/zhougaowei/zhangruichen_workspace
git clone https://github.com/tianwuasd/project.git
cd project/whole_heart/project1
bash run_zmic44.sh
```

按照中文提示：

1. 运行目录、结果目录正确就按回车。
2. 填写**已解压原始数据**的实际位置。应包含完整 106 对 `Case..._image.nii.gz` 和 `Case..._label.nii.gz`；不要填压缩包或 Windows 的 `D:\...`。
3. 从显示的显卡列表中填一张获准使用的空闲卡编号，例如 `0`。脚本不会默认占满八张卡，也不会结束别人的任务。
4. 首次建议选 **2：短测**。等短测通过，看预算后，再次运行同一条命令，选 **3：完整实验**，输入 `RUN`。

也可以直接选择 3；程序会自动先短测，通过后开始正式训练。已经通过且代码、环境、划分完全一致的短测会复用，避免重复。

```bash
# 已经知道数据路径时可先传入；仍会询问GPU和阶段。
bash run_zmic44.sh --data-dir "/实际数据目录"
```

不需要自己创建 Conda 环境、不需要手动激活环境。首次下载数 GB 依赖。程序固定 Python 3.11、PyTorch 2.8.0+cu128、torchvision 0.23.0+cu128、nnU-Net 2.8.1。nvidia-smi 显示 CUDA 13.0 是驱动兼容能力，不要求把项目改成 CUDA 13。

## 3. 菜单含义

| 选项 | 做什么 |
|---|---|
| 1 检查 | 安装环境、核对106例文件、检查GPU计算，不做正式训练 |
| 2 短测 | CT/MRI各做B0/B1的短步骤训练，验证保存、恢复、完整来源开发病例推理，并估算本服务器耗时 |
| 3 完整实验／继续 | 自动短测 → 五方向B0/B1训练 → 全部冻结 → 目标中心预测 → 指标与对比报告 |
| 4 仅评价／继续评价 | 要求十次正式训练全部完成，继续尚未完成的目标病例评价 |

CT 每模型 250 轮，MRI 每模型 300 轮，每轮 250 个训练步骤、50 个来源开发集验证步骤。保留原方案 **5 GiB 规划预算**，不会因为换成 24 GB 显卡而自动改变网络规模或 batch。显卡剩余显存不等于可任意扩大本次实验配置。

全套是数天级任务的可能性很高。过去本机估计的约 155 小时仅是核心计算外推，**不是 RTX 3090 的实测耗时**。以服务器短测的 `09_reports/短测与预算.md` 为准；预处理、NAS写入、完整预测和距离评价还需要额外时间。

## 4. 关掉SSH后怎么查看

默认会后台执行，安装过程也写日志；不依赖 tmux。启动后显示的“已提交后台任务”不代表成功，请先查看日志是否正常更新。

```bash
bash run_zmic44.sh --mode status
```

它会显示最近任务的状态和日志位置。复制启动器显示的命令，例如：

```bash
tail -f "/启动时显示的完整日志路径.log"
```

看到正常运行后可以断开 SSH；在 `tail` 画面按 Ctrl+C 只退出查看，不会停止后台训练。管理员回收进程、服务器重启或会话资源策略仍可能结束任务。

若确实中断，再运行 `bash run_zmic44.sh`，使用原来路径并选 **3**：已完成模型跳过，未完成模型从同一实验的 latest/final 继续，评价按病例跳过已校验的结果。如果某个模型连第一轮检查点都没来得及保存，下一次你选择继续时，程序将该模型的失败记录移到 `07_experiments/recovery_archive/`，只把这一模型用seed0重新初始化；其他完成的模型保留。不会拿短测或 best 冒充续训权重，也不会在失败后无限自动重试。

不要同时重复启动；存在环境与结果目录锁。任务被强制结束时状态文件可能滞后，请结合日志与进程确认。脚本只检查单卡是否空闲，不替代实验室的显卡预约或调度规则。

## 5. 最终到哪里看效果

默认输出目录 `/data_nas/zhangruichen/project1_results` 下：

| 文件／目录 | 初学者看什么 |
|---|---|
| `09_reports/final_evaluation/README.md` | 首先打开：CT/MRI各自的B0/B1 Dice与差值 |
| `09_reports/final_evaluation/case_metrics.csv` | 每一例的分数、哪些病例较差、空结构数量 |
| `09_reports/final_evaluation/class_metrics.csv` | 七个结构各自的Dice、HD最大距离和HD95，单位mm |
| `09_reports/final_evaluation/summary.json` | 各中心、最差中心、中心等权均值和成对bootstrap区间 |
| `08_results/predictions/` | 原始图像坐标中的三维预测；raw和lcc两套，可用医学影像查看器叠加原图 |
| `08_results/evaluation/frozen_models.json` | 评价前冻结的全部模型与检查点校验值 |
| `07_experiments/formal/` | 十个实验的日志、状态与best/final模型 |

Dice越大越好。B1−B0为正代表增强组在本次实验更好，不是事先保证增强有效。raw 为主要结果；lcc 为逐类最大连通块次要结果，B0/B1使用同样规则。只使用来源开发集的平滑小块Dice选择 best，**全部十次训练完成并冻结后才评价目标中心**。

单侧结构缺失会保留 Dice=0、HD/HD95=inf，不能删除失败病例。双侧都缺失记为未定义，并保留有效类别数量。该本地评价没有官方血管截断，Dice也是研究用宏平均／中心等权，不能等同于官方体积加权WHS排行榜。区间只描述当前固定中心和模型中的病例抽样变化；当前只有seed0。

原始数据仍只读，排除之前确认的5例，实际101例参与研究。两个方法合计202次目标病例推理，每次同时保存raw/lcc指标。

## 6. 报错时保留这些信息

把最近的后台日志末尾、`job_state.json`、对应实验 `run.json` 发回来即可。不要删除模型或反复改依赖重试。代码、环境或划分变动会拒绝混用旧实验；应在新结果目录启动独立实验。

Miniforge使用[官方26.7.2-0发行包](https://github.com/conda-forge/miniforge/releases/tag/26.7.2-0)，执行前核对官方SHA256。不强行使用Windows本机的127.0.0.1:7897代理；服务器沿用自身网络设置。网络下载失败会停止并保留URL与日志。

旧的 `bash start_server.sh` 仍可用于已有Conda的其他Linux服务器，其 `train/resume` 只负责训练；本服务器请统一用上面的 **run_zmic44.sh**，避免混淆两个菜单。
