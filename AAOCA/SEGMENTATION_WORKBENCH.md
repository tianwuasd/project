# AAOCA 命令行直接分割

版本 0.6.0。入口为 `start_segmentation.py` / `start_segmentation.sh`，直接接收命令参数，不提供交互菜单。输入为有 HU/mm 来源依据的原始三维 CT，输出为原始 CT 网格上的 `aaoca_11.seg.nrrd`，供医生在 3D Slicer 中修订。

## 环境、许可与清单

代码更新使用 [AAOCA 子目录更新](CODE_UPDATE.md)，下载范围限定为当前 AAOCA 源码目录。分割环境、许可、模型、输出和冻结任务继续保存在原 runtime 与结果目录。

在 AAOCA 目录运行，系统 Python 使用 3.11–3.13。运行环境、缓存、模型和许可保存在指定的私人 runtime；已有 runtime 的环境和许可可以复用。

```bash
bash start_segmentation.sh install --runtime /srv/private/aaoca-runtime
bash start_segmentation.sh license --runtime /srv/private/aaoca-runtime
bash start_segmentation.sh check --runtime /srv/private/aaoca-runtime
bash start_segmentation.sh inventory --inventory /srv/private/data/CT总清单.json --data-root /srv/private/data --manifest /srv/private/aaoca.csv
```

license 提示输入许可且不回显；自动化可用 --stdin 通过私人管道提供许可。许可不进入命令参数。inventory 仅导出已核验的 AAOCA 原始体积，私有 CSV 不上传。单个三维期相对应一行，格式见[分割说明](tools/coarse_segmentation/README.md)。

## 直接运行与 GPU 设置

下面的示例要求 0、1 已分配给当前用户：

```bash
bash start_segmentation.sh run --runtime /srv/private/aaoca-runtime --manifest /srv/private/aaoca.csv --output-root /srv/private/labels --gpus 0,1 --workers 4 --io-workers 4 --prepare-workers 1 --export-workers 1 --prefetch 2 --min-free-gib 8 --gpu-reserve-gib 2 --background
```

--gpus 使用当前 PyTorch 可见编号，程序保存完整 GPU UUID。--workers 是全部阶段的进程并发上限；--gpu-count 限制同时使用的卡数。准备、GPU 推理和 CPU 导出分别调度，每张卡同时处理一个本任务病例；同一病例依次运行 total、heartchambers_highres、coronary_arteries。

默认准备并发1、导出并发1，workers 和 io-workers 默认为卡数加这两项，因此两张卡默认最多4个活跃进程。--prefetch 限制正在准备和已就绪病例的合计数量，默认等于卡数。CPU 提前读取、校验并无损解压下一病例；GPU推理结束后CPU导出可以与后续推理重叠。实际并发受CPU、RAM、磁盘和读写预算限制。

允许所选显卡已有计算任务。派发前要求空闲显存至少为 --min-free-gib 加 --gpu-reserve-gib，默认合计 10 GiB；不会因已有进程或高利用率拒绝这张卡。每个模型开始前重新检查显存，不满足则等待并响应停止请求。该门槛不构成独占预约或显存硬限制，其他进程的新增分配和病例大小仍可能导致不足，失败会记录。

--cpu 使用 CPU。--limit 1 或重复 --case-id 可限定检查范围。默认等待任务结束并定时打印进度；--background 提交后立即返回。已提交任务由独立进程运行，SSH 断开后继续。首次完整推理串行建立权重缓存，之后开放多卡病例并行。

队列在派发前累计活跃阶段的保守磁盘预算，准备、推理和导出分别估算。就绪文件的实际占用包含在文件系统剩余空间中，数量受prefetch限制；没有活跃阶段且空间不足时明确停止并保留可验证缓存。临时准备仅执行无损解压，模型内部的重采样仍由TotalSegmentator完成。

## 查询、停止与恢复

已准备病例在推理中断后保留缓存并进行哈希核验。旧任务继续使用冻结代码；更新不会替换正在运行任务的调度。先停止旧任务并确认进程退出后，可在新run命令追加 --continue-from 旧任务编号（或latest）。程序校验旧任务代码、相同的模型/标签/导出算法、依赖版本、原始输入及最终标签；跳过已完成病例，保留旧标签位置并在新任务中记录引用。未完成病例使用新调度重新准备和推理。损坏成果和不兼容算法会明确拒绝接续。

下面命令默认操作同一 runtime 中最近的分割任务；--job 可指定任务编号。

```bash
bash start_segmentation.sh status --runtime /srv/private/aaoca-runtime
bash start_segmentation.sh logs --runtime /srv/private/aaoca-runtime
bash start_segmentation.sh stop --runtime /srv/private/aaoca-runtime
bash start_segmentation.sh resume --runtime /srv/private/aaoca-runtime --background
```

任务请求、代码及输入身份冻结。恢复使用原冻结任务，完整标签通过哈希检查后复用；正在推理的未完成病例重新执行三个模型，已完成推理可继续 CPU 导出。状态与日志查询不会安装环境或启动推理。

## 文件与医生审核

结果保存在结果根目录下的 `seg-<任务编号>/cases/<体积代号>/attempt-*/aaoca_11.seg.nrrd`。保留十一种结构的名称、颜色、全部重叠区域和原始 CT 空间；最终文件原子写入并完整回读后，删除本任务十一份中间掩膜和临时解压 CT。

模型内部执行必要重采样、裁剪和标准化；外部不保存预处理影像或概率图。小型私人来源、质检、状态和日志随任务保留。医生加载原始 CT 与标签，重点复核冠脉开口、近端走行及细小分支。程序质检不表示临床准确性已验证。

模型返回空 ROI 时，只有任务日志明确记录空裁剪、掩膜全部为零、尺寸及空间矩阵与毫米单位原图一致，才为缺失单位的空掩膜补回毫米标记，并记录在 QC。其他缺失单位或空间不符的输出会拒绝导出。QC 的 empty_masks 列出未检出的结构；十一项全部为空表示模型未检出结构，需要人工检查原图及模型适用范围。

GitHub 仅发布源码、通用测试和说明。影像、病例编号、清单、输出、许可、权重与私人日志保留在私人服务器。
