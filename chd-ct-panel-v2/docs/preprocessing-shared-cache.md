# 三折训练共用一个数据缓存

STU-Net 保存三折独立的训练划分、强度统计、目标 spacing、训练合同、模型和检查点。共享缓存同时存放一个活动折的预处理数组。切换到下一折时，按照该折原参数重新生成数组，使用相同的数据目录。三折按顺序训练。

各折 train 的统计与 spacing 可能不同；归一化、重采样及前景采样继续沿用该折参数。活动折编号和临时构建编号保存在 `shared_cache.json`，临时编号不进入训练的数值身份。

## 使用顺序

```bash
python run.py prepare --method stunet-s-partial --fold all --cache-fold 0 --output /workspace/prepared/shared
python run.py train --fold 0 --prepared /workspace/prepared/shared --output /workspace/training/fold_0/run
python run.py prepare --method stunet-s-partial --fold all --cache-fold 1 --output /workspace/prepared/shared --resume
python run.py train --fold 1 --prepared /workspace/prepared/shared --output /workspace/training/fold_1/run
python run.py prepare --method stunet-s-partial --fold all --cache-fold 2 --output /workspace/prepared/shared --resume
python run.py train --fold 2 --prepared /workspace/prepared/shared --output /workspace/training/fold_2/run
```

每次预处理都显式生成所选活动折。`--fold all` 保存三折完整计划，训练命令不会自动改写缓存。未指定输出路径时，默认共享目录为结果根的 `方法/协议/shared_cache`；已有目录需要显式 `--resume`。`--storage independent` 可选择各折分别保存的布局。

## 磁盘与计算

`shared_data/` 每例常驻三个文件：float32 CT、int8 标签和属性 `.pkl`。三折的小型计划和完成记录各自保留。原有的审计样本、源合同、冻结代码及详细日志保留。

缓存数组的常驻容量约为一折数组容量。实际容量取决于病例尺寸、该折 spacing 和无损压缩率。写入期间还需要当前并行病例的暂存空间；源数据和旧的独立运行不会自动删除。因此这项结构变化不能保证已有目录占用立即减少，也不能给出固定的 GiB 数字。

换折需要重新执行对应的数值转换和写入，增加计算时间。当前折缓存已经切换后，如需续跑该折，先按原合同切回，再调用该模型的恢复入口。预测从模型记录读取各自的计划，训练模型分别保留。

## 占用与恢复

预处理和训练取得同一个互斥锁。读取进程及预处理子进程分别登记系统文件锁；主进程异常退出后，只要子进程仍占用数据，切换请求就会停止。系统锁随进程退出释放，下一次构建可清理失去占用的登记文件，不终止其他进程。

切换先记录 `building`，所有病例按合同提交后才记录 `ready`。构建中断时，训练拒绝读取；对同一目录和活动折执行 `prepare --resume`，逐例核验三个文件的 SHA256，复用完整病例并重新计算缺失项。旧折完成记录无法让另一折的数组通过核验。

只有已登记、当前不再使用的病例文件可以删除。链接和未知文件会阻止切换。缓存折数、计划合同、病例范围和完整文件摘要继续核验。

## 已有任务

旧独立缓存保持原目录结构；本次不会迁移或清除它们。旧预处理恢复继续运行原冻结代码，旧训练模型继续使用原冻结代码及其缓存。已有半程训练无需切换到共享布局。

MedSAM2 保持独立的逐病例轴位缓存。它读取 STU-Net 保存的源病例清单和三折计划；采用 `--fold all` 生成的共享运行仍提供这些文件。

限定验收入口为 `scripts/check_shared_cache.py --raw PATH --output PATH --case CASE_ID`。入口只处理显式指定的一个病例，不训练模型；验收记录保存在调用者指定的本机目录。
