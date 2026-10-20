# 服务器资源调度小型验收

这是大 `project` 仓库中的独立案例，与 `chd-ct-reproduction/` 并列。本地副本放在 `D:/code_project/server-resource-smoke/`。需要同仓库 CT 版本 0.13.0 或更新版本。

生成 6 例 18×16×16 的合成 CT，使用 16³/16²、base=2、每阶段 1 epoch 的三折缩小模型。通过真实服务器入口运行环境检查、预处理、短测、测试、预测、评估。训练本身仍使用原来的逻辑；本例不运行正式尺寸预检，也不证明医学效果、真实 CT 内存需求或性能收益。

在服务器的大 `project` 目录执行（Python 指向已有环境，不安装依赖）：

```bash
python3 server-resource-smoke/run.py \
  --python /data5/zhougaowei/zhangruichen_workspace/chd_ct_runtime/envs/chd_py311/bin/python \
  --work-root /data_nas/zhangruichen/chd_ct_results/resource-smoke \
  --gpu-count 2 --gpu auto
```

只使用获准的 GPU 范围；可以把 `--gpu auto` 改为明确编号或 UUID。保留继承的 CUDA 限制。默认总时限 900 秒，包含等待显卡，使用 `--timeout` 调整；实际耗时需服务器实测。少卡仍能完成，但只有一张空闲卡时报告不会声称验证了多卡并行。纯 CPU 验收可加 `--device cpu`，它不能验证 CUDA 路径或释放显卡。

每次创建唯一目录，原始数据、缓存、模型、runtime 和报告都在里面，不修改正式任务。结束输出逐项 PASS/FAIL，并生成 `acceptance.json`。检查三折 OOF、病例数量、CPU 分配记录、推理队列上限、GPU 提前释放和无残留 holder；日志进度和 ETA 使用实际记录。保留目录便于定位失败，不自动删除。

查看实时日志和进度：

```bash
python3 chd-ct-reproduction/start_server.py --runtime /本次验收目录/runtime --query logs --follow
```

Ctrl+C 退出日志查看不会停止任务；运行验收脚本时 Ctrl+C 会请求停止它创建的验收流程。CPU 尾部续跑、低内存预算和中断清理由 CT 项目内自动回归测试补充覆盖；此小型验收用于服务器实际软件联通性核验。

## 追加与上传入口验收（需要 CT 0.15.0）

上述小型案例成功后，使用它实际打印的验收目录替换下面路径。先在原 workflow 追加合成诊断测试，再查看包含新增步骤的结果。成功步骤应显示保留，不应重新训练：

```bash
python3 chd-ct-reproduction/start_server.py --runtime /本次验收目录/runtime \
  --resume-job --append-tasks diagnosis-demo --device cpu --foreground --non-interactive
python3 chd-ct-reproduction/start_server.py --runtime /本次验收目录/runtime --query results
python3 chd-ct-reproduction/start_server.py --runtime /本次验收目录/runtime --upload-job --non-interactive
```

最后一条只准备并预览合成日志、指标、配置及权重上传包，不发送文件、不需要 GitHub 登录。核对清单包含三折 best/last 权重，且不含原始影像和缓存。若确实需要发布，再执行程序打印的带实际包路径的 `--upload-package ... --yes-upload` 命令；它使用官方 GitHub CLI 认证。

运行锁、重复步骤拒绝、旧权重字节/时间戳不变和假 GitHub 失败重试由 CT 项目回归覆盖。真实镜像/上传权限与网络需服务器自行验证，不能把预览成功当成发布成功。

## 续跑校验读取验收（需要 CT 0.16.0）

无需安装训练环境、不用 GPU。用服务器系统 Python 在大 project 目录运行：

```bash
python3 server-resource-smoke/check_resume.py \
  --work-root /data_nas/zhangruichen/chd_ct_results/resume-read-smoke
```

创建唯一测试目录和 6 个共 48 MiB 的合成文件，模拟五个步骤共享缓存。实际调用与面板相同的校验器，打印自动选择的线程数、进度和速度，验证 30 次文件引用合并为 6 个文件、只读取 48 MiB，原内容/mtime 不变，以及保持大小/mtime 的内容损坏仍会被拒绝。成功输出 PASS 并保存 acceptance.json。测试不读写真实 CT、模型或任务，只在合成文件中制造损坏并保留目录；无需重跑前面的训练案例。

这是功能验收，短任务可能来不及显示多次刷新。速度会受操作系统缓存影响，不能用它推断正式 NAS 上的加速倍数。实际任务更新并重开面板后，仍通过 `r` 选择原任务，校验自动使用同一优化。
