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
