# Windows模型检查入口

本项目的检查调用实际生产代码、已有环境、官方权重及私有病例配置。统一监督程序可以在本项目目录直接调用以下程序，向每项传入独立work-root与report路径。源码目录、已有依赖和原始数据保持原位置。

| 程序 | 范围与参数 |
|---|---|
| tests/verify_models_real.py | Q各阶段、完整恢复、无标签预测和共享评价；--model q_route1 --dataset imagechd --world 1，报告参数为--report-root |
| tests/verify_model_startup_real.py | 实际初始化错误、文件保护、同目录重试和完整恢复；--world 2 --batch-layout 1,3 --forget-layout-on-resume，报告参数为--report |
| tests/verify_external_medsam_real.py | MedSAM2官方权重、普通无标签预测、原空间导出和共享评价；零训练更新，报告参数为--report |
| tests/verify_model_budget_real.py | 两模型实际网格和内存规划、Med恢复worker布局及不足等待；零训练更新，无CUDA计算 |
| tests/verify_model_guards_real.py | --mode admission检查资源不足后重试；--mode q-groups检查冻结患者分组。另需--bundle指向私有Q输入包 |
| tests/verify_windows_supervisor.py | --entry指向统一Windows工具；实际调用其supervise和clean_owned，顺序执行本项目预算与准入检查 |

除分组元数据检查外，以上程序使用--spec指向私有project.local.yaml；每次最多选择一至两例真实病例，每阶段最多两次更新。技术范围由程序明确设置，报告保存实际输入摘要、网格、设备、更新数、耗时、资源采样与文件清单。旧版恢复可为verify_models_real.py提供--legacy-root，需指向已经核验的完整旧源码副本。

示例从本项目根目录执行，SPEC、WORK和REPORT由调用方提供已获准的实际路径：

```powershell
& $TaskPython -B tests/verify_model_startup_real.py --spec $SPEC --work-root $WORK --report $REPORT --world 2 --batch-layout 1,3 --forget-layout-on-resume
```

普通单元检查使用本项目src作为PYTHONPATH，CHD_TEST_ROOT指向获准的独占检查目录，并使用-B禁止字节码写出。Windows上导入smoke的各项程序应顺序执行，避免项目代码锁相互等待。

SmokeWorkspace在正常结束和可传播异常结束时核对目录归属、记录产物并删除独占工作目录。外部监督程序强制终止进程时，应先确认其全部子进程已退出，再依据本轮所有权及报告清理遗留目录。不能删除模型依赖、源病例或其他任务目录。

## 设备与覆盖

CPU/Gloo检查覆盖真实多进程业务、恢复与资源接口。tests/verify_pretrained_device.py含真实单CUDA加载检查，执行该程序需要已有对应显卡许可；该程序不执行GPU训练。两个模型的CUDA训练、NCCL、Linux cgroup及正式规格性能需单独记录实际证据。

统一工具位于D:/projects/test。validate.py已有assistant2-budget、assistant2-admission、assistant2-startup、assistant2-q和assistant2-medsam-evaluation阶段，分别调用上表中的实际项目程序。本轮verify_windows_supervisor.py已使用其common.py监督和清理接口完成预算与准入两项检查，同时核对项目及监督程序摘要。具体工作与报告目录由调用方传入。

本轮接入检查使用真实CPU资源，零训练更新。服务器硬件档案与虚拟显卡场景独立记录；本项目接入结果只覆盖实际执行的阶段，完整仿真场景与CUDA/NCCL结果以各自报告为准。
