# 验证入口

在固定 Python 3.11 / nnunetv2 2.8.1 训练环境中：

```bash
python -B -X utf8 -m unittest discover -s tests -v
python -B -X utf8 tests/synthetic_api_smoke.py
bash -n start_server.sh bootstrap_server.sh
```

单测使用临时合成文件／模拟显卡，不安装环境、不联网、不读真实病例。显式合成集成检查使用官方 CPU 训练和推理 API，生成并自动清理小体积；不代表真实效果。

`test_task_posix.py` 必须在 Linux 执行；Windows 明确跳过，不冒充通过。`python -S 05_src/task_panel.py --query status` 可单独验证只读入口不依赖 torch 或 nnU-Net。
