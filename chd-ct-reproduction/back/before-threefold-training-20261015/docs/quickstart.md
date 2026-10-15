# 中文引导

Windows双击 start.bat，其他系统执行 python start.py。菜单提供六个选项：环境检查、预处理、训练/短测、预测、评估、合成全流程测试。

一次选择只执行一个功能。数据路径可以粘贴，输入 browse 可打开目录选择窗口。训练/预测需要先准备好的 dataset.json；预测还需完整模型目录 models.json。默认训练模式是smoke。

```bash
python start.py --task preprocess --dataset data/imagechd_raw/ImageCHD_dataset --prepared data/imagechd7_native
python start.py --task train --prepared data/imagechd7_native --mode smoke --device cpu
python start.py --task predict --prepared data/imagechd7_native --models /完整模型目录 --split test --device cuda
python start.py --task evaluate --prepared data/imagechd7_native --predictions /完整预测目录 --split test
```

非交互任务加 --non-interactive，并给出所需路径。每次保存独立日志目录和 report.json。环境探测包括依赖版本、磁盘、CPU梯度计算和CUDA实际计算；桌面引导缺依赖时提示安装方法，服务器入口可以建立独立环境。

正式 --mode train 自动先做preflight；任一步失败停止。每一步的具体参数与结果目录见 [主流程说明](workflow.md)。
