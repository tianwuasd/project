# Excel 重采样接入前的预处理备份

保存日期：2026-10-13。来源提交：`c41434277256588514895e39360964739c398bfa`。

保存了旧版预处理入口、实现及其本地依赖，逐文件 SHA-256 见 `snapshot.json`。所有快照代码均直接取自该提交，未修改；当前项目根目录继续使用加入 Excel 重采样后的版本。

旧版行为：不读取 Excel 间距，不进行毫米间距重采样；统一影像方向并按第 1/99 百分位归一化。默认 `--size 0` 保留方向统一后的网格，正数 `--size` 将体积缩放为指定大小的立方体。标签规则、病例划分和续跑逻辑均保留原实现。

在项目根目录使用现有 Python 环境运行：

```powershell
.venv\Scripts\python.exe back\preprocess-before-excel-20261013\preprocess.py --input data\imagechd_raw\ImageCHD_dataset --output data\imagechd7_before_excel
```

Linux 下同样可使用当前环境的 `python` 运行该入口。入口会优先加载备份目录中的源码，不依赖主流程当前的预处理实现。请指定新的输出目录；本备份不包含原始影像、缓存或 Python 环境。
