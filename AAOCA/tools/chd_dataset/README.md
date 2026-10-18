# CHD 成对研究影像批处理

Python 3.11+。本工具解压 AES ZIP 标注，核对成对 3D NIfTI，读完全部体素与 gzip 尾部，并以原文件字节复制到统一目录。没有重采样、归一化、配准、标签重映射或训练。未知物理单位保留为未知，文件名 class 不当作标签语义。

安装 `requirements.txt`，随后运行：

```powershell
python prepare.py --source 'D:/sample/images' --archive 'D:/sample/labels.zip' --output 'D:/sample/research' --workers 2
```

密码交互输入，不保存在程序或日志中；自动化可用 `--password-stdin` 通过标准输入提供。已解压完成后继续运行可省略 `--archive`，不需要再次提供密码。

已有本地研究项目可运行 `python prepare.py --project-root 'D:/sample/project' --workers 4`，从项目的 `project.local.json` 读取 `chd_reference_root`，输出到该项目的 `CHD/`。项目内 `继续整理CHD.cmd` 使用这一入口。开始新一轮检查前应先确认上次运行已经结束；运行中的输出目录受操作系统锁保护。

`--limit 3` 仅检查前 3 例，报告记录为小样本；省略 limit 为全量。默认复用版本、源文件大小/修改时间及输出文件签名都匹配的成功记录。`--rehash-resume` 进一步重算源/目标 SHA256；`--force` 重读全部体素。输出内容冲突时拒绝覆盖。

输出目录：

- `extracted_labels/`：原始标签解压文件及 `extraction.json` 中的压缩包成员/哈希。
- `native/images/CHD_<hash>_0000.nii.gz`、`native/labels/CHD_<hash>.nii.gz`：完整检查通过的原字节副本。研究编号由完整文件配对键生成，不等于患者身份。
- `review_required/images/`、`review_required/labels/`：文件像素完整但空间网格不一致的原字节副本，单独存放，禁止冒充已经对齐的训练对。程序不自动移动标签或改头。
- `records/`：逐例源路径、源/目标签名、SHA256、形状、空间变换、强度统计、标签值与体素数、问题记录。
- `manifest.json`、`配对清单.csv`、`问题清单.csv`：全量已检查明细；`inventory.json` 包括尚未检查项。
- `查看进度.html`：运行时自动刷新；`progress.json`、`summary.json` 为机器可读统计；`events.jsonl` 为追加日志。

空间网格一致仅表示头信息一致，不证明真实物理标定或解剖对应正确。标签为 0 的背景沿用现有数据编码假设；空前景、未知单位、缺少类别解剖说明均保留提示。标签负值、小数或超过65535会被拒绝；若数据源另有编码规范，应先核实后适配，不能直接取整。

中断后使用相同参数继续处理。文件在 staging 中完成复制和 SHA256 验证后才发布；操作系统锁防止同一输出目录并发执行，残留锁文件不表示程序仍运行。单例失败记录后继续。已有成果不符时保留并报告，不自动删改原始影像。

测试：`python -m unittest discover -s . -p test_prepare.py -v`，全部使用虚构小体积。真实资料、运行清单、密码和统计报告不得提交公开仓库。
