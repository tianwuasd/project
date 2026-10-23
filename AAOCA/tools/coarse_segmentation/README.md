# AAOCA 粗分割与独立服务器面板

使用 TotalSegmentator 2.18.0，为 AAOCA CT 生成心腔、大血管、冠脉的 **11 类粗标注**。包含命令行批处理和独立中文网页面板；不依赖资料整理的桌面 GUI，也不需要重新训练模型。

输入为已确认 HU 和真实毫米几何的原分辨率三维 NIfTI。**不要输入被截断并缩放至 `[-1,1]` 的训练预处理图像。** TotalSegmentator 自行执行模型所需的重采样和归一化；本工具不提前做一次额外插值，输出要求与输入网格一致。

## 分割内容

| 本项目标签 | 结构 | 模型任务 |
|---|---|---|
| 0 | 背景 | — |
| 1 / 2 | 左心室 / 右心室 | `heartchambers_highres` |
| 3 / 4 | 左心房 / 右心房 | `heartchambers_highres` |
| 5 | 心肌 | `heartchambers_highres` |
| 6 | 主动脉 | `heartchambers_highres` |
| 7 | 肺动脉 | `heartchambers_highres` |
| 8 | 肺静脉 | `total` |
| 9 / 10 | 上腔静脉 / 下腔静脉 | `total` |
| 11 | 冠脉整体 | `coronary_arteries` |

这是本工具自定义的标签编号，不应直接套用其他论文或数据集的编号。冠脉不细分 LAD/LCx/RCA，不自动判断异常开口、壁内段或风险。通用预训练模型在先心畸形上的质量必须逐例核查。

## 推荐：一体化服务器面板

无需提前安装模型依赖，在 Linux 服务器的本目录执行（Python 3.11–3.13）：

```bash
python3 server.py --workspace /srv/private/aaoca-seg
```

默认脱离终端在后台运行，显示面板地址和私有访问令牌。通过 SSH / VS Code 转发 8787 端口，在网页内依次完成：**环境安装 → 许可验证 → 清单导入与设备设置 → 选少量体积预检 → 粗分割**。安装和分割在独立后台进程执行，SSH 或网页断开后继续；重连仍能查询日志和请求停止。

可用 `--profile /srv/private/server.private.json` 导入私人配置。普通设置和许可分开保存；许可通过 stdin 传给官方验证函数，不写入命令行、公开代码或 API 返回。网页只接受固定操作，不接受任意 shell 命令。

```bash
# 重连时查看地址与令牌；也可直接重复原启动命令
python3 server.py --workspace /srv/private/aaoca-seg --status
```

详细步骤、配置格式和恢复边界见 [服务器面板说明](docs/server.md)。旧 `run.py panel` 入口保留，仅适用于已准备环境和固定路径的手动运行。

## 手动安装（可选）

使用独立 Python 3.11 或以上环境。从本目录运行：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-inference.txt
python run.py doctor
```

Windows 激活命令为 `.venv\Scripts\Activate.ps1`。GPU 推理需要与服务器驱动兼容的 PyTorch；按 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/)选择，不能从本地测试推断远程 CUDA 可用。只运行预检、面板和合成测试时，安装 `requirements.txt` 即可，不必下载大模型。

`total` 开放使用；心腔和冠脉任务需要许可。到 [TotalSegmentator 官方入口](https://backend.totalsegmentator.com/license-academic/)申请免费非商业许可，在**运行模型的同一用户/环境**配置：

```bash
totalseg_set_license -l YOUR_LICENSE
```

密钥不写进本仓库或 CSV。可以通过 `TOTALSEG_HOME_DIR`、`TOTALSEG_WEIGHTS_PATH` 将配置和模型放在私人缓存目录。首次推理可能联网下载权重及验证许可；面板本身不上传影像。`doctor` 仅检查本地是否配置许可，不保证许可有效。

## 准备清单

使用已整理的研究 `CT总清单.json` 导出，只筛选 AAOCA 中已复制核验且记录为 HU/mm 的体积：

```bash
python run.py inventory --inventory /srv/private/research/CT总清单.json \
  --data-root /srv/private/research --manifest /srv/private/preseg/aaoca.csv
```

`--data-root` 用于把原清单中的 `.../AAOCA/...` 路径定位到服务器的研究根目录，保留其后相对路径；没有迁移时可省略。导出不会扫描历史输出目录，不包含姓名和身份字段。研究代号基于源路径散列生成，但 CSV 含私人文件路径，**仍不能公开**。不会覆盖已有清单。

也可参照 [manifest.example.csv](examples/manifest.example.csv) 手工准备：

```csv
case_id,image,spacing_confirmed,hu_confirmed
demo_001,/srv/private/aaoca/ct_001.nii.gz,true,true
```

相对路径以 CSV 所在目录为基准。确认字段须有 DICOM/转换记录等依据；不能为了通过检查而把未知间距或归一化图像标为 `true`。一个三维期相对应一行；代号不能重复，同一影像不能重复列入。

## 原固定配置面板（兼容入口）

此旧入口不管理依赖安装，也不提供重连后的任务接管。新部署请使用上面的 `server.py`。手动运行：

```bash
python run.py panel --manifest /srv/private/preseg/aaoca.csv \
  --output-root /srv/private/preseg/results --device gpu:0 --port 8787
```

面板固定绑定 `127.0.0.1`，启动后显示访问令牌。通过 SSH 转发，在自己的电脑浏览器打开：

```bash
ssh -N -L 8787:127.0.0.1:8787 USER@SERVER
```

访问 `http://127.0.0.1:8787`，填入令牌。支持病例筛选/多选、只预检、开始、停止本面板的任务、阶段和计数、模型日志、GPU 查询。网页不修改路径或 GPU 可见范围，不提供任意命令接口。首轮建议选 3 例，复核后再扩大范围。

## 命令行批处理

```bash
# 只读文件头，不下载模型、不创建分割结果
python run.py plan --manifest /srv/private/preseg/aaoca.csv \
  --output-root /srv/private/preseg/results --limit 3

# 实际推理；默认逐例串行使用一张获准使用的可见 GPU
python run.py run --manifest /srv/private/preseg/aaoca.csv \
  --output-root /srv/private/preseg/results --device gpu:0 --limit 3
```

省略 `--limit` 会处理整份清单。可重复传入 `--case-id` 指定体积，`--device cpu` 使用 CPU；`--threads` 控制 CPU 重采样和保存线程，默认 2。无 fast 模式、不删小连通域、不自动补血管连接。

## 输出与人工复核

```text
results/
  status.json                       # 本次批次计数和阶段
  panel/                            # 面板提交的冻结清单、日志、停止请求
  cases/<研究代号>/
    latest.json                     # 最近一次完整成功的结果记录
    attempt-<时间>-<随机后缀>/
      total/*.nii.gz                # 三种静脉的原始二值掩膜
      heartchambers_highres/*.nii.gz # 七种心脏结构的二值掩膜
      coronary_arteries/*.nii.gz    # 冠脉整体二值掩膜
      aaoca_11_labels.nii.gz        # 合并图
      labels.ctbl                  # Slicer 颜色/名称表
      qc.json                      # 空掩膜、体素数、体积和重叠记录
      record.json                  # 输入、代码、环境、输出哈希及状态
      <task>.log / <task>.command.json
```

合并图重叠优先级为：冠脉 > 心腔/心肌/动脉 > 静脉；同组顺序在 `qc.json` 中明确列出。优先级只是合并规则，不代表真实解剖判断。原始二值掩膜完整保留，重叠不会从它们中删除。空冠脉等情况会记录待复核，而不是当作解剖结构不存在。

在 3D Slicer 中加载原始 CT，加载 `labels.ctbl` 为颜色表，导入合并 NIfTI 为 LabelMap 后转换为 Segmentation，并选择此颜色表；也可直接导入各二值掩膜。在 Segment Editor 中修订，另存人工审核版本。重点复核开口、主动脉根部附近连接、近端走行、邻近肺动脉的边界和细小分支。程序的 QC 不计算准确率，不替代人工验证。

重启后完整病例只有在输入 SHA256、程序源码、依赖版本、设置及全部结果文件哈希一致时才复用。中断病例从新 attempt 重跑三个任务，不声称从神经网络内部断点继续。旧结果不会覆盖；原始 CT 不复制、不改写。

## 验证范围

```bash
python -m unittest discover -s tests -v
```

测试仅使用合成 CT、合成分割和模拟推理后端，覆盖几何、合并、哈希复用、失败隔离、中断、面板 HTTP 和访问控制。已做本地软件验证；真实模型权重推理、Linux GPU 驱动与 AAOCA 标注质量须在实际服务器上验证。没有把软件测试描述为患者分割完成。

Windows 可将 `TEMP` 和 `TMP` 设置为可写短目录后运行测试。

## 来源及许可边界

- [TotalSegmentator 官方项目](https://github.com/wasserth/TotalSegmentator) / [类别字典](https://github.com/wasserth/TotalSegmentator/blob/master/totalsegmentator/map_to_binary.py)。
- [TotalSegmentator 2.18.0](https://pypi.org/project/TotalSegmentator/2.18.0/) 为本工具固定依赖；许可证及权重分发条件遵循上游。
- 使用研究成果时请按上游要求引用 TotalSegmentator 和 nnU-Net。仓库不分发模型权重和许可密钥。
- [设计说明](docs/design.md) / [实施计划](docs/implementation-plan.md)。
