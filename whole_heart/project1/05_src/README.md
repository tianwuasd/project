# 后续代码约定

现有 `check_project.py` 是带中文注释的准备检查工具：核对 106 例清单、5 例隔离记录、原始路径存在性、既有审计与 skill 快照哈希、文档链接及 PDF 文件头。它不读取全量体素、不训练模型；结果写入 `00_admin/project_check.json`。在项目环境中运行 `python 05_src/check_project.py`。

方案已获准，现有 data_pipeline.py（标签转换与划分）、prepare_nnunet.py（来源专用指纹和预处理）、appearance.py（外观增强）、project_trainers.py（受控 B0/B1）、metrics.py（本地物理距离指标）、gpu_probe.py（合成输入显存检查）、smoke_train.py 和 run_smoke_suite.py（工程短测）、summarize_smoke.py（成本估计）。均带中文说明。

代码只读取原始数据；派生文件写到项目 04_data/derived。短测检查点位于 07_experiments/smoke_*，不能当成正式模型使用。当前入口只运行有限步数，完整研究训练将在预算确认后另行启动。配对统计和目标中心批量评价尚未执行。

项目环境下可运行 `python -B -X utf8 -m unittest discover -s tests -v` 检查实现。数据转换和预处理会拒绝覆盖已完成目录，勿重复执行准备命令。短测也会创建新的独立目录，不覆盖先前结果。
