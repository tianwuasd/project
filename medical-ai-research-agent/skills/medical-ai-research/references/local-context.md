# 本地配置与接续

项目根为仓库的 `medical-ai-research-agent` 目录；运行源码时自动定位。把技能单独安装到用户目录后，设置环境变量 `MEDICAL_AI_PROJECT` 为项目绝对路径。脚本也可用 `--db`、`--config`、`--output` 指定路径。

复制 `config/sources.example.json` 为 `config/sources.json`，将来源路径改为自己有权使用的论文目录。示例中的 `../papers` 相对于config文件所在目录。原始资料只读，数据库写入独立data目录。不要配置包含病历或患者影像的项目根目录。

公开包不含私人资料、论文全文、全文派生索引及本机研究记录。索引数值取决于用户配置；PDF可提取不表示已经审阅。同题目的不同PDF版本保留，需人工关联后再统计独立研究。

PDF读取需要Poppler的 `pdftotext` 在PATH，或通过 `PDFTOTEXT` 指定可执行文件。现有研究项目按用户提供的目录读取其README、AGENTS.md、研究协议和实验状态，确认后复用，不自动修改训练流程。

本包自带全部三阶段流程、研究模板和二分类实验工具；academic技能库只作为可选方法参考，不构成运行依赖。
