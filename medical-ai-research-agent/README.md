# 医疗 AI 论文研究 Agent

基于论文证据开展 **找场景 → 特征工程 → 算法与消融实验**。目前专精心脏CT/CMR、医学影像、跨中心泛化与可解释性，其他AI方向可调整领域配置。

由宿主AI完成研究推理，Python工具负责本地论文索引、页码证据、研究状态和受控实验；无需独立模型API。此目录是便于分享和复现的源码包，个人文献和机器配置未上传。

## 使用技能

让支持SKILL.md的AI助手读取 [技能入口](skills/medical-ai-research/SKILL.md)，或将 `skills/medical-ai-research` 安装至用户技能目录。独立安装后，把 `MEDICAL_AI_PROJECT` 环境变量指向本项目目录。

示例请求：

> 使用 $medical-ai-research，结合我的论文库，找三个医疗AI场景，选一个，设计特征字典、基线、消融和跨中心实验，并保存可继续的研究包。

可以只做一个阶段，或依据已有state.json继续。参见 [研究流程](skills/medical-ai-research/references/workflow.md)、[医疗方法](skills/medical-ai-research/references/medical.md)、[研究模板](skills/medical-ai-research/assets/study.md)。

## 安装与验证

开发验证环境为Python 3.14，依赖版本在requirements.txt。建议使用独立环境。PDF提取另需Poppler的pdftotext加入PATH（或设置PDFTOTEXT）。PyMuPDF用于测试PDF夹具。

```text
python -m pip install -r requirements.txt
python -X utf8 -m unittest discover -s tests -v
```

## 构建自己的论文库

复制 `config/sources.example.json` 为 `config/sources.json`，修改论文目录路径。若保留示例路径，先建立项目下的papers目录并放入有权使用的PDF。索引支持PDF、名称含“精读”的Markdown和JSON文献记录，保留原文、二手笔记、摘要与元数据的区别。

```text
python -X utf8 skills/medical-ai-research/scripts/corpus.py build --config config/sources.json
python -X utf8 skills/medical-ai-research/scripts/corpus.py stats
python -X utf8 skills/medical-ai-research/scripts/corpus.py search "心脏 跨中心 分割" --limit 8
python -X utf8 skills/medical-ai-research/scripts/corpus.py read DOCUMENT_ID --page 1
python -X utf8 skills/medical-ai-research/scripts/new_study.py "我的研究" --output studies/my-study
```

全局参数 `--db` 放在子命令之前。输出数据库须位于原始资料目录之外。SHA256相同文档去重并保留多个来源；不同版本不自动认定为独立研究。检索是中英文词汇匹配，PDF解析可用不代表完整精读。

## 可运行实验与结果

```text
python -X utf8 skills/medical-ai-research/scripts/make_demo.py --output examples/synthetic/new_input
python -X utf8 skills/medical-ai-research/scripts/experiment.py --data examples/synthetic/new_input/features.csv --manifest examples/synthetic/new_input/manifest.json --output examples/synthetic/new_run
```

输出使用新目录，不覆盖既有研究。自带演示含240名模拟患者、480条模拟扫描记录，3组特征×3类模型。数据划分为120/40/80名训练/验证/测试患者，模拟中心C留出。

查看 [完整结果](examples/synthetic/run_42_verified/results.json)、[划分](examples/synthetic/run_42_verified/split.json)、[患者预测](examples/synthetic/run_42_verified/predictions.csv)和[特征声明](examples/synthetic/input/manifest.json)。示例只验证工程流程，不是医学性能证据。

## 能力边界

实验器支持数值特征的患者级单一二分类，含训练内插补/标准化、验证选参、患者聚合和AUROC bootstrap区间。字段可用时间由manifest声明，运行前仍须核查原始时间戳与数据来源。未来变量、矛盾结局、跨院重复患者与重复CSV表头会按契约拦截。

三维分割、多标签、生存或语言模型实验需要按任务接入代码；固定单次划分不替代正式研究验证。阅读材料、研究假设和执行结果分层记录，真实数据的授权、脱敏、标签、资源预算由具体项目确认。

[验证记录](docs/verification.md) · [实现结构](docs/architecture.md) · [方法来源](skills/medical-ai-research/references/source-map.md)
