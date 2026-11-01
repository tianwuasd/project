# Academic Agent 0.1.0 发行记录

日期：2026-09-08。发行标签：`academic-agent-v0.1.0`。应用目录：`academic_agent/`。

## 冻结证据

- 功能分支候选：`1fd7fc0`，核心最后修复：`9b8977f`。
- 本机完整验证：192 passed in 22.98s，Ruff 通过。
- 构建 sdist/wheel；从新 wheel 在独立环境安装并运行完整演示成功。
- [候选提交 CI](https://github.com/tianwuasd/testing_tools/actions/runs/34244701709)：Windows/Linux × Python 3.11/3.12/3.14 六组全部成功。
- 组件首轮 10 项缺陷修复并原审查者逐项确认；整体 4 项修复后限定复核全部 ADDRESSED，独立定向 45 passed in 17.80s。
- 真实 Codex 阅读/独立复核各 1 次完成；公共来源小查询结果、S2 限流和未做的学术评价均在 validation.md 披露。

## Git Flow

功能分支以非快进合并进入 develop，再从 develop 建立 release/academic-agent-0.1.0。发行分支合并 main 并回合 develop，main 上创建附注版本标签。发行内容以仓库实际分支和标签指向为准；没有强制推送。

未来选题验证应先建立真实领域语料和人工基准，再开发趋势统计、创新候选与正式全文写作。本版不把合成样例或测试通过数当作学术正确性评价。
