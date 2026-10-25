"""Create the local handoff bundle from this conversation's existing artifacts."""
from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DISCUSSION = Path(r'D:\BaiduSyncdisk\obsidian\仓库\小张的书桌\科研\文献信息\研究构思\VISTA3D_选择性纠错与跨中心学生训练')
APP = Path(r'D:\projects\testing_tools\academic_agent')
SOURCE_PROJECT = APP / 'projects/vista_expert_clicks_20261025'
SNAPSHOT = ROOT / 'research/academic_agent_snapshot'
VERSION = 'v-2026-10-25T054638-996790-0000-0128989e'
EXPORT = SNAPSHOT / 'exports' / VERSION
COPIES = []


def lp(p):
    s = str(p)
    return Path(s if s.startswith('\\\\?\\') else '\\\\?\\' + s)


def plain(p):
    return str(p).removeprefix('\\\\?\\')


def read(p):
    return lp(p).read_text(encoding='utf-8-sig')


def write(rel, value):
    p = ROOT / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(value.strip() + '\n', encoding='utf-8')


def jswrite(rel, value):
    write(rel, json.dumps(value, ensure_ascii=False, indent=2))


def sha(p):
    h = hashlib.sha256()
    with lp(p).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def copy(source, dest):
    source, dest = lp(source), lp(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)
    a, b = sha(source), sha(dest)
    if a != b:
        raise RuntimeError(f'Copy mismatch: {plain(dest)}')
    COPIES.append({'source': plain(source), 'destination': Path(plain(dest)).relative_to(ROOT).as_posix(), 'sha256': b, 'bytes': dest.stat().st_size})


if (ROOT / '.codex/STATE.json').exists():
    raise SystemExit('This bundle already exists. Do not overwrite subsequent project work.')

for rel in ['.codex', 'docs', 'literature/papers', 'archive/discussion', 'archive/local_notes', 'research', 'experiments']:
    (ROOT / rel).mkdir(parents=True, exist_ok=True)

# Preserve all existing discussion files without changing the originals.
discussion_files = []
for p in DISCUSSION.iterdir():
    if not lp(p).is_file():
        continue
    name = 'Consensus_历史检索报告.md' if p.name.startswith('请用中文') else p.name
    dest = ROOT / 'archive/discussion' / name
    copy(p, dest)
    discussion_files.append({'original_name': p.name, 'archived_path': dest.relative_to(ROOT).as_posix()})

# Snapshot the complete research project. SQLite backup produces a consistent copy.
for p in SOURCE_PROJECT.rglob('*'):
    if not p.is_file() or '.runs' in p.relative_to(SOURCE_PROJECT).parts:
        continue
    if p.name in {'research.db', 'research.db-wal', 'research.db-shm'}:
        continue
    copy(p, SNAPSHOT / p.relative_to(SOURCE_PROJECT))
with sqlite3.connect(SOURCE_PROJECT.joinpath('research.db').as_uri() + '?mode=ro', uri=True) as source_db:
    with sqlite3.connect(SNAPSHOT / 'research.db') as target_db:
        source_db.backup(target_db)
        assert target_db.execute('pragma integrity_check').fetchone()[0] == 'ok'
COPIES.append({'source': str(SOURCE_PROJECT / 'research.db'), 'destination': 'research/academic_agent_snapshot/research.db', 'sha256': sha(SNAPSHOT / 'research.db'), 'method': 'sqlite_online_backup', 'bytes': (SNAPSHOT / 'research.db').stat().st_size})

run = json.loads(read(SOURCE_PROJECT / 'run-manifest.json'))
local = json.loads(read(DISCUSSION / '本地证据.json'))
historical_search = json.loads(read(DISCUSSION / '跨领域检索记录_2026-10-25.json'))
library = json.loads(read(EXPORT / 'library.json'))
work_lookup = {v['id']: v for v in library}
selected_by_title = {v['title']: v for v in run['documents']}
records = []

aliases = {
    'L1': 'VISTA3D', 'L2': 'REDNet', 'L3': 'Deep_Closing', 'L4': 'Learned_Iterative_CHD',
    'L5': 'Structure_Self_Supervision', 'L6': 'Variational_UDA_Cardiac',
    'L7': 'SDF4CHD', 'L8': 'NETracer', 'L9': 'Hierarchical_CMR',
}
for entry in local:
    record_id = entry['document_id'] + '_' + aliases[entry['document_id']]
    folder = ROOT / 'literature/papers' / record_id
    source_pdf = Path(entry['path'])
    copy(source_pdf, folder / 'paper.pdf')
    assert sha(folder / 'paper.pdf').lower() == entry['sha256'].lower()
    notes = []
    for p in lp(source_pdf.parent).iterdir():
        if not p.is_file() or p.suffix.lower() != '.md':
            continue
        short_name = '精读.md' if p.stem.endswith('_精读') else ('批注.md' if p.stem.endswith('_批注') else p.name)
        copy(p, ROOT / 'archive/local_notes' / record_id / short_name)
        s = read(p)
        s = s.replace(source_pdf.name, 'paper.pdf')
        s = re.sub(r'!?\[\[paper\.pdf(?:#([^\]|]+))?(?:\|[^\]]+)?\]\]', lambda m: '[原文 PDF](paper.pdf' + ('#' + m.group(1) if m.group(1) else '') + ')', s)
        write(str((folder / short_name).relative_to(ROOT)), s)
        notes.append((folder / short_name).relative_to(ROOT).as_posix())
    doc = selected_by_title.get(entry['title'])
    records.append({
        'id': record_id, 'title': entry['title'], 'group': '本地文献',
        'pdf': (folder / 'paper.pdf').relative_to(ROOT).as_posix(), 'notes': notes,
        'source_path': entry['path'], 'sha256': sha(folder / 'paper.pdf'),
        'reading_scope': 'Academic Agent 全文阅读及独立会话复核；另保留原精读/批注' if doc else entry['reading_scope'] + '；另保留既有笔记，未计入本轮 12 篇全文',
        'app_fulltext_reviewed': bool(doc), 'work_id': doc['work_id'] if doc else None,
        'urls': [], 'metadata_status': '本地书目名称原样保留',
    })

remote_specs = [
    ('R01_DnR', 'work_1e05e2b0eb6c452c', 'https://arxiv.org/html/2511.11005v3', '视觉专家意见转成提示，支持先预测再修正；视觉利用度是代理指标。'),
    ('R02_SegAgent', 'work_aa6dce0ed7f14b25', 'https://arxiv.org/html/2503.08625v1', '学习点击轨迹及执行后评价；第一版采纳头需要明确不修改选项。'),
    ('R03_Two_Stage_L2D', 'work_5d1c00a2b3764ef3', 'https://papers.neurips.cc/paper_files/paper/2023/hash/0b17d256cf1fe1cc084922a8c6b565b7-Abstract-Conference.html', '先预测器后选择器；分类理论不是三维点击或跨中心保证。'),
    ('R04_CGS', 'work_6572d29cec464414', 'https://arxiv.org/html/2504.00862v1', '同源标注与未标注资料训练通才/专长分支；原作测试可丢弃专长分支。'),
    ('R05_DeferredSeg', 'work_750f4a75835240a3', 'https://arxiv.org/html/2604.12411v2', '像素级专家路由融合；实验专家信息条件与训练的学生不同。'),
    ('R06_NeuroRefiner', 'work_864fba4def554ce3', 'https://arxiv.org/html/2608.09636v1', '已有全局检查/局部修正/验证循环；神经元拓扑不可直接迁移到先心病。'),
    ('R07_U_CFR', 'work_dc3db188901f4579', 'https://arxiv.org/html/2607.20705v1', '不确定性与边界驱动自动伪点击；便宜而必要的对照。'),
    ('R08_Generalist_Specialist_RL', 'work_b8b1ef49ee17445d', 'https://arxiv.org/abs/2206.12984', '强化学习中专长反馈通才；原作主要在训练阶段协作。'),
]
for rid, wid, url, relevance in remote_specs:
    d = next(v for v in run['documents'] if v['work_id'] == wid)
    destination = ROOT / 'literature/papers' / rid / 'paper.pdf'
    copy(SOURCE_PROJECT / d['local_source_path'], destination)
    records.append({'id': rid, 'title': d['title'], 'group': '本轮在线全文', 'pdf': destination.relative_to(ROOT).as_posix(), 'notes': [], 'sha256': sha(destination), 'reading_scope': 'Academic Agent 全文阅读及独立会话复核；主会话另核对关键方法段', 'app_fulltext_reviewed': True, 'work_id': wid, 'urls': [url, d['source_pdf_url']], 'relevance': relevance, 'doi': d.get('doi'), 'arxiv_id': d.get('arxiv_id'), 'metadata_status': '题录年份可能为预印本年份，引用时核对具体版本'})

# Prior cross-domain sources retain their original bounded reading scopes.
for old in historical_search['sources']:
    if old['id'] in {'R1', 'R2'}:
        continue
    records.append({'id': 'H' + old['id'][1:].zfill(2), 'title': old['title'], 'group': '历史跨领域近邻', 'pdf': None, 'notes': [], 'urls': list(dict.fromkeys([old['url']] + ([old['method_url']] if old.get('method_url') else []))), 'reading_scope': old['scope'] + '；历史核查范围，本次归档未重新阅读', 'relevance': old['relation'], 'app_fulltext_reviewed': False, 'metadata_status': '沿用历史记录，未重新联网核验'})

supplemental = [
    ('S01', 'DualClickMed / Rethinking RoI Strategy in Interactive 3D Segmentation for Medical Images', 'https://link.springer.com/chapter/10.1007/978-3-032-23496-4_12', '历史出版社摘要及公开原文片段；未收齐全文'),
    ('S02', 'SAM-REF', 'https://arxiv.org/html/2408.11535v4', '历史核对方法 3.2–3.5；未归档全文 PDF'),
    ('S03', 'SAT3D / Segment any tumour', 'https://www.nature.com/articles/s41467-026-76531-2', '历史出版社/索引摘要和方法片段；完整页面访问受限'),
    ('S04', 'MedSAM-Agent', 'https://arxiv.org/html/2602.03320v1', '历史子任务核对方法 3.1–3.2 及奖励；未归档全文 PDF'),
    ('S05', 'SemiSAM+', 'https://arxiv.org/html/2502.20749v1', '历史原文 3.4–3.5、4.4；训练学生背景，非最终部署角色'),
    ('S06', 'PseudoClick', 'https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/2903_ECCV_2022_paper.php', '会议摘要及原文索引方法片段；不计入本轮 12 篇全文'),
    ('S07', 'CT-PromptSAM: Collaborative Prompting With Hybrid CNN–Transformer for Remote Sensing Semantic Segmentation', 'https://github.com/WangXin81/CT-PromptSAM/', '摘要和作者仓库；未取得全文，优先补核近邻'),
    ('S08', 'From Few-Shot Segmentation to Clinician-in-the-Loop Medical Image Analysis', 'https://arxiv.org/abs/2609.10001', '摘要/观点性背景，不构成实验验证'),
    ('S09', 'When More Experts Hurt: Underfitting in Multi-Expert Learning to Defer', 'https://arxiv.org/abs/2602.17144v1', '题录与摘要背景；未阅读全文'),
    ('C05', 'Evidence modeling for reliability learning and interpretable decision-making under multi-modality medical image segmentation', 'https://doi.org/10.1016/j.compmedimag.2024.102422', '历史 Consensus 输出中引用摘要；未独立全文复核'),
    ('C06', 'Evidence fusion with contextual discounting for multi-modality medical image segmentation', 'https://arxiv.org/abs/2206.11739', '历史 Consensus 输出中引用摘要；未独立全文复核'),
    ('C07', 'ProMISe: Prompt-Driven 3D Medical Image Segmentation Using Pretrained Image Foundation Models', 'https://doi.org/10.1109/ISBI56570.2024.10635207', '历史 Consensus 输出中引用摘要；未独立全文复核'),
    ('C08', 'Few-Shot Adaptation of Training-Free Foundation Model for 3D Medical Image Segmentation', 'https://arxiv.org/abs/2501.09138', '历史 Consensus 输出中引用摘要/引言；未独立全文复核'),
    ('C09', 'ENSAM: an efficient foundation model for interactive segmentation of 3D medical images', 'https://doi.org/10.1007/978-3-032-23496-4_2', '历史 Consensus 输出中引用摘要/正文；未独立全文复核'),
    ('C10', 'Iterative Segmentation from Limited Training Data: Applications to Congenital Heart Disease', 'https://doi.org/10.1007/978-3-030-00889-5_38', '历史 Consensus 输出中引用摘要；不要与较新的 Learned iterative CHD 全文混同'),
    ('C11', 'Adaptability of Vision Foundation Models for 3D Medical Image Segmentation', 'https://doi.org/10.1109/OJSP.2025.3650437', '历史 Consensus 输出中引用摘要/正文；未独立全文复核'),
    ('C12', 'HVSMR-2.0: A 3D cardiovascular MR dataset for whole-heart segmentation in congenital heart disease', 'https://doi.org/10.1038/s41597-024-03469-9', '历史 Consensus 输出中引用正文；未独立全文复核'),
]
for rid, title, url, scope in supplemental:
    records.append({'id': rid, 'title': title, 'group': '补充与历史提及', 'pdf': None, 'notes': [], 'urls': [url], 'reading_scope': scope, 'app_fulltext_reviewed': False, 'metadata_status': '只保留已有记录，不补写未核实作者/年份/细节'})

jswrite('literature/catalog.json', records)
with (ROOT / 'literature/catalog.csv').open('w', encoding='utf-8-sig', newline='') as f:
    fields = ['id', 'title', 'group', 'pdf', 'app_fulltext_reviewed', 'reading_scope', 'urls']
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for v in records:
        w.writerow({k: ' | '.join(v.get(k, [])) if k == 'urls' else v.get(k, '') for k in fields})

index = ['# 文献总索引', '', '这里收齐本轮研究报告、历史讨论和历史检索输出中明确列出的论文。完整检索库的 61 条候选记录另保留在学术 Agent 快照中，未纳入者不能当作已读文献。', '', '现有 PDF 共 17 篇：本地 9 篇 + 本轮在线取得 8 篇。其中 12 篇经过本轮 Academic Agent 全文阅读与复核；另外 5 篇保留原有定向阅读范围。模型复核不是人工准确率验证。没有 PDF 的文献保留入口与证据范围，本次归档没有新下载或重新核查。', '', '| ID | 文献 | 已归档材料 | 阅读范围 |', '|---|---|---|---|']
for r in records:
    title = '[' + r['title'] + '](' + r['urls'][0] + ')' if r['urls'] else r['title']
    links = ['[PDF](' + r['pdf'].removeprefix('literature/') + ')'] if r['pdf'] else ['链接/历史记录']
    for n in r['notes']:
        links.append('[' + Path(n).stem + '](' + n.removeprefix('literature/') + ')')
    index.append('| ' + ' | '.join([r['id'], title, ' · '.join(links), r['reading_scope'].replace('|', '/')]) + ' |')
index += ['', '## 使用入口', '', f'- [12 篇全文阅读卡](../research/academic_agent_snapshot/exports/{VERSION}/reading-cards.md)', f'- [比较矩阵](../research/academic_agent_snapshot/exports/{VERSION}/comparison.md)', f'- [完整检索库 BibTeX](../research/academic_agent_snapshot/exports/{VERSION}/references.bib)', f'- [证据台账](../research/academic_agent_snapshot/exports/{VERSION}/evidence-ledger.json)', '- [结构化目录](catalog.json)', '', '文献文件为本地研究资料。发表前按最终版本复核作者、年份、页码和 DOI；不能将历史摘要核查描述为全文复核。']
write('literature/README.md', '\n'.join(index))

# A readable current report uses paths within the new project. Original bytes remain archived.
report = read(DISCUSSION / '通用学术Agent实跑_近邻与模型框架_2026-10-25.md')
report = report.replace('D:/projects/testing_tools/academic_agent/projects/vista_expert_clicks_20261025/exports/', '../research/academic_agent_snapshot/exports/')
report = report.replace('同目录的 `通用学术Agent运行记录_2026-10-25.json`', '[归档运行记录](../archive/discussion/通用学术Agent运行记录_2026-10-25.json)')
write('docs/research_framework.md', report)

write('README.md', f'''# 全局老师与局部学生：VISTA3D 选择性点击修正

创建日期：2026-10-25。阶段：文献研究与框架设计完成；尚未实现或训练分割系统。

## 研究问题

来源中心有少量标注 CT 和额外未标注 CT。老师与学生学习相同的当前来源材料，老师关注整体，学生分别强化局部结构。在目标中心预测时，学生继续提出局部分割意见；老师系统判断是否采纳，将有益意见转成三维正负点击，通过 VISTA3D 交互分支更新老师掩膜。最终输出是修正后的老师分割。

## 从这里阅读

1. [交接说明](.codex/HANDOFF.md)：当前状态、不可丢失的用户意图、下一步。
2. [对话整理](docs/conversation_context.md)：需求演变与关键原话，非完整逐字聊天导出。
3. [最新研究方案](docs/research_framework.md)：相似工作、训练方法、架构取舍与创新边界。
4. [最小实验计划](experiments/PLAN.md)：互补性 → 点击可达收益 → 采纳能力 → 跨中心评价。
5. [文献总索引](literature/README.md)：{len(records)} 项明确提及的论文，17 篇 PDF、18 份已有笔记。
6. [Academic Agent 阅读卡](research/academic_agent_snapshot/exports/{VERSION}/reading-cards.md)及[比较表](research/academic_agent_snapshot/exports/{VERSION}/comparison.md)。

## 项目结构

```text
.codex/                  交接、决策、环境、下一步、状态和文件校验清单
docs/                    当前研究方案与对话整理
literature/              文献索引；17 篇 PDF；本地精读与批注
research/                Academic Agent 项目数据库、原文、证据及固定导出快照
experiments/              最小实验计划与待填写的数据条件表；没有虚构结果
archive/                 原讨论和原笔记的保真副本；包含已过时的历史解释
tools/                   本次归档脚本和只读完整性检查
AGENTS.md                后续助手的读取顺序与项目工作约定
```

## 已完成与未完成

- 实际使用 `D:/projects/testing_tools/academic_agent` 的 Academic Agent 0.4.0，已完成 12 篇全文阅读与复核。它与 `academic_tools` 中的 ARS 是不同项目。
- 本项目保存完整研究快照：61 条书目、12 篇阅读卡、195 条提取陈述、检索日志和证据。193 条支持、2 条部分支持是模型复核状态，不是人工准确率。
- 已将原有研究文件复制归档；原 Obsidian 文献库及原 Agent 项目仍保留。
- 尚未核定标签结构、病例数量、患者划分、硬件、目标中心评价集，也未运行 CT 推理或训练。
- 测试时固定参数是推荐的第一版协议；用户已确认的是目标中心 CT 用于预测。不要把建议提升成用户永久禁止其他协议。

## 后续入口

在这个目录继续工作时，先读 `AGENTS.md` 和 `.codex/HANDOFF.md`。优先核对已有数据说明与标签清单，再填写 [数据条件表](experiments/data_requirements.json)，随后执行互补性与点击收益实验。

本项目是本地目录，未创建远程仓库或注册 Codex 侧边栏项目。学术 Agent 软件与运行环境的现有位置、研究快照恢复方式见 [.codex/ENVIRONMENT.md](.codex/ENVIRONMENT.md)。
''')

write('AGENTS.md', '''# 项目工作约定

1. 开始时依次阅读 `.codex/HANDOFF.md`、`.codex/STATE.json`、`.codex/DECISIONS.md`；做研究前再读 `docs/research_framework.md` 与 `literature/README.md`。
2. 保留用户真实目标：同源训练、局部学生在测试时继续提出建议、老师选择采纳、点击修正老师、最终输出老师掩膜。不能自行退回“仅训练时伪标签蒸馏、最终只部署学生”的设定。
3. 区分用户确认、助手建议和待实验验证的假设。两名学生、固定参数、点击前收益预测是首版建议，不是已实施结果或不能修改的限制。
4. 当前没有模型实验结果。所有数值必须附真实实验记录；不能将学术 Agent 阅读复核计数当作模型性能或人工文献准确率。
5. 最新设计以 `docs/research_framework.md` 为准。`archive/` 为保真历史资料，其中一些角色解释、工具识别和访问诊断已过时；按 `.codex/DECISIONS.md` 处理冲突。
6. 使用学术 Agent 时定位 `D:/projects/testing_tools/academic_agent`，不要与 `D:/projects/project/academic_tools` 的 ARS 混淆。现有研究快照位于 `research/academic_agent_snapshot`；续做应复制到工作目录并保留固定快照。
7. 先恢复已有上下文，再询问缺失信息。常规本地整理和检查直接继续；没有必要重新请求用户已经给予的授权。本项目不授权自动开启长时间 GPU 训练或发布资料。
8. 分割数据、训练/验证划分与患者信息另行管理。本归档没有 CT 数据；后续不得从文献中的示例臆测本项目类别和数量。文献、数据库和私人讨论不默认公开上传。
9. 文献结论必须与阅读范围一致；保留摘要/片段/全文区别。早期资料不是当前自动核验结果。不要为凑新颖性而忽略 CGS、DnR、SegAgent、DeferredSeg、NeuroRefiner 等近邻。
10. 后续实质进展同步更新 `.codex/STATE.json`、`.codex/HANDOFF.md` 与 `.codex/DECISIONS.md`，注明哪些是用户确认、哪些仍是建议。保持 UTF-8，尽量使用项目内相对链接。
''')

write('.codex/HANDOFF.md', f'''# 接手说明

最后整理：2026-10-25。项目：`D:/projects/vista3d_local_expert_refinement`。

## 一分钟恢复

用户要做少标注、跨中心 CT 分割。老师和局部学生学习同一来源材料。测试时学生在专长区域提出意见，老师判断，认可的意见通过点击修改老师掩膜；最终部署结果是老师吸收意见后的分割。学生不是训练完就丢弃的辅助蒸馏分支。

用户已确认：来源少量标注 CT + 来源未标注 CT；目标中心有未标注 CT，用于预测。实际结构标签、数量、中心数、划分与外部评价集未知，不能假定已存在冠脉开口或肺静脉连接标签。

## 已完成

- 梳理既有文献和跨领域近邻，实际运行正确的 Academic Agent 0.4.0。
- 5 条定向查询，OpenAlex/arXiv 成功；Semantic Scholar 5 次均 429。61 条书目不是 61 篇已读独立研究。
- 12 篇全文阅读 + 独立 Codex 会话复核：4 篇来自本地、8 篇在线取得；195 条陈述中 193 支持、2 部分支持。没有人工准确率测试。
- 保存固定导出 `{VERSION}`。没有运行应用的自动创新/反例流水线；模型框架由主会话综合原文提出。
- 本次建立独立本地项目，归档 17 篇 PDF、18 份原精读/批注、全部旧研究文档和 Agent 项目快照。文献总索引 {len(records)} 项，另保存 61 条检索候选库。

## 当前设计建议

一个支持自动与交互分支的 VISTA3D 老师 + 两个局部学生 + 轻量采纳头。学生以局部高分辨率、类别加权和专长采样学习，同源未标注数据提供一致性监督。学生与老师的分歧产生带类别的正负三维候选点；采纳头预测实际执行后的净收益，无益则不修改。

采纳头训练信号来自来源中心有真值病例上的真实点击前后差值，不是点击体素的正确性。患者级交叉拟合/留出，固定执行器生成标签；若执行器后续改变，应重做或验证收益标签。第一版先单次点击，后扩展短轨迹。

固定参数在目标中心推理是建议实验协议，用户没有作出永远不允许测试时适配的决定。跨中心准确率需要独立标注评价集；仅未标注预测不构成效果验证。

## 下一步

1. 读 `experiments/PLAN.md`，核对可恢复的数据说明和标签清单，填 `experiments/data_requirements.json`，不要迁入患者数据。
2. 确定真实支持的两个专长结构、老师检查点及三维点/类别接口；先验证坐标与重采样映射。
3. 测学生互补性，再测候选的真实点击可达收益；无空间则先修训练或执行器。
4. 有空间后训练采纳头，与全拒绝、全接受、置信度筛选、概率融合和同预算老师局部细化比较。

## 未做事项

没有实现分割网络、加载 VISTA3D 权重、运行 CT 预测、启动训练或产生跨中心指标；没有定好标签预算和采纳阈值；没有完成全面新颖性查重。不要将建议当作这些工作已经完成。

## 易混淆点

- 初期“最终只部署学生”的理解已撤回，按用户后续澄清执行。
- 正确工具是 `testing_tools/academic_agent`；ARS 与历史 Consensus 资料仅作来源留痕。
- 原生 VISTA3D 不含已经训练好的学生建议采纳器；滑窗处理也不保证全体积上下文。
- 不能把“更连通/更正常”视为先心病分割正确。保留真实异常拓扑。
- 历史诊断中的“未运行正确 Agent”已被后续真实运行解决；不能再次按旧结论汇报。

## 导航

- [需求与澄清](../docs/conversation_context.md)
- [当前研究方案](../docs/research_framework.md)
- [决策与冲突处理](DECISIONS.md)
- [环境与恢复](ENVIRONMENT.md)
- [文献索引](../literature/README.md)
- [实验计划](../experiments/PLAN.md)
- [文件校验结果](ARCHIVE_VALIDATION.json)
''')

write('.codex/DECISIONS.md', '''# 决策记录与冲突处理

日期：2026-10-25。以下各项按证据来源标记；“建议”不是用户已拍板的模型选择。

| ID | 类型 | 内容 | 影响 |
|---|---|---|---|
| U01 | 用户确认 | 老师和学生学习相同来源材料；学生特化不同局部，老师专注整体 | 当前训练集相同不等于历史预训练资料完全相同 |
| U02 | 用户确认 | 测试时老师吸收学生意见，通过点击优化自己 | 最终输出为老师，学生测试时保留 |
| U03 | 用户确认 | 目标是少量标注条件下改善模型并解决跨中心问题 | 需要标注预算曲线与独立外部评价 |
| U04 | 用户确认 | 来源有少量标注和额外未标注 CT；目标 CT 作为预测数据 | 不默认把目标 CT 纳入训练 |
| U05 | 用户授权 | 结合本地已有文献，使用 project 的通用学术 Agent，不限医疗领域检索 | 已执行正确应用，保留实际运行证据 |
| U06 | 用户授权 | 新建项目，放入讨论、思路、文献与内容，并建立 .codex 交接 | 本次创建本地项目及归档 |
| P01 | 助手建议 | 首版一个老师、两个学生、单次点击 | 先验证机制，不是固定最终规模 |
| P02 | 助手建议 | 点击前学习预期净收益，提供不修改选项 | 另保留试点后评价与直接融合作强基线 |
| P03 | 助手建议 | 首版目标中心推理固定参数 | 为隔离机制；后续适配应单列协议 |
| P04 | 助手建议 | 先监督学习采纳器，再考虑多轮或强化学习 | 不因采用 Agent 类比而引入不必要的语言模型 |
| H01 | 待验证 | 专长学生存在可迁移局部优势 | 用老师错/学生对统计与等预算对照验证 |
| H02 | 待验证 | 点击能把正确建议转化成净改善 | 必须实际调用执行器，不能用点正确率代替 |
| H03 | 待验证 | 采纳器可在外部中心识别有害修改 | 源中心校准不能视为域外保证 |
| H04 | 待验证 | 点击比直接融合更好 | 必须比较同预算融合与局部高分辨率细化 |

## 历史内容的解释优先级

用户后续明确澄清 > 当前交接与最新研究方案 > 旧版框架 > 最初备忘录。历史文件保留原貌，不据它们恢复已撤回的最终只部署学生方案。

旧文档把“测试时参数冻结”写得像已确认条件；本项目纠正为推荐首版协议。旧 ARS 访问诊断和 Consensus 报告中的运行情况只对当时成立；后续正确 Academic Agent 已真实执行。不要把三种工具混写。

DnR 应按本轮取得的 v3（而非历史 v1）描述。DeferredSeg 保留 2026 v2 与正式元数据的时间差，不能未经复核将未来卷期当成当前已发表日期。NeuroRefiner 的会议归属为作者说明；若用于投稿参考文献应再核正式记录。

两项部分支持陈述不作为强结论：DnR 遮挡比率实现细节须保留自适应条件；Learned iterative CHD 的相关显著性结论须保留 severe 病例限定。
''')

write('.codex/NEXT_STEPS.md', '''# 下一步清单

- [ ] 读取已有项目说明，核对来源标注数、未标注数、中心数、标签名称和许可使用范围。
- [ ] 确认独立患者划分，避免同一患者切片、增强、点击轨迹跨训练/验证集。
- [ ] 确认可用 GPU、VISTA3D 检查点及自动/交互分支，不假定现成权重支持所有标签。
- [ ] 选择两个有真实标注的专长结构，制定同预算普通学生和老师局部高分辨率对照。
- [ ] 在非患者或授权开发样例上验证物理坐标、重采样和正负点击的类别语义。
- [ ] 实验 E1：学生互补性；E2：执行器实际点击收益上界。
- [ ] 若 E1/E2 有空间，再做 E3：收益预测采纳器；随后扩展短轨迹。
- [ ] 评估直接融合、试点后评价、自动伪点击等强基线。
- [ ] 锁定模型和阈值，在独立外部标注集验证跨中心；没有标签时仅记录预测与动作日志。
- [ ] 补核 CT-PromptSAM 全文及其他未收齐近邻，投稿前重新核对最近工作和正式元数据。

目前以上均未完成。文献归档完成不等于模型实现或数据审计完成。
''')

write('.codex/ENVIRONMENT.md', f'''# 环境、工具与恢复

## 本项目

- 根目录：`D:/projects/vista3d_local_expert_refinement`。
- 本轮只归档文档、论文和研究工具输出；没有患者 CT、模型权重或训练环境。
- `.codex` 保存项目交接，未写入全局 Codex 设置或全局长期记忆。
- 没有注册 Codex 侧边栏项目的可用工具；本地目录已经建立，可将该目录作为项目打开。

## 已实际运行的学术工具

- 应用：`D:/projects/testing_tools/academic_agent`，Academic Agent 0.4.0。
- GitHub：`https://github.com/tianwuasd/testing_tools`。
- 研究运行时提交：`922c702f37545051097f3dbeca965c8fce14ab89`。这是运行留痕，不代表以后远程最新状态。
- 原研究项目：`D:/projects/testing_tools/academic_agent/projects/vista_expert_clicks_20261025`。
- 本项目快照：`research/academic_agent_snapshot`；固定导出 `{VERSION}`。
- Python：`C:/Users/Lenovo/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe`，运行记录为 3.12.14。
- 应用本地依赖：应用目录的 `.runs/python-deps`；属于兼容范围 pip 安装，不是 uv.lock 精确复刻。
- 启动辅助：应用目录 `.runs/run_agent.py`；将 app/src 与 app/.runs/python-deps 加入 Python 搜索路径。
- 后端使用既有 Codex 登录，未设置新 API key；实际模型名称未由结果暴露，不得自行补写。
- 国外检索当时使用临时本地代理 `127.0.0.1:7897`；续跑前检查可达，不修改全局代理。

## 恢复规则

现有快照可直接阅读。数据库使用 SQLite backup API 生成一致副本，原始 artifacts 和 exports 保留相对布局，导出哈希未改写；历史 JSON 中的绝对来源路径保留用于溯源。

若继续文献研究，先复制快照到新的 `research/academic_agent_working`，再通过现有应用指定该项目目录。不要直接修改固定快照，也不要在原项目和新工作副本上同时编辑同一研究状态。应用文件与 Python 依赖没有复制进本项目；机器迁移时应按应用 README 重建环境。

可用下列只读入口检查本地快照（不启动论文检索或模型调用）：

```powershell
$env:PYTHONUTF8 = '1'
& 'C:/Users/Lenovo/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' 'D:/projects/testing_tools/academic_agent/.runs/run_agent.py' --project 'D:/projects/vista3d_local_expert_refinement/research/academic_agent_snapshot' status
```

上面的可恢复性由本次归档后实际状态检查记录验证；后续仍需检查路径是否存在。不要默认运行 `resume`，它可能触发新的模型调用。

## 历史问题

- ARS：`D:/projects/project/academic_tools`，与本应用不同。早期稀疏检出和缺失入口的诊断不再适用于正确应用。
- 两次在途阅读因添加文献触发版本保护而拒绝结果；固定语料后重跑成功，预算仍计入两次未知尝试。
- Semantic Scholar 的五次 429 是检索范围缺口；没有声称系统综述或完全查重。
- Windows 很长的中文/英文文献路径需使用长路径支持；整理副本采用短文件名。控制台乱码先检查 UTF-8 解码，不据显示认定文件损坏。
''')

write('docs/conversation_context.md', '''# 本轮对话整理

整理日期：2026-10-25。本文件保存当前可见用户要求、澄清和已完成工作，是结构化交接摘要，不是所有历史消息的逐字完整导出。详细原报告、检索输出和证据保存在 archive/ 与 research/。

## 思路怎样被明确

用户最初希望借助 VISTA3D 的点击修正：另一个模型给出不同结果，老师判断是否可能成立，可能成立才点击。随后强调结合已有文献、使用自己的通用学术 Agent，并扩展到非医疗领域寻找相似思路。

用户表达过“主要想用少量标注训练更好的学生模型”，随后作出决定性的角色澄清：

> 不是，我的意思其实是老师和学生学习相同材料，学生会特化不同地方的学习，老师专注于整体，在于测试时老师会根据学生不同地方的标注吸收他们的意见，通过点击的形式进行优化修正

因此，当前目标以这条澄清为准：学生必须有实际训练和局部专长，测试时继续参与；最终改善的是老师输出。早期“纠错生成伪标签、最终只部署学生”的解释已撤回。

## 已确认的数据与目标

- 希望利用少量标注，并解决跨中心问题。
- 能获得目标中心未标注 CT；用户进一步明确“目标中心未标注ct作为预测数据”。
- 来源中心还有未标注 CT，可以辅助来源训练。
- 尚未提供或核定结构标签、病例数量、中心数量、硬件和独立目标评价集。

## 工具需求与纠正

用户允许了解本地已有文献、动用 project 中的通用学术 Agent，要求找相似机制并思考模型搭建。过程中多次询问 Agent 使用问题、是否仅在 GitHub、要求拉到 project，并明确指向 GitHub testingtools。

此前混用了 ARS 的工作流快照和历史检索资料。最终确认正确工具为 `D:/projects/testing_tools/academic_agent`（0.4.0），其仓库为 tianwuasd/testing_tools。正确应用已实际完成检索、导入、筛选、PDF 附件、阅读、独立会话复核与导出；不能继续说它只是 GitHub 上的文件或还没有运行。

历史 ARS、Consensus 资料保留为过程记录；它们不等于正确应用这次的执行结果。用户没有要求把新项目公开发布，也没有要求本轮启动模型训练。

## 已形成的研究判断

文献分别已有通才/专才、专家选择、自动点击和全局检查/局部修改。CGS 最接近同源专长训练；DnR 接近专家提示反馈；SegAgent 接近点击及过程评价；DeferredSeg 接近局部路由；NeuroRefiner 接近检查—修改—验证循环。自动点击本身也已有 PseudoClick 等先例。

更具体的待验证问题是：同源少标注学生能否提供互补意见；这些意见经三维交互执行器能否改善老师；采纳头能否预测真实净收益；收益能否迁移到外部中心并胜过同预算融合。

## 当前方案与边界

推荐第一版：一个老师、两个局部学生、一个轻量采纳头，从单轮开始。学生通过局部高分辨率、采样和类别权重形成专长。来源未标注数据做一致性训练；有标注数据上的真实点击前后差值训练采纳器。候选生成不使用真值，真值只用于来源训练与评价。

测试冻结参数是助手为隔离机制提出的第一版协议，不是用户永久禁止其他适配路线。最终输出老师；没有有益候选可以不修改。真实异常解剖不能因“不像正常心脏”而被否定。

目前完成文献研究和方案建议，未完成分割实现或实验。先做学生互补性、点击可达收益、学习采纳三步，再扩大模型规模和外部评价。

## 本次归档请求

> 把目前这个对话的情况和思路，新建一个project，提到的文献，内容都放进去，设立.codex文件夹做好交接

执行结果见根 README 和 .codex/HANDOFF.md。原始讨论文件保真保留，当前文档明确纠正已过时的角色、工具身份和协议确认程度。
''')

write('experiments/PLAN.md', '''# 最小实验计划（尚未执行）

这是既有讨论的实施顺序，不代表已训练模型或已批准任何具体 GPU 预算。

## 数据与接口前置条件

先填写 data_requirements.json。按患者固定来源训练/验证划分，必要时交叉拟合。外部评价集独立锁定，不用它选择模型或阈值；目标 CT 当前用于预测。明确标签映射、VISTA3D 自动/交互检查点、点坐标变换及局部合并语义。学生 ROI 应覆盖老师完全漏检的位置。

## E1：专长互补

训练/适配整体老师和两个真实标签支持的局部学生，同一来源病例池与相同标注预算。学生以专长 Dice+CE、局部高分辨率采样和未标注一致性为起点；不强制所有位置复制老师，不奖励故意不同。

比较老师、专长学生、同容量普通学生、同预算老师高分辨率局部细化。统计老师错/学生对与老师对/学生错、误差相关性，按患者和结构报告。缺乏互补则先改学生学习重点。

## E2：建议经点击能否形成改善

固定老师/学生/执行器；根据预测分歧生成有类别、正负属性和三维坐标的候选。每个候选从相同初始状态单独试运行，计算实际净质量变化与对相邻结构的损害。只在来源分析中用真值选择最佳候选，作为上界，不能把这种选择用于实际目标推理。

若单步上界很低，先检查候选表达、坐标映射和交互执行器；必要时分析少量短序列，不直接投入大规模强化学习。

## E3：学习是否采纳

有标注来源数据提供 r(a)=Q(M_after,Y)-Q(M_before,Y)-lambda_h*damage-lambda_c*cost；不修改的收益为 0。监督采纳头回归收益并可辅助分类有害/无效/有益。真值不用来生成候选。患者级留出或交叉拟合，采纳阈值使用独立来源验证集。同一患者的所有增强、点和轨迹同组。

对照全拒绝、全接受、学生置信度阈值、学习收益。不能只报有害率；全拒绝也可能有零损害，必须同时报告有效修改覆盖、净收益和最终分割。

## E4：机制与成本消融

比较点击前预测收益、试点后比较新旧掩膜（含不修改）、概率/局部掩膜融合、老师自己局部高分辨率细化、无专长集成、便宜自动伪点击。匹配预训练资源、分辨率、模型数和推理预算。只有优于合理替代方法，才能支持点击与专长机制的增量。

## E5：少标签与跨中心

按实际可用规模设置多个来源标注预算，所有方法用同一患者划分。首版目标推理参数固定，仅更新掩膜与点击历史。锁定策略后在外部标注集评价；只有未标注 CT 时，只交付预测与行为日志，不计算或臆测 Dice。

## 记录指标

逐结构 Dice/NSD，逐患者变化及变差患者比例，相邻结构误改、异常连接损害，每点击净收益、接受率/停止率、点击数、耗时与峰值显存。小结构和连接指标应由真实标签定义。患者级不确定性估计，不能把体素当独立病例。每条点击日志记录患者匿名 ID、模型版本、类别、原始物理坐标与变换、学生身份、采纳分数、执行前后摘要和停止原因。

## 退出或改向条件

无互补则调整学生；有正确意见但无执行收益则修候选/执行器；学习采纳不如便宜阈值则检查标签与校准；融合更好则不能宣称点击必要；外院下降则跨中心假设尚未成立。这里没有预设数值结果或未经数据确定的成功阈值。
''')

jswrite('experiments/data_requirements.json', {
    'status': 'not_yet_audited',
    'source_labeled_ct_available': True, 'source_unlabeled_ct_available': True,
    'target_unlabeled_ct_available': True, 'target_use_confirmed_by_user': 'prediction',
    'source_labeled_patients': None, 'source_unlabeled_patients': None, 'source_centers': None,
    'target_centers': None, 'target_patients': None, 'labels_and_id_map': None,
    'patient_split_manifest': None, 'external_labeled_evaluation_available': None,
    'teacher_checkpoint': None, 'student_specialties': None, 'gpu_resources': None,
    'note': 'Availability comes from conversation, not a filesystem data audit. Do not insert patient identifiers here.'
})

write('archive/README.md', '''# 历史资料

discussion/ 保存本轮归档前研究目录中的所有 11 个文件，内容不改。过长的 Consensus 文件名改为短名，映射见 .codex/ARCHIVE_MANIFEST.json。local_notes/ 保存 9 篇本地文献的 18 份精读/批注原始内容；literature/papers 中的阅读副本仅把论文引用路径调整到本项目。

历史文件可能包含被后续用户澄清取代的角色定义、已修复的 Agent 访问问题、旧版本论文或较窄阅读范围。阅读优先级见 ../.codex/DECISIONS.md。保持它们作为来源记录，不覆盖原文件，不将历史状态误认作当前结论。
''')

write('research/README.md', f'''# 学术 Agent 研究快照

academic_agent_snapshot 是 2026-10-25 原研究项目的本地副本，包含 SQLite 一致备份、原文附件、证据、阅读结果、查询日志和固定导出。没有复制软件依赖、登录信息、API 密钥或患者 CT。

阅读入口：[阅读卡](academic_agent_snapshot/exports/{VERSION}/reading-cards.md)、[比较表](academic_agent_snapshot/exports/{VERSION}/comparison.md)、[综述工作稿](academic_agent_snapshot/exports/{VERSION}/review.md)、[质量记录](academic_agent_snapshot/exports/{VERSION}/quality.json)。

本轮共 61 条书目，12 篇全文完成分析；其余候选不等于已阅读全文。2 条部分支持陈述保留原复核结论，不人为改为支持。outline.md 是工具的简短提纲，不是本项目详细实施计划；实施计划在 experiments/PLAN.md。

快照用于留痕。继续研究请复制到新的工作目录，按 .codex/ENVIRONMENT.md 使用真实应用，不直接修改固定导出。原始绝对来源路径留在 JSON 中作溯源，常用阅读入口均可从本项目相对链接打开。
''')

state = {
    'schema_version': 1, 'updated_at': datetime.now(timezone.utc).isoformat(), 'timezone': 'Asia/Shanghai',
    'project_root': str(ROOT), 'title': '全局老师与局部学生：VISTA3D 选择性点击修正',
    'phase': 'literature_and_framework_ready_model_experiments_not_started',
    'confirmed': {'same_source_material': True, 'source_labeled_ct': True, 'source_unlabeled_ct': True, 'target_ct_use': 'prediction', 'student_role': 'local_specialization_and_test_time_advice', 'teacher_role': 'global_segmentation_and_selective_refinement', 'final_output': 'teacher_mask_after_accepted_student_clicks'},
    'proposed_not_user_fixed': {'student_count': 2, 'first_stage_clicks': 1, 'test_time_parameter_updates': False, 'gate': 'predicted_executor_specific_net_utility'},
    'model_experiments_executed': False, 'patient_data_copied': False,
    'actual_academic_agent': {'path': str(APP), 'version': '0.4.0', 'commit': run['commit'], 'original_project': str(SOURCE_PROJECT), 'snapshot': 'research/academic_agent_snapshot', 'export_version': VERSION, 'model_identity': 'not_exposed', 'automatic_innovation_pipeline_executed': False},
    'counts': {'mentioned_paper_records': len(records), 'unique_pdf_papers': 17, 'existing_note_files': 18, 'archived_discussion_files': len(discussion_files), 'app_bibliographic_records': 61, 'app_full_text_reads': 12, 'claims': 195, 'supported_by_model_review': 193, 'partial_by_model_review': 2},
    'gaps': ['label_names_and_ids', 'patient_counts_and_splits', 'source_center_count', 'teacher_checkpoint_and_click_interface', 'gpu_resources', 'external_labeled_evaluation', 'CT-PromptSAM_fulltext'],
    'next_action': 'Recover existing data documentation, then evaluate student complementarity and oracle analysis of executable click utility.',
    'app_sidebar_registered': False, 'remote_repository_created': False,
}
jswrite('.codex/STATE.json', state)
jswrite('.codex/ARCHIVE_MANIFEST.json', {'created_at': datetime.now(timezone.utc).isoformat(), 'discussion_source': str(DISCUSSION), 'agent_source': str(SOURCE_PROJECT), 'discussion_filename_mapping': discussion_files, 'copied_files': COPIES, 'integrity_note': 'All copy entries byte-checked against source. SQLite is a consistent backup and separately integrity-checked. Normalized reading copies are derived; originals preserved under archive/local_notes.'})
print(json.dumps({'project': str(ROOT), 'catalog_entries': len(records), 'pdf_papers': 17, 'local_notes': 18, 'discussion_files': len(discussion_files), 'copied_files': len(COPIES), 'status': 'created_needs_validation'}, ensure_ascii=False))
