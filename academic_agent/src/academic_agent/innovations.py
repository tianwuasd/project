"""Service orchestration and host-authored innovation publications."""

from __future__ import annotations

import json
from pathlib import Path

from .bibliography import export_bibtex
from .challenges import (
    candidate_currentness,
    candidate_version_currentness,
    load_candidate,
    load_candidate_version,
    read_challenges,
)
from .exports import escape_md, safe_url
from .innovation_analysis import (
    ASSESSMENT_DIMENSIONS,
    InnovationError,
    assessment_host_fields,
    assessment_input_hash,
    assessment_units,
    generate_assessment,
    generate_candidates,
    review_assessment,
    review_candidates,
    validate_assessment,
    validate_assessment_review,
)
from .innovation_models import (
    AssessmentContext,
    AssessmentReview,
    CandidateBatch,
    GroundingReview,
    InnovationAssessment,
    InnovationCandidate,
)
from .research_artifacts import _check_history, load_research, publish_research
from .research_context import (
    ResearchSnapshot,
    assert_research_current,
    diagnose_research_state,
    freeze_research,
    read_research,
    research_fingerprint,
)
from .store import BudgetExceeded, StaleJobError
from .workflow import Runner


class _StaleResearch(ValueError):
    pass


def _current(runner: Runner, snapshot: ResearchSnapshot) -> None:
    try:
        assert_research_current(runner.store, snapshot)
    except ValueError as exc:
        raise _StaleResearch("Research inputs changed during innovation processing") from exc


def _json_bytes(value) -> bytes:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode("utf-8")


def _evidence_appendix(snapshot: ResearchSnapshot, batch: CandidateBatch,
                       review: GroundingReview) -> dict:
    claim_ids = {claim_id for candidate in batch.candidates
                 for claim_id in candidate.closest_claim_ids + candidate.grounding_claim_ids}
    claim_ids.update(claim_id for item in review.items for claim_id in item.claim_ids)
    evidence_ids = {evidence_id for candidate in batch.candidates
                    for evidence_id in candidate.grounding_evidence_ids}
    evidence_ids.update(evidence_id for item in review.items for evidence_id in item.evidence_ids)
    return {
        **_reference_appendix(snapshot, batch.input_hash, claim_ids, evidence_ids),
        "candidate_references": [{
            "candidate_id": candidate.id,
            "closest_work_ids": candidate.closest_work_ids,
            "closest_claim_ids": candidate.closest_claim_ids,
            "grounding_claim_ids": candidate.grounding_claim_ids,
            "grounding_evidence_ids": candidate.grounding_evidence_ids,
        } for candidate in batch.candidates],
        "review_references": [item.model_dump(mode="json") for item in review.items],
    }


def _reference_appendix(snapshot: ResearchSnapshot, input_hash: str,
                        claim_ids: set[str], evidence_ids: set[str]) -> dict:
    claims = {claim.id: claim for card in snapshot.cards.values() for claim in card.claims}
    for claim_id in claim_ids:
        evidence_ids.update(claims[claim_id].evidence_ids)
        evidence_ids.update(snapshot.decisions[claim_id].evidence_ids)
    work_ids = {claims[claim_id].work_id for claim_id in claim_ids}
    return {
        "schema_version": 1,
        "input_hash": input_hash,
        "works": {work_id: snapshot.works[work_id].model_dump(mode="json") for work_id in sorted(work_ids)},
        "claims": {claim_id: claims[claim_id].model_dump(mode="json") for claim_id in sorted(claim_ids)},
        "evidence": {evidence_id: snapshot.evidence[evidence_id].model_dump(mode="json")
                     for evidence_id in sorted(evidence_ids)},
        "decisions": {claim_id: snapshot.decisions[claim_id].model_dump(mode="json")
                      for claim_id in sorted(claim_ids)},
    }


def _report(snapshot: ResearchSnapshot, batch: CandidateBatch, review: GroundingReview) -> str:
    reviews = {item.candidate_id: item for item in review.items}
    lines = [
        "# 创新候选（本次材料范围）\n\n",
        "本报告记录系统提出的可证伪研究候选。独立审查只判断候选对所给材料的引述是否有根据；",
        "它不证明全球首次、新颖性或尚未执行实验的成功。作者自述贡献仍属于来源陈述，",
        "系统假设不是论文已经观察到的结果，所有候选仍需人工学术判断。\n\n",
        f"输入指纹：`{batch.input_hash}`。所选材料：{len(snapshot.work_ids)} 篇。\n\n",
    ]
    for index, candidate in enumerate(batch.candidates, 1):
        item = reviews[candidate.id]
        titles = "；".join(
            f"{escape_md(snapshot.works[work_id].title)} (`{snapshot.works[work_id].citekey}`)"
            for work_id in candidate.closest_work_ids
        )
        lines.extend([
            f"## 候选 {index} · `{candidate.id}`\n\n",
            f"研究问题：{escape_md(candidate.question)}\n\n",
            f"最接近工作：{titles}\n\n",
            f"提出的具体差异：{escape_md(candidate.claimed_difference)}\n\n",
            f"可证伪假设：{escape_md(candidate.falsifiable_hypothesis)}\n\n",
            f"最低验证：{escape_md(candidate.minimum_validation)}\n\n",
            f"数据条件：{escape_md(candidate.data_requirements)}\n\n",
            f"算力条件：{escape_md(candidate.compute_requirements)}\n\n",
            "可能反例：" + escape_md("；".join(candidate.possible_counterexamples)) + "\n\n",
            "风险：" + escape_md("；".join(candidate.risks)) + "\n\n",
            "定向查询：" + escape_md("；".join(candidate.targeted_queries)) + "\n\n",
            f"材料截止：{escape_md(candidate.material_cutoff)}\n\n",
            f"独立依据审查：**{item.verdict}**。{escape_md(item.rationale)}\n\n",
            "引用链见 `evidence.json`；宿主生成的参考文献见 `references.bib`。\n\n",
        ])
    if snapshot.gaps:
        lines.append("## 当前材料缺口\n\n")
        for gap in snapshot.gaps:
            prefix = f"`{gap.work_id}`：" if gap.work_id else "项目："
            lines.append(f"- {prefix}{escape_md(gap.reason)}\n")
    return "".join(lines)


def _payloads(snapshot: ResearchSnapshot, batch: CandidateBatch,
              review: GroundingReview) -> dict[str, bytes]:
    works = [snapshot.works[work_id] for work_id in snapshot.work_ids]
    return {
        "candidates.json": _json_bytes(batch),
        "grounding-review.json": _json_bytes(review),
        "report.md": _report(snapshot, batch, review).encode("utf-8"),
        "references.bib": export_bibtex(works).encode("utf-8"),
        "evidence.json": _json_bytes(_evidence_appendix(snapshot, batch, review)),
    }


def _result(status: str, input_hash: str = "", batch: CandidateBatch | None = None,
            output: Path | None = None, gaps: list | None = None) -> dict:
    return {
        "status": status,
        "input_hash": input_hash,
        "output": str(output) if output else None,
        "candidate_ids": [candidate.id for candidate in batch.candidates] if batch else [],
        "gaps": gaps or [],
    }


def create_innovations(runner: Runner, *, work_ids: list[str] | None = None) -> dict:
    """Generate, independently review, and publish a current candidate batch."""
    snapshot = None
    batch = None
    input_hash = ""
    gaps = []
    try:
        snapshot = freeze_research(runner.store, work_ids=work_ids, require_supported=True)
        input_hash = research_hash = research_fingerprint(snapshot)
        gaps = [gap.model_dump(mode="json") for gap in snapshot.gaps]
        with runner.backend.validated() as backend:
            batch = generate_candidates(
                snapshot,
                backend,
                timeout_seconds=runner.config.timeout_seconds,
                max_input_chars=runner.config.max_input_chars,
            )
            _current(runner, snapshot)
        with runner.backend.validated() as backend:
            review = review_candidates(
                snapshot,
                batch,
                backend,
                timeout_seconds=runner.config.timeout_seconds,
                max_input_chars=runner.config.max_input_chars,
            )
            _current(runner, snapshot)
        _current(runner, snapshot)
        try:
            output = publish_research(runner.store, "innovation", snapshot, _payloads(snapshot, batch, review))
        except ValueError as exc:
            _current(runner, snapshot)
            raise exc
        return _result("complete", research_hash, batch, output, gaps)
    except BudgetExceeded:
        return _result("budget_paused", input_hash if snapshot else "", batch,
                       gaps=[*gaps, {"reason": "model_budget_exhausted"}])
    except InnovationError as exc:
        status = "needs_input" if exc.status == "needs_input" else "failed"
        if exc.status == "input_too_large":
            reason = {
                "reason": "model_input_too_large",
                "guidance": "Choose fewer works with work_ids and retry; selected documents remain complete.",
            }
        else:
            reason = {"reason": "model_input_required" if status == "needs_input"
                      else "model_artifact_invalid"}
        return _result(status, input_hash if snapshot else "", batch,
                       gaps=[*gaps, reason])
    except (_StaleResearch, StaleJobError):
        return _result("stale", input_hash if snapshot else "", batch,
                       gaps=[*gaps, {"reason": "research_inputs_changed"}])
    except (OSError, ValueError, KeyError, TypeError):
        return _result("failed", input_hash if snapshot else "", batch,
                       gaps=[*gaps, {"reason": "research_input_invalid"}])


def _assessment_context(root: Path, candidate_id: str, manifest_hash: str) -> AssessmentContext:
    premise = candidate_currentness(root, candidate_id)
    coverage = read_challenges(root, candidate_id)
    return AssessmentContext(candidate_manifest_hash=manifest_hash,
                             candidate_current=premise['current'], premise_gaps=premise['gaps'],
                             rounds=coverage['rounds'], hits=coverage['hits'],
                             unresolved_tasks=coverage['unresolved_tasks'], coverage_status=coverage['status'])


def _assessment_current(runner, snapshot, candidate, context):
    _current(runner, snapshot)
    try:
        now = _assessment_context(runner.store.root, candidate.id, context.candidate_manifest_hash)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise _StaleResearch('Assessment dependencies became unreadable') from exc
    if assessment_input_hash(snapshot, candidate, now) != assessment_input_hash(snapshot, candidate, context):
        raise _StaleResearch('Candidate or challenge coverage changed during assessment')


def _assessment_payloads(snapshot, candidate, context, assessment, review):
    units = assessment_units(assessment)
    references = [*units, *(item.model_dump(mode='json') for item in review.items)]
    appendix = _reference_appendix(snapshot, assessment.input_hash,
                                   {cid for item in references for cid in item['claim_ids']},
                                   {eid for item in references for eid in item['evidence_ids']})
    appendix['assessment_references'] = units
    appendix['review_references'] = [item.model_dump(mode='json') for item in review.items]
    bibliography_works = []
    for wid, work_data in appendix['works'].items():
        work = snapshot.works[wid].model_copy(deep=True)
        work.urls = [safe for url in work.urls if (safe := safe_url(url))]
        work_data['urls'] = work.urls
        bibliography_works.append(work)
    labels = dict(zip(ASSESSMENT_DIMENSIONS, ('新颖性线索', '研究价值', '可行性', '证据强度')))
    lines = ['# 创新候选评估（本次材料范围）\n\n',
             '这是材料范围内的评估，仍待人工学术判断，不证明全球首次或学术真实性。'
             '原候选作为历史系统假设保留；最低验证是拟议工作，不是已经观察到的结果。'
             '作者自述贡献归属于原作者，不能改写为系统已证明的创新。\n\n',
             f'候选：`{candidate.id}`。输入：`{assessment.input_hash}`。\n\n',
             f'材料截止：{escape_md(assessment.material_cutoff)}。'
             f'当前纳入材料：{len(snapshot.work_ids)} 篇。\n\n',
             '## 宿主核验的覆盖与证据可用性\n\n',
             f'检索行动状态：{context.coverage_status}。'
             f'证据可用性：{assessment.evidence_sufficiency}（只描述可用性，不认证结论）。\n\n',
             '原候选前提现在有效。\n\n' if context.candidate_current else
             '原候选前提已失效或范围改变：旧引用仅供历史查看，不能作为当前支持。\n\n']
    for gap in assessment.host_gaps:
        lines.append('- ' + escape_md(json.dumps(gap, ensure_ascii=False, sort_keys=True)) + '\n')
    lines.append('\n完整逐轮查询、来源、时间、历史失败及当前命中状态见 `context.json`；'
                 '历史失败恢复后仍保留记录，当前缺口由宿主重新读取。\n\n')
    for name in ASSESSMENT_DIMENSIONS:
        dimension = getattr(assessment, name)
        lines.append(f'## {labels[name]}\n\n{escape_md(dimension.judgment)}\n\n')
        for reason in dimension.reasons:
            lines.append('- ' + escape_md(reason.text) + '\n')
    lines.append('\n## 模型补充缺口\n\n')
    for gap in assessment.gaps:
        lines.append('- ' + escape_md(gap.text) + '\n')
    lines.extend(['\n## 最低验证（尚未执行）\n\n',
                  escape_md(assessment.minimum_validation.proposal) + '\n\n',
                  '否定条件：' + escape_md(assessment.minimum_validation.falsification_condition) + '\n\n',
                  '## 独立逐单元依据审查\n\n'])
    for item in review.items:
        lines.append(f'- {escape_md(item.unit_id)}：**{item.verdict}**。{escape_md(item.rationale)}\n')
    lines.append('\n引用链与原作者元数据见 `evidence.json`；宿主生成文献表见 `references.bib`。\n')
    return {'candidate.json': _json_bytes(candidate), 'assessment.json': _json_bytes(assessment),
            'assessment-review.json': _json_bytes(review), 'context.json': _json_bytes(context),
            'report.md': ''.join(lines).encode('utf-8'),
            'references.bib': export_bibtex(bibliography_works).encode('utf-8'),
            'evidence.json': _json_bytes(appendix)}


def assess_candidate(runner: Runner, candidate_id: str) -> dict:
    """Evaluate an immutable historic idea against complete current bounded evidence."""
    result = {**_result('failed'), 'candidate_ids': [candidate_id], 'coverage_status': None}
    snapshot = None
    try:
        loaded = load_candidate(runner.store.root, candidate_id)
        candidate = loaded['candidate']
        snapshot = freeze_research(runner.store, require_supported=False)
        context = _assessment_context(runner.store.root, candidate_id, loaded['manifest_hash'])
        result['input_hash'] = assessment_input_hash(snapshot, candidate, context)
        result['coverage_status'] = context.coverage_status
        result['gaps'] = assessment_host_fields(snapshot, candidate, context)['host_gaps']
        with runner.backend.validated() as backend:
            assessment = generate_assessment(snapshot, candidate, context, backend,
                                             timeout_seconds=runner.config.timeout_seconds,
                                             max_input_chars=runner.config.max_input_chars)
            _assessment_current(runner, snapshot, candidate, context)
        with runner.backend.validated() as backend:
            review = review_assessment(snapshot, candidate, context, assessment, backend,
                                       timeout_seconds=runner.config.timeout_seconds,
                                       max_input_chars=runner.config.max_input_chars)
            _assessment_current(runner, snapshot, candidate, context)
        _assessment_current(runner, snapshot, candidate, context)
        output = publish_research(runner.store, 'assessment', snapshot,
                                  _assessment_payloads(snapshot, candidate, context, assessment, review))
        result.update(status='complete', output=str(output.relative_to(runner.store.root)))
    except BudgetExceeded:
        result.update(status='budget_paused', gaps=[*result['gaps'], {'reason': 'model_budget_exhausted'}])
    except InnovationError as exc:
        status = 'needs_input' if exc.status == 'needs_input' else 'failed'
        gap = {'reason': 'model_input_required' if status == 'needs_input' else 'model_artifact_invalid'}
        if exc.status == 'input_too_large':
            gap = {'reason': 'model_input_too_large', 'guidance': (
                'Assessment uses all currently included material without truncation. Review the true research '
                'scope and screen unrelated works only with explicit reasons, preserving candidate premises '
                'and relevant counterexamples; otherwise address currentness gaps first. If the scope cannot '
                'shrink, keep this failure or explicitly choose a compatible backend input limit.')}
        result.update(status=status, gaps=[*result['gaps'], gap])
    except (_StaleResearch, StaleJobError):
        result.update(status='stale', gaps=[*result['gaps'], {'reason': 'research_inputs_changed'}])
    except (OSError, ValueError, KeyError, TypeError):
        result.update(status='failed', gaps=[*result['gaps'], {'reason': 'research_input_invalid'}])
    return result


def _assessment_history(root: Path, candidates: dict) -> list[dict]:
    base = root / 'analyses' / 'assessment'
    _check_history(base)
    history = []
    if not base.exists():
        return history
    for directory in sorted(base.glob('v-*')):
        loaded = load_assessment_version(root, directory)
        candidate = loaded['candidate']
        assessment = loaded['assessment']
        if candidate.id not in candidates or candidate != candidates[candidate.id]['candidate']:
            raise ValueError('Assessment historical candidate does not match its verified origin')
        history.append({'candidate_id': candidate.id, 'input_hash': assessment.input_hash,
                        'output': str(directory.relative_to(root)), 'created': loaded['snapshot'].captured_at})
    return history


def _candidate_origin(root: Path, manifest_hash: str, candidate_id: str) -> dict:
    base = Path(root) / 'analyses' / 'innovation'
    _check_history(base)
    matches = []
    if base.exists():
        for original in sorted(base.glob('v-*')):
            artifact = load_research(original)
            if artifact['manifest']['manifest_hash'] == manifest_hash:
                matches.append(load_candidate_version(original, candidate_id))
    if len(matches) != 1:
        raise ValueError('Assessment candidate provenance is missing or ambiguous')
    return matches[0]


def load_assessment_version(root: Path, directory: Path) -> dict:
    """Load one assessment from one verified byte capture and check host closure."""
    root = Path(root).resolve()
    directory = Path(directory).resolve()
    loaded = load_research(directory)
    files = loaded['files']
    if set(files) != {'candidate.json', 'assessment.json', 'assessment-review.json', 'context.json',
                      'report.md', 'references.bib', 'evidence.json'}:
        raise ValueError('Assessment payload set is invalid')
    candidate = InnovationCandidate.model_validate_json(files['candidate.json'], strict=True)
    context = AssessmentContext.model_validate_json(files['context.json'], strict=True)
    origin = _candidate_origin(root, context.candidate_manifest_hash, candidate.id)
    if candidate != origin['candidate']:
        raise ValueError('Assessment historical candidate does not match its verified origin')
    snapshot = ResearchSnapshot.model_validate(loaded['manifest']['snapshot'], strict=True)
    assessment = validate_assessment(snapshot, candidate, context,
                                     InnovationAssessment.model_validate_json(files['assessment.json'], strict=True))
    review = validate_assessment_review(
        snapshot, assessment, AssessmentReview.model_validate_json(files['assessment-review.json'], strict=True))
    if files != _assessment_payloads(snapshot, candidate, context, assessment, review):
        raise ValueError('Assessment host payloads do not match verified structured content')
    return {'candidate': candidate, 'context': context, 'assessment': assessment, 'review': review,
            'snapshot': snapshot, 'version_path': directory,
            'manifest_hash': loaded['manifest']['manifest_hash']}


def assessment_version_currentness(root: Path, loaded: dict) -> dict:
    """Recompute currentness for the selected assessment without resolving assessment/latest."""
    candidate = loaded.get('candidate')
    candidate_id = candidate.id if candidate is not None else ''
    try:
        context = loaded['context']
        origin = _candidate_origin(Path(root), context.candidate_manifest_hash, candidate_id)
        premise = candidate_version_currentness(Path(root), origin)
        research_state, current = diagnose_research_state(Path(root), loaded['snapshot'])
        if research_state == 'invalid' or current is None:
            raise ValueError('Assessment research inputs are invalid')
        coverage = read_challenges(
            Path(root), candidate_id, verified_research=loaded['snapshot']
        )
        current_context = AssessmentContext(
            candidate_manifest_hash=context.candidate_manifest_hash,
            candidate_current=premise['current'], premise_gaps=premise['gaps'],
            rounds=coverage['rounds'], hits=coverage['hits'],
            unresolved_tasks=coverage['unresolved_tasks'], coverage_status=coverage['status'],
        )
        if research_state in {'changed', 'source_changed'}:
            return {'candidate_id': candidate_id, 'current': False,
                    'gaps': [{'reason': 'assessment_inputs_changed'}]}
        expected = assessment_input_hash(current, candidate, current_context)
        if expected != loaded['assessment'].input_hash:
            return {'candidate_id': candidate_id, 'current': False,
                    'gaps': [{'reason': 'assessment_inputs_changed'}]}
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return {'candidate_id': candidate_id, 'current': False,
                'gaps': [{'reason': 'assessment_inputs_invalid'}]}
    return {'candidate_id': candidate_id, 'current': True, 'gaps': []}


def innovation_status(root: Path) -> dict:
    """Read history and freshness without constructing a writable Store or a backend."""
    root = Path(root).resolve()
    result = {'status': 'complete', 'candidates': [], 'gaps': []}
    try:
        base = root / 'analyses' / 'innovation'
        _check_history(base)
        ids = set()
        for directory in base.glob('v-*'):
            batch = CandidateBatch.model_validate_json(load_research(directory)['files']['candidates.json'])
            ids.update(candidate.id for candidate in batch.candidates)
        candidates = {cid: load_candidate(root, cid) for cid in sorted(ids)}
    except (OSError, ValueError, KeyError, TypeError):
        return {'status': 'failed', 'candidates': [], 'gaps': [{'reason': 'candidate_artifact_invalid'}]}
    try:
        snapshot = read_research(root, require_supported=False)
    except (OSError, ValueError, KeyError, TypeError):
        snapshot = None
        result.update(status='failed', gaps=[{'reason': 'research_input_invalid'}])
    try:
        history = _assessment_history(root, candidates)
        history_error = False
    except (OSError, ValueError, KeyError, TypeError):
        history, history_error = [], True
        result.update(status='failed', gaps=[*result['gaps'], {'reason': 'assessment_artifact_invalid'}])
    for cid, loaded in candidates.items():
        premise = candidate_currentness(root, cid)
        assessment = {'available': False, 'current': False, 'output': None,
                      'status': 'invalid' if history_error else 'missing', 'gaps': []}
        entries = [entry for entry in history if entry['candidate_id'] == cid]
        if entries:
            latest = max(entries, key=lambda entry: (entry['created'], entry['output']))
            assessment.update(available=True, output=latest['output'], input_hash=latest['input_hash'])
            try:
                context = _assessment_context(root, cid, loaded['manifest_hash'])
                current = snapshot is not None and latest['input_hash'] == assessment_input_hash(
                    snapshot, loaded['candidate'], context)
                assessment.update(current=current, status='current' if current else 'stale')
                if not current:
                    assessment['gaps'] = [{'reason': 'assessment_inputs_changed'}]
            except (OSError, ValueError, KeyError, TypeError):
                assessment.update(status='stale', gaps=[{'reason': 'assessment_inputs_invalid'}])
        result['candidates'].append({'candidate_id': cid, 'historical_available': True,
                                     'premises_current': premise['current'], 'premise_gaps': premise['gaps'],
                                     'assessment': assessment})
    return result
