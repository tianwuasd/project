"""Versioned innovation protocol for the two bundled original scheduling notes."""

from __future__ import annotations

from pydantic import ValidationError

from .demo import FIXTURES
from .documents import verify_span
from .innovation_models import CandidateBatch, InnovationAssessment, InnovationCandidate
from .models import Claim, Document, Evidence, ModelJob, ModelResult
from .research_context import ResearchDecision

_GENERATION = "innovation-generation-v1"
_REVIEW = "innovation-grounding-review-v1"
_ASSESSMENT = 'innovation-assessment-v1'
_ASSESSMENT_REVIEW = 'innovation-assessment-review-v1'


def _failure(message: str) -> ModelResult:
    return ModelResult(status="failed", backend="demo", error=message)


def _validated_fixture(job: ModelJob):
    research = job.payload.get("research", {})
    works = research.get("works", []) if isinstance(research, dict) else []
    expected = {title: text for title, text in FIXTURES}
    if len(works) != 2 or {item.get("work", {}).get("title") for item in works} != set(expected):
        raise ValueError("Demo innovation requires the exact two original scheduling fixtures")
    chains = []
    by_title = {}
    for item in works:
        if not isinstance(item, dict) or not isinstance(item.get("work"), dict):
            raise ValueError("Demo innovation work context is malformed")
        title = item["work"]["title"]
        metadata = item["work"]
        expected_source = f"study-{1 if title.endswith('A') else 2}"
        if (metadata.get("authors") != ["Academic Agent original fixture"]
                or metadata.get("year") != 2026
                or metadata.get("source_ids") != {"synthetic": expected_source}
                or metadata.get("screening") != "include"):
            raise ValueError("Demo innovation metadata does not match the known fixture protocol")
        document = Document.model_validate(item["document"])
        if "".join(page.text for page in document.pages) != expected[title]:
            raise ValueError("Demo innovation fixture text does not match the known protocol")
        if not all(line in expected[title] for line in (
            "ORIGINAL_SYNTHETIC_FIXTURE", "no real participants", "10 to 8" if title.endswith("A") else "10 to 11"
        )):
            raise ValueError("Demo innovation fixture conditions do not match the known protocol")
        for chain in item.get("reviewed_claims", []):
            claim = Claim.model_validate(chain.get("claim"))
            decision = ResearchDecision.model_validate(chain.get("decision"))
            evidence = [Evidence.model_validate(value) for value in chain.get("evidence", [])]
            evidence_ids = {span.id for span in evidence}
            expected_type = "author_contribution" if claim.category == "contribution" else "source_claim"
            if (not evidence or any(not verify_span(span, document) for span in evidence)
                    or claim.work_id != document.work_id or set(claim.evidence_ids) != evidence_ids
                    or decision.claim_id != claim.id or decision.work_id != document.work_id
                    or decision.document_id != document.id
                    or not set(decision.evidence_ids) <= evidence_ids
                    or chain.get("statement_type") != expected_type):
                raise ValueError("Demo innovation requires exact validated evidence spans")
            chains.append(chain)
        by_title[title] = item
    return research, by_title, chains


def _generation_data(research, by_title, chains):
    supported = [chain for chain in chains if chain["decision"]["verdict"] == "supported"]
    results = [chain for chain in supported if chain["claim"]["category"] == "result"]
    contributions = [chain for chain in supported if chain["claim"]["category"] == "contribution"]
    if len(results) != 2 or len(contributions) != 2:
        raise ValueError("Demo innovation requires reviewed result and contribution chains for both fixtures")
    work_ids = [by_title[title]["work"]["id"] for title, _ in FIXTURES]
    return {"candidates": [{
        "statement_type": "system_hypothesis",
        "question": "在不同原创合成负载下，负载感知的策略选择能否降低平均等待时间？",
        "closest_work_ids": work_ids,
        "closest_claim_ids": [chain["claim"]["id"] for chain in contributions],
        "claimed_difference": "提出按预先定义的负载特征选择调度策略，避免把固定优先级规则当作无条件改进。",
        "grounding_claim_ids": [chain["claim"]["id"] for chain in results],
        "grounding_evidence_ids": sorted({evidence_id for chain in results
                                           for evidence_id in chain["decision"]["evidence_ids"]}),
        "possible_counterexamples": ["60 个作业的突发到达负载中，平均等待时间从 10 增至 11。"],
        "falsifiable_hypothesis": "预注册的负载感知切换规则在留出合成负载上优于两种固定策略。",
        "minimum_validation": "在 40 作业和 60 作业条件及至少一组留出负载上预注册规则并比较平均等待时间。",
        "data_requirements": "原创合成作业的到达时间、服务时间、优先级与负载类型。",
        "compute_requirements": "可重复的小型离散事件模拟和固定随机种子。",
        "risks": ["切换规则可能只拟合两个已知合成负载，且不代表真实参与者或真实系统。"],
        "targeted_queries": ["workload adaptive scheduling mean waiting time counterexample"],
        "material_cutoff": research["material_cutoff"],
    }]}


def run_demo_innovation(job: ModelJob) -> ModelResult:
    """Handle only the exact versioned synthesis protocol and bundled A/B pair."""
    task = job.payload.get("task")
    if task not in {_GENERATION, _REVIEW, _ASSESSMENT, _ASSESSMENT_REVIEW}:
        return _failure("Demo does not support this synthesis protocol")
    try:
        research, by_title, chains = _validated_fixture(job)
        if task == _GENERATION:
            data = _generation_data(research, by_title, chains)
        elif task in {_ASSESSMENT, _ASSESSMENT_REVIEW}:
            data = _assessment_data(job, chains)
        else:
            batch = CandidateBatch.model_validate(job.payload.get("batch"))
            if len(batch.candidates) != 1:
                raise ValueError("Demo grounding review requires its one known candidate")
            candidate = batch.candidates[0]
            data = {"items": [{
                "candidate_id": candidate.id,
                "verdict": "partial",
                "rationale": (
                    "原创 A/B 材料支持固定策略效果随负载改变；它不证明所提切换规则的新颖性或实验成功。"
                ),
                "claim_ids": candidate.grounding_claim_ids,
                "evidence_ids": candidate.grounding_evidence_ids,
            }]}
    except (AttributeError, KeyError, TypeError, ValueError, ValidationError):
        return _failure("Demo innovation input failed the exact original-fixture protocol")
    return ModelResult(status="succeeded", backend="demo", data=data,
                       usage={"mode": "synthetic_demo", "model_calls": 0})


def _assessment_data(job, chains):
    from .innovation_analysis import ASSESSMENT_DIMENSIONS, assessment_units

    candidate = InnovationCandidate.model_validate(job.payload['historical_candidate'])
    results = [c for c in chains if c['claim']['category'] == 'result'
               and c['decision']['verdict'] == 'supported']
    if len(results) != 2:
        raise ValueError('Demo assessment requires both current supported original result chains')
    refs = {'claim_ids': [c['claim']['id'] for c in results],
            'evidence_ids': sorted({eid for c in results for eid in c['decision']['evidence_ids']})}
    if job.payload['task'] == _ASSESSMENT_REVIEW:
        assessment = InnovationAssessment.model_validate(job.payload['assessment'])
        if assessment.candidate_id != candidate.id or job.payload['units'] != assessment_units(assessment):
            raise ValueError('Demo assessment review requires exact complete units')
        return {'items': [{'unit_id': unit['unit_id'], 'verdict': 'partial', **refs,
                            'rationale': '固定原创 A/B 协议仅示范逐单元复核，不证明新颖性、提案成功或学术准确率。'}
                           for unit in job.payload['units']]}
    judgments = ('材料不足以确认新颖性', '负载相关差异值得提出验证问题',
                 '小型模拟可作为拟议验证起点', '两份合成材料仅提供有限依据')
    result = {name: {'judgment': judgment, **refs, 'reasons': [{
        'text': '原创 A 在 40 个模拟作业下等待时间 10→8；B 在 60 个突发到达作业下 10→11。'
                '作者报告的是这些条件内的比较，不是所提切换规则的已观察效果。', **refs}]}
              for name, judgment in zip(ASSESSMENT_DIMENSIONS, judgments)}
    result.update(gaps=[{'text': '仍需在实际记录的检索范围内排查反例并进行人工学术判断。',
                         'claim_ids': [], 'evidence_ids': []}],
                  minimum_validation={
                      'proposal': '预注册负载感知规则，在 40、60 个作业及独立留出合成负载上对比两种固定策略。',
                      'falsification_condition': '若留出负载的平均等待时间未优于两种固定策略，否定该假设。',
                  })
    return result
