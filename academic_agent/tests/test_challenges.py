import json
from collections import Counter

import pytest
from pypdf import PdfWriter
from test_innovation_generation import RecordingBackend

from academic_agent import sources
from academic_agent.challenges import (
    candidate_currentness,
    candidate_version_currentness,
    challenge_candidate,
    load_candidate,
    load_candidate_version,
    read_challenges,
)
from academic_agent.config import atomic_json, load_config
from academic_agent.demo import FIXTURES, seed_demo
from academic_agent.innovation_analysis import (
    _search_event,
    validate_candidate_batch,
    validate_grounding_review,
)
from academic_agent.innovations import create_innovations
from academic_agent.models import PublicationCheck, SearchPage, Work, digest
from academic_agent.workflow import Runner


class TwoQueries(RecordingBackend):
    def run(self, job):
        result = super().run(job)
        if job.payload['task'] == 'innovation-generation-v1':
            result.data['candidates'][0]['targeted_queries'] = ['original alpha', 'original beta']
        return result


@pytest.fixture
def project(tmp_path, monkeypatch):
    tmp_path = tmp_path / 'original #% project'
    seed_demo(tmp_path)
    Runner(tmp_path, offline=True).run()
    config = load_config(tmp_path)
    config.sources = ['openalex', 'crossref']
    config.max_candidates = 20
    config.max_model_calls = 20
    config.max_input_chars = 100_000
    atomic_json(tmp_path / 'project.json', config.model_dump())
    result = create_innovations(Runner(tmp_path, offline=True, backend=TwoQueries()))
    assert result['status'] == 'complete'
    monkeypatch.setattr('academic_agent.workflow.check_publication',
                        lambda *a, **k: [PublicationCheck(source='original-fixture')])
    return tmp_path, result['candidate_ids'][0]


def adapters(monkeypatch, outcomes=None):
    counts, limits = Counter(), []
    for name in ['openalex', 'crossref']:
        def make(source):
            class OriginalSource:
                def __init__(self, **kwargs):
                    pass

                def search(self, query, limit):
                    counts[source, query] += 1
                    limits.append((source, query, limit))
                    if outcomes:
                        return outcomes(source, query, counts[source, query])
                    return SearchPage(source=source, status='empty', queries=[query])
            return OriginalSource
        monkeypatch.setitem(sources.SOURCES, name, make(name))
    return counts, limits


def ready_later_challenge_work(root, candidate_id, monkeypatch):
    """Add and fully review a retained hit after the candidate snapshot was published."""
    adapters(monkeypatch, lambda source, query, count: SearchPage(source=source, records=[
        Work(title='Later retained challenge work', doi='10.1234/later-challenge-work'),
    ]))
    runner = Runner(root)
    work_id, = challenge_candidate(runner, candidate_id)['retained_work_ids']
    work = runner.store.work(work_id)
    work.publication_checks = [
        PublicationCheck(source='original-fixture', status='no_notice_found'),
    ]
    runner.store.update_work(work)
    source = root / 'later-challenge-work.txt'
    source.write_text(FIXTURES[0][1], encoding='utf-8')
    runner.attach(work_id, source)
    runner.store.screen(work_id, 'include', 'Later challenge evidence')
    assert runner.read_selected(work_id=work_id)['results'][0]['status'] == 'ok'
    assert runner.review_selected(work_id=work_id)['results'][0]['status'] == 'ok'
    attached_source = root / runner.store.active_document(work_id).source_path
    return work_id, attached_source


def explicit_candidate(root, candidate_id):
    """Publish a candidate from an explicit historical research selection."""
    historical = load_candidate(root, candidate_id)['snapshot']
    result = create_innovations(
        Runner(root, offline=True, backend=TwoQueries()),
        work_ids=historical.work_ids,
    )
    assert result['status'] == 'complete'
    selected_id, = result['candidate_ids']
    selected = load_candidate(root, selected_id)
    assert selected['snapshot'].selection == 'explicit'
    return selected_id, selected['snapshot']


def test_two_rounds_are_fresh_but_same_round_is_zero_dispatch(project, monkeypatch):
    root, cid = project
    counts, limits = adapters(monkeypatch)
    runner = Runner(root)
    # A healthy general-search cache predates the durable round start.
    runner.search('original alpha', limit=10)
    first = challenge_candidate(runner, cid)
    assert first['status'] == 'needs_assessment'
    assert len(first['queries']) == 2
    before = counts.copy()
    assert challenge_candidate(runner, cid)['round'] == 1
    assert counts == before
    challenge_candidate(runner, cid, round_number=2)
    assert all(counts[name, query] == before[name, query] + 1
               for name in ['openalex', 'crossref'] for query in ['original alpha', 'original beta'])
    assert all(limit == 10 for _, _, limit in limits)
    with pytest.raises(ValueError, match='round'):
        challenge_candidate(runner, cid, round_number=3)
    assert len(limits) == 10


def test_failed_query_only_retries_and_preserves_success_receipt(project, monkeypatch):
    root, cid = project
    def outcome(source, query, count):
        status = 'rate_limited' if source == 'crossref' and query.endswith('beta') and count == 1 else 'empty'
        return SearchPage(source=source, status=status, errors=['fixture limit'] if status == 'rate_limited' else [])
    counts, _ = adapters(monkeypatch, outcome)
    runner = Runner(root)
    first = challenge_candidate(runner, cid)
    assert first['status'] == 'source_failed'
    receipt = first['queries'][0]
    second = challenge_candidate(runner, cid)
    assert second['status'] == 'needs_assessment'
    assert second['queries'][0] == receipt
    assert counts['openalex', 'original alpha'] == counts['crossref', 'original alpha'] == 1
    assert counts['openalex', 'original beta'] == counts['crossref', 'original beta'] == 2
    assert (root / first['output']).is_dir()


@pytest.mark.parametrize('limit', [True, False, 0, 101, 1.5, '10'])
def test_invalid_limits_reject_before_dispatch(project, monkeypatch, limit):
    root, _ = project
    counts, _ = adapters(monkeypatch)
    with pytest.raises(ValueError, match='limit'):
        Runner(root).search('original', limit=limit)
    assert not counts


def test_new_search_ids_limit_projection_and_legacy_unknown(project, monkeypatch):
    root, _ = project
    counts, _ = adapters(monkeypatch, lambda source, query, count: SearchPage(source=source, records=[
        Work(title='Original bounded hit', doi='10.1234/original-hit')]))
    runner = Runner(root)
    result = runner.search('original', limit=10)
    assert len(result['retained_work_ids']) == 1
    event = runner.store.logs('search')[-1]
    assert set(event['data']['summary']) == {'query', 'retained', 'dropped_at_project_cap', 'sources'}
    projected = _search_event(event)
    assert projected['retained_work_ids'] == result['retained_work_ids']
    assert projected['retained_ids_available'] is True
    assert all(row['limit'] == 10 for row in projected['sources'])
    assert runner.search('original', cached=True, limit=10)['cached'] is True
    runner.search('original', cached=True)
    assert counts['openalex', 'original'] == counts['crossref', 'original'] == 2
    assert runner.store.logs('search')[-1]['data']['query_key'] != event['data']['query_key']
    key = digest(json.dumps(['old', runner.config.sources, runner.config.max_candidates]))
    runner.store.log('search', {'query_key': key, 'summary': {**event['data']['summary'], 'query': 'old'}})
    old = runner.search('old', cached=True)
    assert old['retained_work_ids'] == [] and old['retained_ids_available'] is False
    assert old['effective_limit'] is None
    projected = _search_event(runner.store.logs('search')[-1])
    assert projected['retained_ids_available'] is False
    assert all(row['limit'] is None for row in projected['sources'])


def test_history_loader_reads_without_store_and_detects_edits(project, monkeypatch):
    root, cid = project
    monkeypatch.setattr('academic_agent.store.Store.__init__', lambda *a: pytest.fail('writable Store'))
    loaded = load_candidate(root, cid)
    assert set(loaded) == {'candidate', 'batch', 'review', 'snapshot', 'version_path', 'manifest_hash'}
    assert loaded['candidate'].id == cid
    assert read_challenges(root, cid)['status'] == 'not_searched'
    assert candidate_currentness(root, cid) == {'candidate_id': cid, 'current': True, 'gaps': []}
    (loaded['version_path'] / 'candidates.json').write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='hash'):
        load_candidate(root, cid)
    assert candidate_currentness(root, cid)['gaps'] == [{'reason': 'candidate_artifact_invalid'}]


def test_single_candidate_version_loader_preserves_legacy_shape(project):
    root, cid = project
    selected = load_candidate(root, cid)
    loaded = load_candidate_version(selected['version_path'], cid)
    assert set(loaded) == {'candidate', 'batch', 'review', 'snapshot', 'version_path', 'manifest_hash'}
    assert candidate_version_currentness(root, loaded) == {'candidate_id': cid, 'current': True, 'gaps': []}


def test_candidate_source_hash_change_is_stale_but_mixed_invalid_history_stays_invalid(project):
    root, cid = project
    loaded = load_candidate(root, cid)
    document = next(iter(loaded['snapshot'].documents.values()))
    source = root / document.source_path
    source.write_bytes(source.read_bytes() + b' changed source')
    assert candidate_version_currentness(root, loaded) == {
        'candidate_id': cid,
        'current': False,
        'gaps': [{'reason': 'candidate_evidence_changed'}],
    }

    (loaded['version_path'] / 'candidates.json').write_text('{}', encoding='utf-8')
    assert candidate_currentness(root, cid) == {
        'candidate_id': cid,
        'current': False,
        'gaps': [{'reason': 'candidate_artifact_invalid'}],
    }


def test_crash_after_second_search_log_recovers_receipt_without_dispatch(project, monkeypatch):
    root, cid = project
    counts, _ = adapters(monkeypatch)
    runner = Runner(root)
    original = runner.refresh_publication
    def crash(works, *, stage):
        if len(runner.store.logs('search')) == 2:
            raise RuntimeError('original fixture crash after logged search')
        return original(works, stage=stage)
    monkeypatch.setattr(runner, 'refresh_publication', crash)
    with pytest.raises(RuntimeError, match='fixture crash'):
        challenge_candidate(runner, cid)
    before = counts.copy()
    result = challenge_candidate(Runner(root), cid)
    assert counts == before
    saved = json.loads((root / result['output'] / 'round.json').read_text(encoding='utf-8'))
    assert len(saved['receipts']) == 2
    assert len(result['queries']) == 2


def test_republished_identical_candidate_does_not_reset_round(project, monkeypatch):
    root, cid = project
    # Interrupt after the durable round boundary, before a search event changes input.
    runner = Runner(root)
    monkeypatch.setattr(runner, 'search', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('interrupt')))
    with pytest.raises(RuntimeError, match='interrupt'):
        challenge_candidate(runner, cid)
    republished = create_innovations(Runner(root, offline=True, backend=TwoQueries()))
    assert republished['candidate_ids'] == [cid]
    counts, _ = adapters(monkeypatch)
    result = challenge_candidate(Runner(root), cid)
    before = counts.copy()
    again = challenge_candidate(Runner(root), cid)
    assert again['identity'] == result['identity'] and counts == before
    assert len(list((root / 'analyses' / 'innovation').glob('v-*'))) == 2
    assert len(Runner(root).store.logs('challenge_start')) == 1


def test_partial_attempt_hits_remain_visible_after_different_retry_hit(project, monkeypatch):
    root, cid = project
    def outcome(source, query, count):
        if source == 'openalex' and query.endswith('beta') and count == 1:
            return SearchPage(source=source, status='partial', records=[Work(title='Original partial hit')],
                              errors=['original page interruption'])
        if source == 'openalex' and query.endswith('beta'):
            return SearchPage(source=source, records=[Work(title='Original retry hit')])
        return SearchPage(source=source, status='empty')
    adapters(monkeypatch, outcome)
    first = challenge_candidate(Runner(root), cid)
    second = challenge_candidate(Runner(root), cid)
    assert set(first['retained_work_ids']) < set(second['retained_work_ids'])
    assert len(second['retained_work_ids']) == 2
    coverage = read_challenges(root, cid)
    assert any('pending_screen' in hit.statuses for hit in coverage['hits'])
    assert not any(task['reason'] == 'partial' for task in coverage['unresolved_tasks'])
    assert any(s.status == 'partial' for a in coverage['rounds'][0].search_attempts for s in a.sources)


def test_cap_dedup_publication_bound_and_current_hit_transitions(project, monkeypatch):
    root, cid = project
    config = load_config(root)
    config.max_candidates = 3
    atomic_json(root / 'project.json', config.model_dump())
    _, limits = adapters(monkeypatch, lambda source, query, count: SearchPage(source=source, truncated=True, records=[
        Work(title='Original retained hit', doi='10.1234/original-one', abstract=FIXTURES[0][1]),
        Work(title='Original dropped hit', doi='10.1234/original-two')]))
    runner = Runner(root)
    result = challenge_candidate(runner, cid)
    assert len(runner.store.works()) == 3
    assert len(result['retained_work_ids']) == 1
    wid = result['retained_work_ids'][0]
    assert all(r['effective_limit'] == 3 for r in result['queries'])
    assert all(limit == 3 for _, _, limit in limits)
    reasons = {t['reason'] for t in result['unresolved_tasks']}
    assert {'pending_screen', 'truncated', 'dropped_at_cap', 'publication_unknown'} <= reasons
    assert runner.store.logs('publication_refresh')[-1]['data']['work_ids'] == [wid]
    hit = runner.store.work(wid)
    hit.screening, hit.screening_reason = 'exclude', 'Original scope mismatch'
    runner.store.update_work(hit)
    read = read_challenges(root, cid)
    assert 'excluded_with_reason' in read['hits'][0].statuses
    assert read['hits'][0].screening_reason == 'Original scope mismatch'
    hit = runner.store.work(wid)
    hit.screening, hit.screening_reason = 'include', 'Original followup reading'
    runner.store.update_work(hit)
    assert 'unread' in read_challenges(root, cid)['hits'][0].statuses
    reader = Runner(root, offline=True)
    reader.read_selected(work_id=wid)
    assert 'unreviewed' in read_challenges(root, cid)['hits'][0].statuses
    reader.review_selected(work_id=wid)
    assert 'ready' in read_challenges(root, cid)['hits'][0].statuses
    # New evidence changes coverage, not the old candidate text or its rounds.
    assert challenge_candidate(Runner(root), cid)['candidate_id'] == cid


def test_verified_hits_use_full_current_research_after_explicit_candidate(project, monkeypatch):
    root, cid = project
    cid, historical = explicit_candidate(root, cid)
    work_id, _ = ready_later_challenge_work(root, cid, monkeypatch)

    ordinary = read_challenges(root, cid)
    verified = read_challenges(root, cid, verified_research=historical)
    assert ordinary['hits'] == verified['hits']
    hit, = verified['hits']
    assert hit.work_id == work_id
    assert hit.statuses == ['ready']
    assert ordinary['status'] == verified['status'] == 'needs_assessment'


@pytest.mark.parametrize('mutation', ['research', 'source'])
def test_verified_hits_reject_mutation_between_full_research_reads(
        project, monkeypatch, mutation):
    import academic_agent.challenges as challenges

    root, cid = project
    cid, historical = explicit_candidate(root, cid)
    work_id, source = ready_later_challenge_work(root, cid, monkeypatch)
    runner = Runner(root, offline=True)
    claim_id = runner.store.logs('reading_card')[-1]['data']['card']['claims'][0]['id']
    original = challenges._active_documents
    mutation_exercised = False

    def mutate_before_first_verified_document_read(
            root, work_ids, *, source_hash_mismatches=None):
        nonlocal mutation_exercised
        if source_hash_mismatches is not None and not mutation_exercised:
            mutation_exercised = True
            if mutation == 'source':
                source.write_bytes(source.read_bytes() + b' changed between reads')
            else:
                runner.manual_review(
                    claim_id, 'supported', 'Changed between full reads', 'fixture reviewer')
        return original(
            root, work_ids, source_hash_mismatches=source_hash_mismatches
        )

    monkeypatch.setattr(
        challenges, '_active_documents', mutate_before_first_verified_document_read)
    with pytest.raises(ValueError):
        read_challenges(root, cid, verified_research=historical)
    assert mutation_exercised


@pytest.mark.parametrize('fault', ['offline', 'excluded', 'retracted', 'document', 'question'])
def test_invalid_current_premises_reject_before_search(project, monkeypatch, fault):
    root, cid = project
    counts, _ = adapters(monkeypatch)
    runner = Runner(root, offline=fault == 'offline')
    loaded = load_candidate(root, cid)
    wid = loaded['snapshot'].work_ids[0]
    if fault in {'excluded', 'retracted'}:
        work = runner.store.work(wid)
        if fault == 'excluded':
            work.screening, work.screening_reason = 'exclude', 'Original changed selection'
        else:
            work.publication_checks.append(PublicationCheck(source='fixture', status='retracted'))
        runner.store.update_work(work)
    elif fault == 'document':
        document = next(iter(loaded['snapshot'].documents.values()))
        (root / document.source_path).write_text('Original changed source text', encoding='utf-8')
    elif fault == 'question':
        config = load_config(root)
        config.question = 'Changed original question'
        atomic_json(root / 'project.json', config.model_dump())
        runner = Runner(root)
    with pytest.raises(ValueError):
        challenge_candidate(runner, cid)
    assert not counts
    if fault != 'offline':
        current = candidate_currentness(root, cid)
        assert current['candidate_id'] == cid and current['current'] is False and current['gaps']


def test_no_configured_sources_is_not_searched(project, monkeypatch):
    root, _ = project
    config = load_config(root)
    config.sources = []
    atomic_json(root / 'project.json', config.model_dump())
    result = create_innovations(Runner(root, offline=True, backend=TwoQueries()))
    counts, _ = adapters(monkeypatch)
    challenge = challenge_candidate(Runner(root), result['candidate_ids'][0])
    assert challenge['status'] == 'not_searched'
    assert not counts and not challenge['queries']


def test_shared_validators_revalidate_mutable_host_identity_and_review_hash(project):
    root, cid = project
    loaded = load_candidate(root, cid)
    loaded['batch'].candidates[0].id = 'innovation_' + '0' * 32
    with pytest.raises(ValueError, match='identity'):
        validate_candidate_batch(loaded['snapshot'], loaded['batch'])
    loaded = load_candidate(root, cid)
    loaded['review'].input_hash = '0' * 64
    with pytest.raises(ValueError, match='snapshot'):
        validate_grounding_review(loaded['snapshot'], loaded['batch'], loaded['review'])


def test_saved_receipt_binds_full_event_and_rejects_log_edits(project, monkeypatch):
    root, cid = project
    adapters(monkeypatch)
    runner = Runner(root)
    challenge_candidate(runner, cid)
    event = runner.store.logs('search')[0]
    event['data']['responses'].append({'original tamper': 'raw response changed'})
    with runner.store.connect() as connection:
        connection.execute('UPDATE logs SET data=? WHERE id=?', (json.dumps(event['data']), event['id']))
    with pytest.raises(ValueError, match='receipt'):
        read_challenges(root, cid)


def test_failed_round_one_round_two_changed_sources_and_resume_one(project, monkeypatch):
    root, cid = project
    def outcome(source, query, count):
        status = 'unavailable' if source == 'crossref' and query.endswith('beta') and count == 1 else 'empty'
        return SearchPage(source=source, status=status)
    counts, _ = adapters(monkeypatch, outcome)
    first = challenge_candidate(Runner(root), cid)
    assert first['status'] == 'source_failed'
    config = load_config(root)
    config.sources = ['openalex']
    atomic_json(root / 'project.json', config.model_dump())
    assert candidate_currentness(root, cid)['current'] is True
    with pytest.raises(ValueError, match='scope changed'):
        challenge_candidate(Runner(root), cid)
    second = challenge_candidate(Runner(root), cid, round_number=2)
    assert second['status'] == 'needs_assessment'
    config.sources = ['openalex', 'crossref']
    atomic_json(root / 'project.json', config.model_dump())
    resumed = challenge_candidate(Runner(root), cid)
    assert resumed['queries'][0] == first['queries'][0]
    assert resumed['status'] == 'needs_assessment'
    assert counts == {('openalex', 'original alpha'): 2, ('crossref', 'original alpha'): 1,
                      ('openalex', 'original beta'): 3, ('crossref', 'original beta'): 2}
    rounds = read_challenges(root, cid)['rounds']
    assert rounds[0].sources == ['openalex', 'crossref'] and rounds[1].sources == ['openalex']
    assert not ({r.search_event_id for r in rounds[0].search_attempts}
                & {r.search_event_id for r in rounds[1].search_attempts})


def test_explicit_search_request_does_not_relabel_global_cache(project, monkeypatch):
    root, _ = project
    counts, _ = adapters(monkeypatch)
    runner = Runner(root)
    old = runner.search('original', limit=10)
    new = runner.search('original', limit=10, cached=True, request_id='original-request')
    assert old['search_event_id'] != new['search_event_id']
    assert runner.store.logs('search')[-1]['data']['request_id'] == 'original-request'
    assert runner.search('original', limit=10, cached=True, request_id='original-request')['cached'] is True
    assert counts['openalex', 'original'] == counts['crossref', 'original'] == 2


@pytest.mark.parametrize('interruption', ['after_start', 'after_dispatch', 'after_first_receipt'])
def test_started_round_gaps_cannot_be_satisfied_by_previous_round(project, monkeypatch, interruption):
    root, cid = project
    counts, _ = adapters(monkeypatch)
    assert challenge_candidate(Runner(root), cid)['status'] == 'needs_assessment'
    # Repeated query text with a revised source scope is still a fresh round.
    config = load_config(root)
    config.sources = ['openalex']
    atomic_json(root / 'project.json', config.model_dump())
    runner = Runner(root)
    original_log, original_search = runner.store.log, runner.search

    def interrupted_log(kind, data):
        result = original_log(kind, data)
        if interruption == 'after_start' and kind == 'challenge_start':
            raise RuntimeError('original round interruption')
        return result

    def interrupted_search(query, **kwargs):
        if interruption == 'after_dispatch' or query == 'original beta':
            raise RuntimeError('original round interruption')
        return original_search(query, **kwargs)

    monkeypatch.setattr(runner.store, 'log', interrupted_log)
    monkeypatch.setattr(runner, 'search', interrupted_search)
    with pytest.raises(RuntimeError, match='original round interruption'):
        challenge_candidate(runner, cid, round_number=2)
    before = counts.copy()
    monkeypatch.setattr('academic_agent.store.Store.__init__', lambda *a: pytest.fail('writable Store'))
    coverage = read_challenges(root, cid)
    assert counts == before
    assert coverage['rounds'][0].status == 'needs_assessment'
    assert coverage['rounds'][1].status == coverage['status'] == 'not_searched'
    expected = {'original beta'} if interruption == 'after_first_receipt' else {'original alpha', 'original beta'}
    gaps = [task for task in coverage['unresolved_tasks'] if task['reason'] == 'not_searched']
    assert {task['query'] for task in gaps} == expected
    assert all(task['round'] == 2 for task in gaps)


@pytest.mark.parametrize(('screening', 'level', 'task'), [
    ('pending', 'full_text', 'pending_screen'),
    ('exclude', 'full_text', 'excluded_with_reason'),
    ('include', 'needs_ocr', 'unread'),
])
def test_current_active_document_access_is_independent_of_screening(project, monkeypatch, screening, level, task):
    root, cid = project
    adapters(monkeypatch, lambda source, query, count: SearchPage(source=source, records=[
        Work(title='Original access-level followup', doi='10.1234/original-access')]))
    runner = Runner(root)
    result = challenge_candidate(runner, cid)
    wid, = result['retained_work_ids']
    if level == 'full_text':
        source = root / 'original-full-text.txt'
        source.write_text('Original full text attached before the screening decision.', encoding='utf-8')
    else:
        source = root / 'original-image-only.pdf'
        writer = PdfWriter()
        writer.add_blank_page(width=200, height=200)
        writer.write(source)
    assert runner.attach(wid, source)['level'] == level
    runner.store.screen(wid, screening, 'Original screening rationale')
    before = (root / 'research.db').read_bytes()
    monkeypatch.setattr('academic_agent.store.Store.__init__', lambda *a: pytest.fail('writable Store'))
    coverage = read_challenges(root, cid)
    hit, = coverage['hits']
    assert hit.access_level == level
    assert task in hit.statuses
    assert hit.screening_reason == 'Original screening rationale'
    assert (root / 'research.db').read_bytes() == before


@pytest.mark.parametrize('fault', ['escape', 'blob_hash', 'identity'])
def test_pending_hit_document_validation_uses_safe_readonly_blobs(project, monkeypatch, fault):
    root, cid = project
    adapters(monkeypatch, lambda source, query, count: SearchPage(source=source, records=[
        Work(title='Original verified attachment', doi='10.1234/original-attachment')]))
    runner = Runner(root)
    wid, = challenge_candidate(runner, cid)['retained_work_ids']
    source = root / 'original-attachment.txt'
    source.write_text('Original attachment bytes.', encoding='utf-8')
    document_id = runner.attach(wid, source)['document_id']
    with runner.store.connect() as connection:
        row = connection.execute('SELECT path,sha FROM documents WHERE id=?', (document_id,)).fetchone()
        if fault == 'escape':
            connection.execute('UPDATE documents SET path=? WHERE id=?', ('../escaped.json', document_id))
        elif fault == 'blob_hash':
            (root / row['path']).write_text('{}', encoding='utf-8')
        else:
            document = json.loads((root / row['path']).read_text(encoding='utf-8'))
            document['work_id'] = 'foreign_work'
            data = json.dumps(document)
            replacement = root / 'original-foreign-document.json'
            replacement.write_text(data, encoding='utf-8')
            connection.execute('UPDATE documents SET path=?,sha=? WHERE id=?',
                               (replacement.name, digest(data), document_id))
    monkeypatch.setattr('academic_agent.store.Store.__init__', lambda *a: pytest.fail('writable Store'))
    with pytest.raises(ValueError):
        read_challenges(root, cid)
