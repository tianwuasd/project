from pathlib import Path

from .executor import completed
from .pipeline import environment, sha256
from .service_state import CODE_ROOT, read_json
from .workbench import digest, status

ALGORITHM_FILES = ('pipeline.py', 'schema.py', 'images.py', 'segmentation.py', 'preprocessed.py')


def remaining_cases(work, job_id, cases):
    if job_id == 'latest':
        entries = work.entries()
        if not entries:
            raise ValueError('没有可接续的历史任务')
        job_id = Path(entries[0]['job']).name
    root_job = work.resolve(job_id)
    versions = environment()['versions']
    skipped = {}
    seen = set()
    wanted = {case['case_id']: case for case in cases}

    def collect(job, expected_request_hash=None):
        if job.name in seen:
            raise ValueError('历史接续关系存在循环')
        seen.add(job.name)
        if len(seen) > 32 or not job.resolve().is_relative_to((work.root / 'jobs').resolve()):
            raise ValueError('历史任务链或路径不合法')
        if status(job)['active']:
            raise ValueError('请先停止旧任务并确认进程退出，再接续剩余病例')
        request_hash = digest(job / 'request.json')
        if read_json(job / 'receipt.json', {}).get('request_sha256') != request_hash:
            raise ValueError('旧任务请求完整性校验失败')
        if expected_request_hash and expected_request_hash != request_hash:
            raise ValueError('旧接续任务引用改变')
        request = read_json(job / 'request.json')
        if request.get('output_format') != 'slicer-seg-v1' or request.get('preprocessing_source'):
            raise ValueError('接续仅支持原始 CT 的可编辑标签任务')
        code = (job / 'code').resolve()
        for relative, expected in request['code_hashes'].items():
            path = (code / relative).resolve()
            if not path.is_relative_to(code) or sha256(path) != expected:
                raise ValueError('旧任务冻结代码发生变化')
        for name in ALGORITHM_FILES:
            if sha256(code / 'aaoca_seg' / name) != sha256(CODE_ROOT / 'aaoca_seg' / name):
                raise ValueError('模型调用、标签定义或导出算法改变，不能接续旧成果')
        identity = read_json(job / 'identity.json', {})
        if identity and (identity.get('versions') != versions or identity.get('source') !=
                         {p.name: sha256(p) for p in (code / 'aaoca_seg').glob('*.py')}):
            raise ValueError('旧任务依赖或执行代码身份不符')
        parent = request.get('continuation')
        if parent:
            collect(work.resolve(parent['source_job']), parent['source_request_sha256'])
        output = Path(request['output']).resolve()
        inputs = read_json(job / 'inputs.json', {})
        for old_case in request['cases']:
            key = old_case['case_id']
            if key not in wanted:
                continue
            if Path(old_case['image']).resolve() != Path(wanted[key]['image']).resolve():
                raise ValueError('同一体积代号的原始输入路径改变')
            stat = Path(old_case['image']).stat()
            if old_case.get('source_stat') != {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}:
                raise ValueError('旧任务的原始输入大小或时间发生变化')
            expected_source = inputs.get(key) or old_case.get('source_sha256')
            if expected_source and sha256(old_case['image']) != expected_source:
                raise ValueError('旧任务冻结的原始输入哈希改变')
            folder = output / 'cases' / key
            pointer = read_json(folder / 'latest-result.json', {})
            if not pointer.get('directory'):
                continue
            attempt = Path(pointer['directory']).resolve()
            if not attempt.is_relative_to(folder.resolve()):
                raise ValueError('历史结果路径越界')
            result = read_json(attempt / 'result.json', {})
            inference = read_json(attempt / 'inference.json', {})
            if not identity or result.get('identity') != identity or inference.get('identity') != identity:
                raise ValueError('历史标签的执行身份不符')
            if not completed(attempt / 'result.json', old_case, attempt / 'inference.json'):
                raise ValueError('历史标签或原始 CT 校验失败，请先核查旧成果')
            label = attempt / 'aaoca_11.seg.nrrd'
            skipped[key] = {'case_id': key, 'result': str(attempt / 'result.json'),
                            'label': str(label), 'sha256': sha256(label)}

    collect(root_job)
    continuation = {'source_job': root_job.name, 'source_request_sha256': digest(root_job / 'request.json'),
                    'skipped': list(skipped.values())}
    return [case for case in cases if case['case_id'] not in skipped], continuation
