from contextlib import contextmanager
from functools import wraps
import hashlib
import json
from pathlib import Path
import time
from uuid import uuid4

from ..observability.locking import run_lock, try_lock, unlock
from ..observability.runlog import write_json

MODE = 'shared_active_fold_v1'
SUFFIXES = ('.b2nd', '_seg.b2nd', '.pkl')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def is_shared(root):
    signature = Path(root)/'signature.json'
    return signature.is_file() and read_json(signature).get('method', {}).get('storage_mode') == MODE


def case_ids(values):
    ids = list(values)
    if any(not isinstance(cid, str) or not cid or cid in ('.', '..') or '/' in cid or '\\' in cid for cid in ids):
        raise ValueError('共享缓存病例标识包含非法路径')
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('共享缓存病例必须非空且不重复')
    return sorted(ids)


def ready_view(root, fold):
    root = Path(root)
    state = read_json(root/'shared_cache.json')
    if state.get('schema') != 1 or state.get('mode') != MODE:
        raise ValueError('未知共享缓存协议')
    if state.get('state') != 'ready':
        raise ValueError('共享缓存正在构建或未完成，请先恢复预处理')
    if state.get('fold') != fold:
        raise ValueError(f'共享缓存当前为第{state.get("fold")}折；请先 prepare --resume --cache-fold {fold} 切换后再训练')
    plan = read_json(root/f'fold_{fold}/preprocessing_plan.json')
    signature = read_json(root/'signature.json')
    if state['contract'] != digest(dict(run=signature['identity'], plan=plan)):
        raise ValueError('活动缓存和本折预处理参数不一致')
    case_ids(state['case_ids'])
    return state


def data_directory(root, fold, *, check=True):
    root = Path(root)
    if not is_shared(root):
        return root/f'fold_{fold}'/'data'
    if check:
        ready_view(root, fold)
    path = root/'shared_data'
    if path.is_symlink() or path.resolve() != root.resolve()/'shared_data':
        raise ValueError('共享数据目录不允许链接到其他位置')
    return path


@contextmanager
def reader_gate(root, *, read_only=False):
    # 登记锁只串行化读者登记及写者准入；文件锁随进程退出自动释放。
    with (Path(root)/'.cache-reader-gate.lock').open('rb' if read_only else 'a+b') as stream:
        if not read_only and stream.tell() == 0:
            stream.write(b'0')
            stream.flush()
        while not try_lock(stream):
            time.sleep(.01)
        try:
            yield
        finally:
            unlock(stream)


def reader_directory(root, *, create=True):
    directory = Path(root)/'.cache-readers'
    if directory.is_symlink() or directory.resolve() != Path(root).resolve()/'.cache-readers':
        raise ValueError('缓存读取登记目录不能链接到其他位置')
    if create:
        directory.mkdir(exist_ok=True)
    return directory


@contextmanager
def reader_claim(root, fold, *, generation=None, writer_token=None, contract=None):
    root = Path(root)
    if not is_shared(root):
        yield
        return
    stream = None
    with reader_gate(root):
        writer = read_json(root/'.cache-writer.json') if (root/'.cache-writer.json').is_file() else None
        if writer_token is not None:
            state = read_json(root/'shared_cache.json')
            if (not writer or writer['token'] != writer_token or state.get('state') != 'building'
                    or state.get('fold') != fold or state.get('contract') != contract):
                raise ValueError('共享缓存写入任务与当前构建请求不一致')
        else:
            if writer:
                raise ValueError('共享缓存存在写入任务，请先完成或恢复预处理')
            state = ready_view(root, fold)
            if generation is not None and state['generation'] != generation:
                raise ValueError('共享缓存已经重新构建；原供数进程必须退出')
        path = reader_directory(root)/(uuid4().hex+'.lock')
        stream = path.open('x+b')
        stream.write(b'0')
        stream.flush()
        if not try_lock(stream):
            stream.close()
            raise RuntimeError('无法登记共享缓存读取锁')
    try:
        yield
    finally:
        with reader_gate(root):
            unlock(stream)
            stream.close()
            path.unlink()


def check_readers(root, *, clean=True):
    for path in reader_directory(root,create=clean).iterdir():
        if (path.suffix != '.lock' or len(path.stem) != 32
                or any(c not in '0123456789abcdef' for c in path.stem)
                or path.is_symlink() or not path.is_file()):
            raise ValueError('缓存读取登记目录包含未知文件：'+str(path))
        with path.open('r+b' if clean else 'rb') as stream:
            if not try_lock(stream):
                raise RuntimeError('仍有训练或供数进程读取共享缓存，请等待进程退出')
            unlock(stream)
        # 登记文件仍在，但系统锁已释放，说明原读取进程已经退出。
        if clean:
            path.unlink()


@contextmanager
def existing_cache_lock(root):
    with (Path(root)/'.shared-cache.lock').open('rb') as stream:
        if not try_lock(stream):
            raise RuntimeError('共享缓存仍被其他任务占用')
        try:
            yield
        finally:
            unlock(stream)


@contextmanager
def lease(root, *, fold=None, enabled=None, read_only=False):
    root = Path(root)
    shared = is_shared(root) if enabled is None else enabled
    if not shared:
        yield
        return
    with existing_cache_lock(root) if read_only else run_lock(root/'.shared-cache.lock'):
        if read_only:
            with reader_gate(root,read_only=True):
                check_readers(root,clean=False)
                yield
            return
        if fold is not None:
            ready_view(root, fold)
            yield
        else:
            with reader_gate(root):
                check_readers(root)
                token=uuid4().hex
                write_json(root/'.cache-writer.json',dict(token=token))
            try:
                yield token
            finally:
                with reader_gate(root):
                    if read_json(root/'.cache-writer.json')['token'] != token:
                        raise ValueError('共享缓存构建归属发生变化')
                    (root/'.cache-writer.json').unlink()


def protect_training(function):
    @wraps(function)
    def protected(prepared, *args, **kwargs):
        with lease(prepared, fold=kwargs.get('fold', 0)):
            return function(prepared, *args, **kwargs)
    return protected


def protect_rank(function):
    @wraps(function)
    def protected(job_path, rank):
        job = read_json(job_path)
        with reader_claim(job['prepared'], job['options']['fold']):
            return function(job_path, rank)
    return protected


def protect_case(function):
    @wraps(function)
    def protected(job):
        if not job.get('shared_cache'):
            return function(job)
        with reader_claim(Path(job['foldroot']).parent, job['plan']['fold'],
                          writer_token=job['cache_write_token'],contract=job['contract']):
            return function(job)
    return protected


def begin_view(root, fold, selected, contract):
    root = Path(root)
    ids = case_ids(selected)
    if fold not in (0, 1, 2):
        raise ValueError('共享缓存只支持第0、1、2折')
    previous = read_json(root/'shared_cache.json') if (root/'shared_cache.json').is_file() else None
    owned = set(ids)
    if previous:
        if previous.get('schema') != 1 or previous.get('mode') != MODE:
            raise ValueError('既有共享缓存的归属未知')
        owned.update(case_ids(previous['owned_case_ids']))
    directory = data_directory(root, fold, check=False)
    expected = {cid+suffix for cid in owned for suffix in SUFFIXES}
    if directory.exists():
        for path in directory.iterdir():
            if path.name not in expected or not path.is_file() or path.is_symlink():
                raise ValueError('共享目录存在未登记文件，保持原状：'+str(path))
    write_json(root/'shared_cache.json', dict(schema=1, mode=MODE, state='building', fold=fold,
               contract=contract, case_ids=ids, owned_case_ids=sorted(owned), generation=uuid4().hex,
               updated_at=time.time()))


def commit_view(root, fold, selected, contract):
    root = Path(root)
    ids = case_ids(selected)
    state = read_json(root/'shared_cache.json')
    if (state.get('state') != 'building' or state.get('fold') != fold
            or state.get('contract') != contract or state.get('case_ids') != ids):
        raise ValueError('共享缓存提交与构建请求不一致')
    directory = data_directory(root, fold, check=False)
    for cid in ids:
        marker = read_json(root/f'fold_{fold}/completion/{cid}.json')
        expected = {cid+suffix for suffix in SUFFIXES}
        if (not marker.get('success') or marker.get('fold') != fold or marker.get('case_id') != cid
                or marker.get('contract') != contract or set(marker['file_sha256']) != expected):
            raise ValueError('共享缓存病例未按本折提交：'+cid)
        for name in expected:
            if not (directory/name).is_file() or (directory/name).is_symlink():
                raise ValueError('共享缓存缺少已提交数据：'+name)
    # 只释放此前登记、当前不再选用的缓存病例，原始数据和未知文件保持完整。
    for cid in set(state['owned_case_ids'])-set(ids):
        for suffix in SUFFIXES:
            path = directory/(cid+suffix)
            if path.is_symlink():
                raise ValueError('旧缓存文件不能是链接：'+str(path))
            path.unlink(missing_ok=True)
    write_json(root/'shared_cache.json', dict(state, state='ready', owned_case_ids=ids, updated_at=time.time()))
