"""Owned CHD V2 STU-Net roots. Validation never creates files."""
from contextlib import contextmanager
import json
import os
import re
from pathlib import Path
import stat
from uuid import uuid4

OWNER='chd-ct-panel-v2/stunet'
MARKER='.ct-v2-stunet.json'
INIT_LOCK='.ct-v2-stunet-init.lock'


def checked_path(value):
    p=Path(value).expanduser().absolute()
    if '..' in p.parts or p==Path(p.anchor):raise ValueError('请选择明确的V2方法专用目录')
    for part in (*reversed(p.parents),p):
        try:info=part.lstat()
        except FileNotFoundError:continue
        if stat.S_ISLNK(info.st_mode) or getattr(info,'st_file_attributes',0)&0x400:
            raise ValueError('V2 STU-Net运行路径不能经过链接或目录联接')
    return p.resolve()


def _read(root):
    p=checked_path(root/MARKER)
    if not p.exists():return None
    try:
        value=json.loads(p.read_text(encoding='utf-8'))
        if not isinstance(value,dict):raise ValueError('身份记录必须是对象')
        return value
    except (OSError,ValueError) as error:raise ValueError('V2 STU-Net身份文件损坏或尚在初始化') from error


def locations(runtime,results=None):
    runtime=checked_path(runtime)
    record=_read(runtime)
    if results is None:
        if record is None:
            from server_profile import load
            results=load()['results']
        else:results=record.get('results')
    if not results:raise ValueError('缺少V2 STU-Net结果根目录')
    results=checked_path(results)
    if runtime==results or runtime in results.parents or results in runtime.parents:
        raise ValueError('V2 STU-Net环境与结果目录必须互相独立')
    code=Path(__file__).resolve().parents[1]
    host=code.parent
    if (host/'src/chd_ct_v2/panel_v2/identity.py').is_file():
        # Source may be nested under V1. Protect the enclosing checkout, while
        # standalone/frozen copies continue to protect their own code root.
        standalone=host.parent if host.name=='v2' and host.parent.name=='chd-ct-reproduction' else host
        code=next((p for p in (host,*host.parents) if (p/'.git').exists()),standalone)
    for p in (runtime,results):
        if p==code or p in code.parents or code in p.parents:
            raise ValueError('运行目录必须与代码目录分开')
    return runtime,results


def check(runtime,results=None,data=None):
    runtime,results=locations(runtime,results)
    if data:
        source=checked_path(data)
        if any(p==source or p in source.parents or source in p.parents for p in (runtime,results)):
            raise ValueError('V2 STU-Net运行根目录不能与源数据重叠')
    identities=set()
    for role,root in [('runtime',runtime),('results',results)]:
        record=_read(root)
        if record is None:
            ignored={INIT_LOCK} if role=='runtime' else set()
            if root.exists() and any(p.name not in ignored for p in root.iterdir()):
                raise ValueError('目录不属于整合版V2 STU-Net；拒绝接管V1、原CT V2或旧独立模型的状态')
            continue
        if (not isinstance(record,dict) or record.get('owner')!=OWNER or record.get('role')!=role
                or record.get('runtime')!=str(runtime) or record.get('results')!=str(results)
                or set(record)!={'owner','role','instance','runtime','results'}
                or not isinstance(record.get('instance'),str)
                or re.fullmatch('[0-9a-f]{32}',record['instance']) is None):
            raise ValueError('V2 STU-Net实例身份或根目录配对不符')
        identities.add(record['instance'])
    if len(identities)>1:raise ValueError('V2 STU-Net根目录来自不同实例')
    return next(iter(identities),None)


def bind_runtime(runtime,requested_results,default_results):
    """Readonly root resolution, retaining a registered instance's result root."""
    runtime=checked_path(runtime)
    record=_read(runtime)
    results=requested_results or (record.get('results') if record else default_results)
    check(runtime,results)
    return str(checked_path(results))


def request_roots(config):
    """The instance owns the base; stage outputs are strict children of it."""
    runtime,base=locations(config['runtime'],config.get('results_base') or config['results'])
    output=checked_path(config['results'])
    if config.get('catalog_schema')==2:
        stage=config.get('stage')
        if stage not in {'environment','prepare','train','predict','assessment'} or output!=base/'pipeline_v2'/stage:
            raise ValueError('V2 STU-Net阶段输出必须属于登记结果根的对应板块')
    elif output!=base:
        raise ValueError('V2 STU-Net环境输出与实例结果根不符')
    check(runtime,base,config.get('data'))
    return runtime,base


@contextmanager
def _lock(runtime):
    with (runtime/INIT_LOCK).open('a+b') as stream:
        stream.seek(0,2)
        if stream.tell()==0:stream.write(b'0');stream.flush()
        stream.seek(0)
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(stream.fileno(),msvcrt.LK_LOCK,1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(),fcntl.LOCK_EX)
        try:yield
        finally:
            stream.seek(0)
            if os.name=='nt':msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(stream.fileno(),fcntl.LOCK_UN)


def claim(runtime,results=None,data=None):
    check(runtime,results,data)
    runtime,results=locations(runtime,results)
    runtime.mkdir(parents=True,exist_ok=True)
    with _lock(runtime):
        instance=check(runtime,results,data) or uuid4().hex
        for role,root in [('runtime',runtime),('results',results)]:
            expected=dict(owner=OWNER,role=role,instance=instance,runtime=str(runtime),results=str(results))
            root.mkdir(parents=True,exist_ok=True)
            p=checked_path(root/MARKER)
            try:
                with p.open('x',encoding='utf-8') as stream:
                    json.dump(expected,stream);stream.flush();os.fsync(stream.fileno())
            except FileExistsError:
                if _read(root)!=expected:raise ValueError('V2 STU-Net正在被不同实例初始化') from None
        return check(runtime,results,data)
