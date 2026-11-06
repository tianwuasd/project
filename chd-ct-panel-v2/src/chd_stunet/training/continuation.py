"""Explicit old-run continuation: exact source audit, new immutable run.

No version-string exemptions, snapshot edits or signature deletion. The source
lock is acquired without writing its bytes; full model/SGD/scaler/cursors are
retained. Only the audited membership policy/source identity can change.
"""
import copy
import json
from pathlib import Path
from ..observability.locking import try_lock,unlock
from ..observability.runlog import write_json
from .contracts import digest
from .code_snapshot import freeze_code
from .checkpoints import read_checkpoint,commit_payload
from .engine import validate_state

LEGACY_052_SOURCES={'c99e2ce3b34207b8a7da40520324d31fb6740579d527450bb5cadad1db685938'}


def _check_contract(old,new):
    for signature in (old,new):
        if signature['identity']!=digest({k:v for k,v in signature.items() if k!='identity'}):
            raise ValueError('Continuation signature digest differs')
    for field in ('data','versions','weight_sha256'):
        if old[field]!=new[field]:raise ValueError('Continuation '+field+' differs')
    previous=copy.deepcopy(old['recipe']);current=copy.deepcopy(new['recipe'])
    if old['producer_ast_sha256']!=new['producer_ast_sha256']:
        if digest(old['producer_ast_sha256']) not in LEGACY_052_SOURCES:
            raise ValueError('Unknown continuation producer; no version-based bypass')
        if 'membership_policy' in previous:raise ValueError('Unexpected legacy membership policy')
        if current.pop('membership_policy',None)!='step-boundary-v1':raise ValueError('Unknown continuation membership policy')
    if previous!=current:raise ValueError('Continuation recipe differs')


def read_continuation(source,signature,settings):
    source=Path(source).resolve(strict=True)
    print('继续训练：核对旧运行合同和冻结代码；源文件只读',flush=True)
    # Existing training locks contain at least one byte. Open without append
    # or truncate so an audited import leaves EVERY source byte unchanged.
    with (source/'.training.lock').open('r+b') as stream:
        if not try_lock(stream):raise RuntimeError('Continuation source is active; stop it safely first')
        try:
            if (source/'continuation.pending.json').exists():raise ValueError('Source has an incomplete checkpoint import')
            old=json.loads((source/'signature.json').read_text(encoding='utf-8'))
            _check_contract(old,signature);freeze_code(source,old,create=False)
            payloads={}
            for role in ('epoch.json','best.json','latest.json'):
                if not (source/'checkpoints'/role).is_file():
                    if role in ('latest.json','epoch.json'):raise ValueError('Missing source checkpoint pointer '+role)
                    continue
                payload=read_checkpoint(source/'checkpoints',old,role)
                validate_state(payload['state'],settings)
                print('继续训练：已校验 '+role+'；epoch='+str(payload['state']['epoch'])+
                      '；phase='+payload['state']['phase']+'；step='+str(payload['state']['step']),flush=True)
                record=payload.get('distributed',{})
                if (not record.get('members') or len(set(record['members']))!=len(record['members'])
                        or len(record['members'])!=len(record.get('rng_by_rank',[]))):
                    raise ValueError('Invalid continuation rank RNG inventory')
                payloads[role]=payload
            if payloads['epoch.json']['state']['step']!=0 or payloads['epoch.json']['state']['phase']!='train':
                raise ValueError('Continuation epoch pointer is not a full boundary')
            if payloads['latest.json']['state'].get('best_epoch') is not None and 'best.json' not in payloads:
                raise ValueError('Missing source best checkpoint pointer')
            controller_path=source/'control/controller.json'
            controller=json.loads(controller_path.read_text(encoding='utf-8')) if controller_path.exists() else {}
            # Performance baselines belong to the old placement, not this one.
            controller={k:controller[k] for k in ('pressure_recoveries','worker_cap','reserve_at_entry') if k in controller}
            for key,value in controller.items():
                if value is None and key=='worker_cap':continue
                if type(value) is not int or value<0:raise ValueError('Invalid source controller '+key)
            pointer=json.loads((source/'checkpoints/latest.json').read_text(encoding='utf-8'))
            provenance=dict(source=str(source),source_signature=old['identity'],new_signature=signature['identity'],
                source_producer_digest=digest(old['producer_ast_sha256']),source_checkpoint=pointer,
                cursor={k:payloads['latest.json']['state'][k] for k in ('epoch','phase','step','attempts','updates','overflows')},
                policy='step-boundary-v1',model_optimizer_scaler_cursor_preserved=True)
            return dict(payloads=payloads,controller=controller,provenance=provenance)
        finally:unlock(stream)


def install_continuation(output,bundle,signature):
    output=Path(output)
    write_json(output/'continuation.pending.json',bundle['provenance'])
    for role,payload in bundle['payloads'].items():
        commit_payload(output/'checkpoints',dict(payload,signature=signature),role)
    write_json(output/'history.json',bundle['payloads']['latest.json']['state']['history'])
    controller=dict(pressure_recoveries=0,worker_cap=None,deferred=[],baseline=None)
    controller.update(bundle['controller'])
    write_json(output/'control/controller.json',controller)
    write_json(output/'continuation.json',bundle['provenance'])
    (output/'continuation.pending.json').unlink()
