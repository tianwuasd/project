"""Rebuild association from durable evidence, with explicit optional name matching."""
from collections import defaultdict
import copy
import json
from pathlib import Path

from patient_pipeline.clinical import normal_id
from repair_policy import name_variants
from patient_pipeline.project_model import digest, issue, identity_evidence


def canonical_names(value):return {s.casefold() for s in name_variants(value) if s}


def source_confirmation(store,record):
    def valid(confirmation,source):
        if not confirmation:return False
        acceptable={identity_evidence(source)}
        # Accept the earlier deployed PDF proof only against the unchanged catalog version.
        if source.get('kind')=='PDF':acceptable.add(digest(['PDF',source.get('sha256'),source.get('document_ids',[])]))
        return confirmation.get('evidence') in acceptable
    direct=store.get('confirmed_links',record['key'])
    if valid(direct,record):return {**direct,'evidence':identity_evidence(record)}
    # Relocation inherits a proof only when identity-bearing content is identical. PDF proof
    # includes its filename name to avoid applying a blank named PDF to a different person.
    if record.get('kind')=='PDF' and record.get('sha256'):
        candidates=store.unit_sources('PDF:'+record['sha256'])
    else:
        sop=record.get('header',{}).get('tags',{}).get('SOPInstanceUID')
        candidates=[json.loads(v[0]) for v in store.db.execute('SELECT payload FROM sources WHERE sop=?',(sop,))] if sop else []
    matches=[]
    for candidate in candidates:
        c=store.get('confirmed_links',candidate['key'])
        if valid(c,candidate) and identity_evidence(candidate)==identity_evidence(record):matches.append(c)
    if matches and len({c['pid'] for c in matches})==1:
        return {**matches[0],'evidence':identity_evidence(record)}
    return None


def unit_key(r):
    if r.get('read_error') and r.get('previous_unit_key'):return r['previous_unit_key']
    if r.get('legacy') and r.get('unit_key') and not r.get('header'):return r['unit_key']
    if r['kind']=='PDF':return 'PDF:'+r.get('sha256',digest(r['key']))
    h=r.get('header') or r.get('previous_header') or {};t=h.get('tags',{})
    if t.get('Modality')=='CT' and t.get('StudyInstanceUID') and t.get('SeriesInstanceUID'):
        return 'CT:'+t['StudyInstanceUID']+':'+t['SeriesInstanceUID']
    if t.get('Modality')=='US':return 'US:'+(t.get('SOPInstanceUID') or digest(r['key']))
    return 'FILE:'+digest(r['key'])


def resolve_identity(store,options,control):
    patients={k:copy.deepcopy(v) for k,v in store.items('legacy_patients')}
    registry_issues=[]
    for r in store.sources(kind='ROSTER'):
        for ident,entry in r.get('roster',{}).items():
            entry=copy.deepcopy(entry);entry['origin']='roster'
            if ident in patients:
                old=patients[ident]
                if old.get('name') and entry.get('name') and old['name']!=entry['name'] and not old.get('confirmed'):
                    old['conflict']=True
                old['aliases']=sorted(set(old.get('aliases',[ident])+entry.get('aliases',[ident])))
                old.setdefault('sources',[]).extend(entry.get('sources',[]))
                if not old.get('name'):old['name']=entry.get('name','')
                old['conflict']=old.get('conflict',False) or entry.get('conflict',False)
            else:patients[ident]=entry
    policy=store.get('meta','identity_policy',{})
    if options.policy_path:
        policy=json.loads(Path(options.policy_path).read_text(encoding='utf-8'))
        store.put('meta','identity_policy',policy)
    for ident,override in policy.get('roster_overrides',{}).items():
        entry=patients.setdefault(ident,{'name':'','aliases':[ident]})
        entry.update(copy.deepcopy(override));entry.update(conflict=False,confirmed=True)
    for old,new in policy.get('id_aliases',{}).items():
        if new not in patients:raise ValueError(f'已确认映射的目标编号不存在：{new}')
        patients[new]['aliases']=sorted(set(patients[new].get('aliases',[new])+[old]))
    aliases=defaultdict(set)
    def refresh_aliases():
        aliases.clear()
        for ident,p in patients.items():
            for a in [ident]+p.get('aliases',[]):aliases[normal_id(a)].add(ident)
    refresh_aliases()
    # First collect explicit image/document identifiers. Name-only documents never invent a patient number.
    for n,r in enumerate(store.sources()):
        if n%200==0:control.checkpoint()
        h=r.get('header') or r.get('previous_header') or {};t=h.get('tags',{})
        ids=[]
        if r['kind']=='DICOM':
            ident=normal_id(t.get('PatientID')) or r.get('pid','')
            if ident:ids=[ident]
        elif r['kind']=='PDF':
            doc_ids=(r.get('previous_good') or r).get('document_ids',[])
            if len(doc_ids or [])==1:ids=doc_ids
            previous_pid=(r.get('previous_good') or {}).get('pid')
            if previous_pid and previous_pid not in patients:
                retained=store.get('patients',previous_pid)
                if retained:patients[previous_pid]=copy.deepcopy(retained);aliases[previous_pid].add(previous_pid)
        for ident in ids:
            if ident not in aliases:
                patients[ident]={'name':str(t.get('PatientName') or ''),'aliases':[ident],
                                 'origin':'image' if r['kind']=='DICOM' else 'document','conflict':False}
                aliases[ident].add(ident)
    names=defaultdict(set);anchor_names=defaultdict(set)
    for ident,p in patients.items():
        for name in [p.get('name','')]+p.get('name_aliases',[]):
            for variant in canonical_names(name):
                names[variant].add(ident)
                if p.get('origin') in {'roster','legacy'} or p.get('confirmed'):anchor_names[variant].add(ident)
    def named(name):
        if not name:return set()
        anchors=set().union(*(anchor_names[v] for v in canonical_names(name)))
        return anchors or set().union(*(names[v] for v in canonical_names(name)))
    def resolve(ident):
        hits=aliases.get(ident,set());return next(iter(hits)) if len(hits)==1 else ''
    used=set()
    for n,r in enumerate(store.sources()):
        if n%200==0:control.progress('documents',n,0,'按编号与姓名核对资料归属')
        r['identity_issues']=[];r['verified']=False;old_pid=r.get('pid','')
        if r['kind']=='ROSTER':continue
        h=r.get('header',{});t=h.get('tags',{})
        raw=normal_id(t.get('PatientID')) if r['kind']=='DICOM' else ''
        folder=h.get('folder_patient_id','')
        name=str(t.get('PatientName') or '') if r['kind']=='DICOM' else r.get('document_name','')
        hits=named(name);unique_name=next(iter(hits)) if len(hits)==1 else ''
        id_hits=set()
        declared=[raw] if raw else r.get('document_ids',[])
        for v in declared:
            if resolve(v):id_hits.add(resolve(v))
        ident=next(iter(id_hits)) if len(id_hits)==1 and all(resolve(v) for v in declared) else ''
        basis='编号一致' if ident else ''
        # Existing per-source owner confirmations are durable and separate from the toggle.
        confirmed=source_confirmation(store,r)
        confirmation_valid=bool(confirmed and confirmed.get('pid') in patients
            and confirmed.get('evidence')==identity_evidence(r))
        if confirmation_valid and confirmed.get('basis')=='unique_name' and hits!={confirmed['pid']}:
            confirmation_valid=False
        if confirmation_valid:
            ident=confirmed['pid'];basis='此前已确认关联'
        elif options.match_names and unique_name:
            # A unique roster name can resolve unknown external IDs. Two established patients with
            # contradictory names remain a conflict; no transliteration or fuzzy matching.
            contradictory={i for i in id_hits if patients[i].get('name') and i!=unique_name
                           and not canonical_names(patients[i]['name']) & canonical_names(name)}
            if not contradictory:ident=unique_name;basis='唯一姓名匹配'
        r['pid']=ident or (resolve(folder) if folder else '') or (old_pid if r.get('read_error') else '')
        if r['kind']=='DICOM':
            if raw and ident:
                mismatch=bool(folder and resolve(folder)!=ident and folder!=raw)
                if mismatch and basis not in {'唯一姓名匹配','此前已确认关联'}:
                    r['identity_issues'].append(issue('DIRECTORY_ID_CONFLICT','目录编号与影像内部编号不一致；未确认归属',r['pid'],r['path']))
                elif mismatch:
                    r['identity_issues'].append(issue('ID_RESOLVED_BY_NAME','编号差异已按姓名或历史确认关联；原编号保留',ident,r['path'],'info'))
                r['verified']=not any(i['severity']=='error' for i in r['identity_issues'])
            elif not r.get('read_error'):
                r['identity_issues'].append(issue('PATIENT_ID_MISSING','影像没有可靠的患者编号',r['pid'],r['path']))
            if ident and hits and ident not in hits and not confirmation_valid:
                r['verified']=False
                r['identity_issues'].append(issue('PATIENT_ID_NAME_CONFLICT','影像编号与姓名分别指向不同的已登记患者；需明确确认',r['pid'],r['path']))
        else:
            r['verified']=bool(ident and (declared or basis in {'唯一姓名匹配','此前已确认关联'}))
            if len(id_hits)>1 and basis not in {'唯一姓名匹配','此前已确认关联'}:r['verified']=False
            if ident and hits and ident not in hits and basis!='此前已确认关联':
                r['verified']=False
                r['identity_issues'].append(issue('DOCUMENT_PAGE_ID_CONFLICT','PDF编号与文件名分别指向不同的已登记患者；需明确确认',r['pid'],r['path']))
            if len(declared)>1 and (not ident or any(resolve(v)!=ident for v in declared)) and not confirmation_valid:
                r['verified']=False
                r['identity_issues'].append(issue('DOCUMENT_PAGE_ID_CONFLICT','PDF多页包含不能归入同一患者的编号；文件名不足以覆盖页内冲突',r['pid'],r['path']))
            if not r['verified']:
                r['identity_issues'].append(issue('DOCUMENT_UNMATCHED','PDF编号不唯一、缺少可核实编号或姓名关联尚未启用；文字已保留',r['pid'],r['path']))
        if r['pid'] and patients.get(r['pid'],{}).get('conflict'):
            r['verified']=False;r['identity_issues'].append(issue('ROSTER_ID_CONFLICT','同一主表编号存在不一致记录；需要明确合并规则',r['pid'],r['path']))
        r['match_basis']=basis;r['unit_key']=unit_key(r)
        if r['verified'] and basis=='唯一姓名匹配':
            store.put('confirmed_links',r['key'],{'pid':r['pid'],'basis':'unique_name',
                'evidence':identity_evidence(r),'name':name,'declared_ids':declared},False)
        if r['pid']:used.add(r['pid'])
        store.put_source(r,commit=False)
        if n%200==0:store.commit()
    # Persist the current registry, retaining legacy patients but not stale automatically inferred IDs.
    store.db.execute("DELETE FROM records WHERE collection='patients'")
    for ident,p in patients.items():
        if ident not in used and p.get('origin') in {'image','document'}:continue
        p['patient_id']=ident
        store.put('patients',ident,p,commit=False)
        if p.get('conflict'):registry_issues.append(issue('ROSTER_ID_CONFLICT','同一编号的主表记录不一致',ident))
    store.put('meta','registry_issues',registry_issues,commit=False);store.commit()
    collisions=store.db.execute("SELECT sop FROM sources WHERE sop IS NOT NULL AND sop!='' GROUP BY sop HAVING count(DISTINCT unit_key)>1 OR count(DISTINCT pid)>1").fetchall()
    for (sop,) in collisions:
        rows=store.db.execute('SELECT payload FROM sources WHERE sop=?',(sop,)).fetchall()
        for (payload,) in rows:
            r=json.loads(payload);r['verified']=False
            r['identity_issues'].append(issue('SOP_IDENTITY_CONFLICT','同一影像唯一编号跨不同患者或不同序列；仅暂停相关单元',r['pid'],r['path']))
            store.put_source(r,False)
    store.commit()
