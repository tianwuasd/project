"""Adopt the latest, manually repaired legacy results without reading raw pixels."""
from collections import defaultdict
import csv
import json
from pathlib import Path
import sqlite3

from patient_pipeline.project_ingest import source_key
from patient_pipeline.project_identity import unit_key
from patient_pipeline.project_model import digest, issue, now, identity_evidence
from patient_pipeline.project_process import error_code


def read_json(path,default):
    return json.loads(path.read_text(encoding='utf-8-sig')) if path.is_file() else default


def rows(root,name):
    path=root/name
    if not path.exists():return []
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def load_legacy(store,root,control=None):
    root=root.resolve()
    if store.get('meta','legacy_adopted'):return store.get('meta','legacy_adopted')
    manifest=read_json(root/'processing_manifest.json',None)
    if not manifest:return None
    if not (root/'患者总表.csv').is_file():raise ValueError('旧版结果缺少患者总表，不能自动迁移')
    external=root!=store.root
    if external:store.put('meta','external_legacy_root',str(root))
    source_hashes={};legacy_units={};known_paths={};roster_sources={}
    def output_path(path):
        p=Path(path).resolve()
        if not p.is_relative_to(root):raise ValueError('旧版输出路径超出结果目录')
        return str(p) if external else p.relative_to(root).as_posix()
    def adopt_output(u,output_files):
        u['outputs']=[];u['proof']={}
        for f in output_files:
            if not f.is_file():
                u['status']='failed';u['error']='旧版登记的输出文件不存在';u['issues'].append(issue('LEGACY_OUTPUT_MISSING',u['error'],u['pid'],f));continue
            rel=output_path(f);u['outputs'].append(rel)
            meta=read_json(f.parent/'metadata.json',{})
            checksum=meta.get('output_sha256')
            u['proof'][rel]={'bytes':f.stat().st_size,'sha256':checksum,'basis':'既有回读验证记录；迁移时核对存在和大小'}
            sources=meta.get('sources_in_slice_order') or ([meta['source']] if meta.get('source') else [])
            for src in sources:
                if src.get('path') and src.get('sha256'):source_hashes[source_key(src['path'])]=src['sha256']
        if external:u['external_root']=str(root)
    for p in rows(root,'患者总表.csv'):
        pid=p['患者编号'];folder=root/p['患者数据目录'];record=read_json(folder/'patient.json',{})
        roster=record.get('clinical_roster') or {'name':p.get('姓名',''),'aliases':[pid],'conflict':False}
        roster.update(patient_id=pid,origin='legacy',confirmed=True)
        for src in roster.get('sources',[]):
            if src.get('path'):roster_sources.setdefault(src['path'],{})[pid]=roster
        store.put('legacy_patients',pid,roster,False);store.put('patients',pid,roster,False)
        if folder.exists():store.put('folders',pid,output_path(folder),False)
    for path,roster in roster_sources.items():
        store.put_source({'key':source_key(path),'path':path,'kind':'ROSTER','pid':'','legacy':True,
                          'roster':roster,'issues':[]},False)
    phase_rows=rows(root,'CT期相输出表.csv');phases=defaultdict(list)
    for p in phase_rows:phases[(p['原检查UID'],p['原序列UID'])].append(p)
    for row in rows(root,'CT序列表.csv'):
        pid=row['患者编号'];key='CT:'+row['检查UID']+':'+row['序列UID']
        success=row['转换状态']=='成功';msg=row.get('错误','')
        u={'key':key,'pid':pid,'kind':'CT','status':'success' if success else 'failed','legacy':True,
           'signature':'legacy','sources':[],'source_count':int(row.get('去重后文件数') or 0),'unique_files':int(row.get('去重后文件数') or 0),
           'slices':int(row.get('声明切片或帧数') or 0),'outputs':[],'issues':[],'repairs':[],'legacy_row':row}
        if success:
            phase=phases.get((row['检查UID'],row['序列UID']),[])
            files=[root/p['NIfTI文件'] for p in phase] if phase else [root/row['输出目录']/'image.nii.gz']
            adopt_output(u,files)
            if phase:u['phase_count']=len(phase);u['repairs']=[{'type':'accepted_legacy_phase','evidence':phase}]
        else:u['error']=msg;u['issues']=[issue(error_code(msg,'CT'),msg,pid,row.get('序列UID',''))]
        legacy_units[key]=u
    us_rows=rows(root,'超声文件表.csv')
    for n,row in enumerate(us_rows):
        if control and n%50==0:control.progress('headers',n,len(us_rows),'接纳已有超声输出与校验记录')
        src=source_key(row['源文件']);pid=row['患者编号']
        # Resolve the SOP from the old index below; preserve a path-key fallback for missing headers.
        key='US:'+digest(src)
        u={'key':key,'pid':pid,'kind':'US','status':'success' if row['转换状态']=='成功' else 'failed',
           'legacy':True,'signature':'legacy','sources':[src],'source_count':1,'unique_files':1,
           'frames':int(row.get('帧数') or 1),'image_kind':'video' if row.get('类型')=='视频' else 'still',
           'outputs':[],'issues':[],'repairs':[],'legacy_row':row}
        if u['status']=='success':
            folder=root/row['输出目录'];file=folder/'frames.npz'
            if not file.exists():file=folder/'image.png'
            adopt_output(u,[file])
        else:u['error']=row.get('错误','');u['issues'].append(issue(error_code(u['error'],'US'),u['error'],pid,row['源文件']))
        known_paths[src]=u
    index=root/'details/文件索引.sqlite'
    if index.exists():
        db=sqlite3.connect('file:'+index.as_posix()+'?mode=ro',uri=True)
        try:
            total=db.execute('SELECT count(*) FROM files').fetchone()[0]
            for n,(_,pid,_,_,_,_,payload) in enumerate(db.execute('SELECT * FROM files')):
                if control and n%500==0:control.progress('headers',n,total,'接纳旧版索引与已确认映射（不重新转换）')
                h=json.loads(payload);src=source_key(h['path'])
                r={'key':src,'path':h['path'],'kind':'DICOM','stamp':[h['bytes'],h['mtime_ns']],
                   'header':h,'pid':pid,'verified':True,'issues':[],'identity_issues':[],
                   'match_basis':'此前已确认关联','legacy':True}
                if src in source_hashes:r['sha256']=source_hashes[src]
                key=unit_key(r);r['unit_key']=key;store.put_source(r,False)
                # Preserve per-file confirmed identity even if original tags contain pre-merge aliases.
                store.put('confirmed_links',src,{'pid':pid,'basis':'latest legacy index','evidence':identity_evidence(r)},False)
                if src in known_paths:
                    u=known_paths.pop(src);u['key']=key;legacy_units[key]=u
                if key.startswith('CT:') and key in legacy_units:legacy_units[key]['sources'].append(src)
                if key.startswith('FILE:'):
                    legacy_units[key]={'key':key,'pid':pid,'kind':'FILE','status':'unsupported','legacy':True,
                        'signature':'legacy','sources':[src],'source_count':1,'outputs':[],
                        'issues':[issue('MODALITY_NOT_CONVERTED','该影像类型仅登记，未转换',pid,h['path'],'warning')]}
                if n%1000==0:store.commit()
        finally:db.close()
    for src,u in known_paths.items():
        legacy_units[u['key']]=u
        store.put_source({'key':src,'path':u['legacy_row']['源文件'],'kind':'DICOM','pid':u['pid'],
                          'unit_key':u['key'],'legacy':True,'verified':True,'issues':[],'identity_issues':[],
                          'header':{'tags':{'Modality':'US','PatientID':u['pid']}}},False)
    extras=[]
    for i in read_json(root/'details/问题清单.json',[]):
        code=i.get('code','');src=source_key(i['source']) if i.get('source') else ''
        if code=='DICOM_HEADER_FAILED':
            pid=i.get('patient_id','');key='US:'+digest(src) if '超声' in i.get('message','') or 'FrameTime' in i.get('message','') else 'FILE:'+digest(src)
            converted_issue={**i,'code':'US_TIMING_INVALID' if key.startswith('US:') else code}
            previous=store.source(src)
            if previous and previous.get('unit_key')!=key:legacy_units.pop(previous.get('unit_key'),None)
            r={'key':src,'path':i.get('source',''),'kind':'DICOM','pid':pid,'legacy':True,'verified':True,
               'read_error':i['message'],'issues':[converted_issue],'identity_issues':[],'unit_key':key}
            store.put_source(r,False)
            legacy_units[key]={'key':key,'kind':key.split(':')[0],'pid':pid,'status':'failed','legacy':True,
                              'outputs':[],'sources':[src],'source_count':1,'issues':[converted_issue],'error':i['message'],'signature':'legacy'}
        elif code in {'US_PHYSICAL_SCALE_UNKNOWN','US_SCALE_UNKNOWN','US_CALIBRATION_REVIEW','CT_PIXEL_PAYLOAD'}:
            r=store.source(src)
            if r and r.get('unit_key') in legacy_units:
                legacy_units[r['unit_key']]['issues'].append({**i,'severity':'warning' if 'SCALE' in code or 'CALIBRATION' in code else 'error'})
            else:extras.append(i)
        elif i.get('severity')=='info' and code!='PDF_TEXT_INCOMPLETE_NOTE':extras.append(i)
        elif code=='SOP_CONFLICT':
            extras.append(i)
    for row in rows(root,'病例文档表.csv'):
        if control:control.checkpoint()
        src=source_key(row['源文件']);pid=row['患者编号'];verified=row.get('关联已核实','').lower()=='true'
        text=root/row['输出文本'] if row.get('输出文本') else None
        meta=read_json(text.with_suffix('.json'),{}) if text else {}
        hashed=meta.get('sha256') or digest(src);key='PDF:'+hashed
        low=json.loads(row.get('需OCR页码') or '[]')
        r={'key':src,'path':row['源文件'],'kind':'PDF','pid':pid,'verified':verified,'legacy':True,
           'unit_key':key,'sha256':hashed,'text_file':output_path(text) if text else '',
           'pages':int(row.get('总页数') or 0),'low_text_pages':low,'document_ids':[pid] if pid else [],
           'document_name':'','match_basis':row.get('关联依据',''),'issues':[],'identity_issues':[]}
        store.put_source(r,False)
        if verified:store.put('confirmed_links',src,{'pid':pid,'basis':'latest legacy documents','evidence':identity_evidence(r)},False)
        u={'key':key,'pid':pid,'kind':'PDF','status':'success' if verified else 'pending','legacy':True,
           'signature':'legacy','sources':[src],'source_count':1,'pages':r['pages'],'outputs':[],
           'low_text_pages':low,'issues':[],'match_basis':r['match_basis']}
        if text:adopt_output(u,[text])
        if low:u['issues'].append(issue('PDF_TEXT_INCOMPLETE_NOTE','部分页面文字提取不足；仅作备注',pid,row['源文件'],'info'))
        if not verified:u['issues'].append(issue('DOCUMENT_UNMATCHED','历史PDF关联尚未确认',pid,row['源文件']))
        legacy_units[key]=u
    for key,u in list(store.items('units')):
        if u.get('legacy') and key not in legacy_units:
            store.db.execute("DELETE FROM records WHERE collection='units' AND key=?",(key,))
    for key,u in legacy_units.items():store.put('units',key,u,False)
    store.put('meta','legacy_extra_issues',extras,False);store.put('meta','legacy_manifest',manifest,False)
    result={'root':str(root),'adopted':now(),'patients':len(list(store.items('patients'))),
            'units':len(legacy_units),'source_count':store.db.execute('SELECT count(*) FROM sources').fetchone()[0],
            'originals_modified':False,'pixel_reconversion':False}
    store.put('meta','legacy_adopted',result,False);store.commit();return result
