"""Read patient rosters and locally extract clinical text, without inferring diagnoses."""
from collections import defaultdict
from datetime import date, datetime
import json
from pathlib import Path
import re
import unicodedata

from common import sha256, token, write_json
from patient_pipeline.runtime import fatal_io


def normal_id(value):
    if value is None:return ''
    if isinstance(value, float) and value.is_integer():return str(int(value))
    return unicodedata.normalize('NFKC',str(value)).strip()


def normalize_text(text):
    text=unicodedata.normalize('NFKC',text).replace('\r\n','\n').replace('\r','\n')
    # Keep clinical wording, negatives, tables and page boundaries; do not strip
    # sentences based on medical keywords or turn consent templates into labels.
    lines=[re.sub(r'[^\S\n]+',' ',line).strip() for line in text.split('\n')]
    return re.sub(r'\n{3,}','\n\n','\n'.join(lines)).strip()


ID_COLUMNS=['门诊号','患者编号','患者ID','PatientID','病人ID','病人编号','患者号','病历号']
NAME_COLUMNS=['姓名','患者姓名','病人姓名','PatientName']


def read_roster(paths,control=None):
    import openpyxl
    roster={}; issues=[]
    for path in paths:
        if control:control.checkpoint()
        try:
            book=openpyxl.load_workbook(path,read_only=True,data_only=True)
            recognized=False
            try:
                for sheet in book:
                    header=None;id_col=None
                    for row_no,cells in enumerate(sheet.iter_rows(),1):
                        if control:control.checkpoint()
                        values=[c.value for c in cells]
                        if header is None:
                            labels=[normal_id(v) for v in values]
                            hits=[labels.index(k) for k in ID_COLUMNS if k in labels]
                            if not hits:
                                if row_no>=20:break
                                continue
                            header=labels;id_col=hits[0];recognized=True;continue
                        if id_col>=len(values):continue
                        ident=normal_id(values[id_col])
                        if not ident:continue
                        fmt=cells[id_col].number_format
                        if isinstance(values[id_col],(int,float)) and re.fullmatch(r'0+',fmt or ''):
                            ident=ident.zfill(len(fmt))
                        fields={h:(v.isoformat() if isinstance(v,(date,datetime)) else v)
                                for h,v in zip(header,values) if h and v is not None}
                        name=next((normal_id(fields[k]) for k in NAME_COLUMNS if k in fields),'')
                        aliases=[ident]
                        for col in ['住院号','门诊号','患者编号','患者ID','PatientID']:
                            v=normal_id(fields.get(col))
                            if v and v not in aliases:aliases.append(v)
                        source={'path':str(path),'sheet':sheet.title,'row':row_no}
                        if ident not in roster:
                            roster[ident]={'name':name,'aliases':aliases,'fields':fields,'sources':[source], 'conflict':False}
                        else:
                            old=roster[ident];old['sources'].append(source)
                            if old['name']!=name or any(k in old['fields'] and old['fields'][k]!=v for k,v in fields.items()):
                                old['conflict']=True
                                issues.append({'patient_id':ident,'code':'ROSTER_ID_CONFLICT','source':str(path),
                                               'message':'同一编号对应不一致的患者表记录；不自动合并病例。'})
                            else:
                                old['fields'].update(fields);old['aliases']=sorted(set(old['aliases']+aliases))
                if not recognized:issues.append({'code':'ROSTER_SCHEMA_UNKNOWN','source':str(path),'message':'未找到患者编号列；未把此表当作患者主表。'})
            finally:book.close()
        except Exception as exc:
            if fatal_io(exc):raise
            issues.append({'code':'ROSTER_READ_FAILED','source':str(path),'message':str(exc)})
    return roster,issues


def alias_map(roster):
    result=defaultdict(set)
    for ident,entry in roster.items():
        for alias in entry.get('aliases',[ident]):result[alias].add(ident)
    return result


def filename_name(path):
    return re.split(r'[-—_]',unicodedata.normalize('NFKC',Path(path).stem),maxsplit=1)[0].strip()


def document_ids(text,known_aliases=None):
    s=normalize_text(text)
    ids=set(re.findall(r'/(?:EMR|BasicInfo|VisitHistory|DignosisHistory|SurgeryHistory|Medication)/([0-9]+)/',s,re.I))
    printed=set(re.findall(r'(?:门诊号|⻔诊号|患者编号|Patient\s*ID)\s*[:：]?\s*([A-Za-z0-9][A-Za-z0-9_-]*)',s,re.I))
    # Some exported page edges truncate a printed ID. Ignore a strict prefix
    # only when it is not itself a known patient's alias and a full URL agrees.
    printed={p for p in printed if p in (known_aliases or {}) or not any(v.startswith(p) and len(v)>len(p) for v in ids)}
    return ids|printed


def choose_document_patient(path,first_page,roster):
    name=filename_name(path)
    names={ident for ident,r in roster.items() if name and r.get('name')==name}
    aliases=alias_map(roster); declared=document_ids(first_page,aliases)
    matches=set().union(*(aliases.get(i,set()) for i in declared)) if declared else set()
    if declared:
        if any(i not in aliases for i in declared):return None,'unknown_document_id'
        if len(matches)!=1:return None,'document_id_conflict'
        ident=next(iter(matches))
        if roster[ident].get('conflict'):return None,'roster_id_conflict'
        if names and ident not in names:return None,'filename_document_id_conflict'
        return ident,'document_id'
    if len(names)==1:
        ident=next(iter(names))
        if roster[ident].get('conflict'):return None,'roster_id_conflict'
        return ident,'name_only_unverified'
    return None,'ambiguous_or_unmatched_name'


def process_documents(paths,roster,out,selected=None,runtime=None):
    import fitz
    results=[];issues=[];read_pages=0;opened=0
    selected=set(selected or [])
    selected_names={r.get('name') for i,r in roster.items() if not selected or i in selected}
    roster_signature=token(json.dumps(roster,ensure_ascii=False,sort_keys=True,default=str))
    for document_index,path in enumerate(paths,1):
        if runtime:runtime.control.progress('documents',document_index-1,len(paths),'关联和提取病例文档')
        if selected and filename_name(path) not in selected_names and not any(re.search(r'(?<!\d)'+re.escape(i)+r'(?!\d)',path.name) for i in selected):
            continue  # A sample run must not open other patients' PDFs.
        start_issues=len(issues);key='PDF:'+str(path.resolve());sig=None
        record={'source':str(path),'kind':'检验' if '检验' in path.name else '病例',
                'patient_id':'','match_basis':'','status':'','pages':0,'pages_read':0,'text_file':'','empty_pages':[]}
        try:
            if runtime:
                sig=runtime.fingerprint([path],roster_signature)
                cached=runtime.lookup(key,sig)
                if cached is not None:
                    record=cached['record'];record['cache_reused']=True
                    results.append(record);issues.extend(cached['issues']);runtime.counts['documents_reused']+=1
                    continue
                runtime.retire(key)
            opened+=1
            with fitz.open(path) as doc:
                record['pages']=len(doc)
                if doc.needs_pass:raise ValueError('PDF需要密码，未提取')
                first=doc[0].get_text(sort=True) if len(doc) else ''
                read_pages+=bool(len(doc));record['pages_read']=min(1,len(doc))
                ident,basis=choose_document_patient(path,first,roster)
                record.update(patient_id=ident or '',match_basis=basis)
                if not ident or (selected and ident not in selected):
                    record['status']='未关联_待核对'
                    issues.append({'code':'DOCUMENT_UNMATCHED','source':str(path),'message':basis})
                    if runtime and runtime.fingerprint([path],roster_signature)==sig:
                        runtime.save(key,sig,{'record':record,'issues':issues[start_issues:]})
                    results.append(record);continue
                verified=basis=='document_id'
                pages=[];conflicting_pages=[];aliases=alias_map(roster)
                for index in range(len(doc)):
                    if runtime:
                        runtime.control.progress('documents',document_index-1+(index/max(len(doc),1)),len(paths),f'病例文档：第 {index+1}/{len(doc)} 页')
                    text=normalize_text(first if index==0 else doc[index].get_text(sort=True))
                    if index:read_pages+=1;record['pages_read']+=1
                    if len(re.sub(r'\W','',text))<15:record['empty_pages'].append(index+1)
                    page_ids=document_ids(text,aliases)
                    if any(aliases.get(i,set())!={ident} for i in page_ids):conflicting_pages.append(index+1)
                    pages.append({'page':index+1,'text':text})
                if conflicting_pages:
                    verified=False;record['match_basis']='document_page_id_conflict'
                    record['conflicting_pages']=conflicting_pages
                    issues.append({'patient_id':ident,'code':'DOCUMENT_PAGE_ID_CONFLICT','source':str(path),
                                   'message':'PDF后续页面患者编号冲突，整份文档移入待确认目录。'})
                folder=out/('patients' if verified else 'pending_documents')/('patient_'+token(ident))/'clinical'
                folder.mkdir(parents=True,exist_ok=True)
                base=folder/('document_'+token(str(path.resolve())))
                if runtime:
                    for existing in [base.with_suffix('.txt'),base.with_suffix('.json')]:
                        if existing.exists():runtime.archive(existing)
                # Every page keeps its origin. No OCR guesses or diagnosis extraction.
                base.with_suffix('.txt').write_text('\n\n'.join(f'===== 第 {p["page"]} 页 =====\n{p["text"]}' for p in pages),encoding='utf-8')
                record['text_file']=str(base.with_suffix('.txt').relative_to(out))
                record['status']=('已提取' if verified else '关联待确认') if not record['empty_pages'] else '无有效文字页_需核对或OCR'
                record['verified_link']=verified
                write_json(base.with_suffix('.json'),{**record,'sha256':sha256(path),'pages_text':pages,
                    'scope':'patient-level association only; not encounter-level pairing',
                    'cleaning':['Unicode NFKC','normalize whitespace','retain page boundaries'],
                    'note':'Text layer only. Scanned or clipped content is not reconstructed; template statements are not diagnosis labels.'})
                if not verified or record['empty_pages']:
                    issues.append({'patient_id':ident,'code':'DOCUMENT_REVIEW','source':str(path), 'message':record['status']})
                if runtime:
                    if runtime.fingerprint([path],roster_signature)!=sig:raise OSError('提取期间PDF发生变化，请重试')
                    runtime.save(key,sig,{'record':record,'issues':issues[start_issues:]},[base.with_suffix('.txt'),base.with_suffix('.json')])
        except Exception as exc:
            if fatal_io(exc):raise
            record['status']='读取失败';record['error']=str(exc);record['verified_link']=False
            issues.append({'code':'DOCUMENT_READ_FAILED','source':str(path),'message':str(exc)})
        results.append(record)
    if runtime:runtime.control.progress('documents',len(paths),len(paths),'病例文档处理结束',True)
    return results,issues,{'pdf_files_opened':opened,'pdf_pages_read':read_pages}
