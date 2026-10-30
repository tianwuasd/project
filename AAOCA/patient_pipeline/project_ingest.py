"""Read new/changed files independently. Sources and extracted pages are retained."""
import os
from pathlib import Path
import re

from common import sha256, write_json
from patient_pipeline.clinical import read_roster, normalize_text, document_ids, filename_name
from patient_pipeline.imaging import read_header, folder_id
from patient_pipeline.project_model import digest, issue
from patient_pipeline.runtime import fatal_io

CLINICAL_PARSER_VERSION = 2
PDF_VALIDATION_VERSION = 1


class PDFInputError(ValueError):
    def __init__(self, code, message, data):
        super().__init__(message)
        import hashlib
        self.code=code
        self.details={'size_bytes':len(data),'input_sha256':hashlib.sha256(data).hexdigest()}


def validate_pdf_payload(path):
    """Identify proven absent content, never reject a valid PDF just for being small."""
    with Path(path).open('rb') as f:data=f.read(4097)
    if not data:raise PDFInputError('PDF_EMPTY_FILE','PDF文件为0字节，没有页面内容；需重新导出完整原件',data)
    lines=[line.strip() for line in data.splitlines() if line.strip()]
    if len(data)<=4096 and lines and re.match(rb'%PDF-\d+\.\d+',lines[0]) and all(
            line.startswith(b'%') for line in lines):
        raise PDFInputError('PDF_HEADER_ONLY',f'PDF仅有文件头和注释（{len(data)}字节），没有页面正文；需从原系统重新导出完整原件',data)


def clinical_metadata_current(record):
    if record.get('kind')=='DICOM':return True
    return record.get('clinical_parser_version')==CLINICAL_PARSER_VERSION and not (
        record.get('kind')=='PDF' and record.get('read_error') and
        record.get('pdf_validation_version')!=PDF_VALIDATION_VERSION)


def is_sidecar(path):
    p=Path(path)
    return p.name.startswith('._') or any(part.casefold()=='__macosx' for part in p.parts)


def source_key(path):return os.path.normcase(str(Path(path).resolve()))


def stamp(path):
    st=Path(path).stat();return [st.st_size,st.st_mtime_ns]


def discover(sources, output, control):
    found=set()
    for raw in sources:
        root=Path(raw).resolve()
        if not root.exists():raise FileNotFoundError(f'本次添加的资料目录不可访问：{root}')
        if root==output or root.is_relative_to(output) or output.is_relative_to(root):
            raise ValueError('输入与输出目录不能相互包含；请选择单独的结果目录。')
        if is_sidecar(root):continue
        if root.is_file():found.add(root);continue
        def scan_error(exc):raise exc
        for base,dirs,files in os.walk(root,onerror=scan_error,followlinks=False):
            control.checkpoint()
            dirs[:]=[n for n in dirs if n not in {'.venv','.git','__pycache__'} and not is_sidecar(Path(base,n))
                     and not Path(base,n).is_symlink() and not Path(base,n).is_junction()]
            for name in files:
                p=Path(base,name)
                if p.suffix.lower() in {'.dcm','.dicom','.pdf','.xlsx'} and not name.startswith('~$') and not is_sidecar(p):
                    found.add(p)
    return sorted(found,key=lambda p:str(p).lower())


def extract_pdf(store,path,key,control):
    import pymupdf
    validate_pdf_payload(path)
    pages=[];ids=set();low=[]
    with pymupdf.open(path) as doc:
        if doc.needs_pass:raise ValueError('PDF需要密码')
        for n,page in enumerate(doc,1):
            control.checkpoint()
            text=normalize_text(page.get_text(sort=True));declared=sorted(document_ids(text))
            pages.append({'page':n,'text':text,'ids':declared});ids.update(declared)
            if len(re.sub(r'\W','',text))<15:low.append(n)
    content_hash=sha256(path)
    # Separate extraction from patient assignment. A later roster can link this text offline.
    rel=Path('documents')/content_hash/'text.txt';dest=store.path(rel);dest.parent.mkdir(parents=True,exist_ok=True)
    if not dest.exists():
        dest.write_text('\n\n'.join(f'===== 第 {p["page"]} 页 =====\n{p["text"]}' for p in pages),encoding='utf-8')
    write_json(dest.with_name('pages.json'),{'source':str(path),'sha256':content_hash,'pages':pages})
    review=[]
    footer=re.findall(r'第\s*(\d+)\s*/\s*(\d+)\s*[页⻚]',pages[-1]['text']) if pages else []
    if footer:
        current,total=map(int,footer[-1])
        if current==len(pages) and current<total:
            review.append(issue('PDF_EXPORT_REVIEW',f'PDF实际{len(pages)}页，末页页脚为{current}/{total}；需回源核对导出范围，尚不能确认缺失正文',
                source=path,severity='warning'))
    return dict(document_ids=sorted(ids),document_name=filename_name(path),pages=len(pages),
                low_text_pages=low,text_file=rel.as_posix(),sha256=content_hash,issues=review)


def ingest(store,sources,control):
    control.progress('discover',0,0,'查找本批次资料')
    paths=discover(sources,store.root,control);changed=0;reused=0
    for n,path in enumerate(paths):
        control.progress('headers',n,len(paths),'登记资料并独立检查')
        key=source_key(path);old=store.source(key);before=stamp(path)
        kind={'.pdf':'PDF','.xlsx':'ROSTER'}.get(path.suffix.lower(),'DICOM')
        current_parser=clinical_metadata_current(old or {})
        if old and old.get('stamp')==before and current_parser:
            reused+=1;continue
        r={'key':key,'path':str(path),'kind':kind,'stamp':before,'pid':'','issues':[]}
        if kind!='DICOM':r['clinical_parser_version']=CLINICAL_PARSER_VERSION
        if kind=='PDF':r['pdf_validation_version']=PDF_VALIDATION_VERSION
        if store.get('meta','legacy_adopted'):r['new_since_legacy']=True
        try:
            if kind=='DICOM':r['header']=read_header(path,tolerate_timing=True)
            elif kind=='ROSTER':r['roster'],r['issues']=read_roster([path],control)
            else:r.update(extract_pdf(store,path,key,control))
            if stamp(path)!=before:raise OSError('读取过程中源文件发生变化；本次不接纳，请重试')
        except Exception as exc:
            if fatal_io(exc):raise
            r['read_error']=str(exc);r['pid']=folder_id(path) if kind=='DICOM' else ''
            r['read_error_code']=getattr(exc,'code',kind+'_READ_FAILED')
            r['issues']=[issue(r['read_error_code'],exc,r['pid'],path,**getattr(exc,'details',{}))]
        # Keep last readable metadata for explicit history, never silently substitute it for a damaged input.
        if old and r.get('read_error'):
            r['previous_header']=old.get('header') or old.get('previous_header')
            r['pid']=old.get('pid','');r['previous_unit_key']=old.get('unit_key') or old.get('previous_unit_key')
            r['previous_good']=old.get('previous_good') or {k:old.get(k) for k in
                ['pid','sha256','document_ids','document_name','pages','low_text_pages','text_file','match_basis','verified']}
        store.put_source(r,commit=False);changed+=1
        if n%100==0:store.commit()
    store.commit();control.progress('headers',len(paths),len(paths),'本批次资料登记完成',True)
    result={'files_seen':len(paths),'sources_changed':changed,'sources_reused':reused}
    store.put('meta','last_ingest',result);return result


def refresh_catalog(store,control):
    """Apply parser updates to existing catalogs; preserve excluded rows as evidence."""
    stale=[];excluded=0
    for r in list(store.sources()):
        control.checkpoint()
        if is_sidecar(r['path']):
            store.put('excluded_sources',r['key'],{'reason':'Mac附带文件，不是临床资料','source':r},False)
            store.db.execute('DELETE FROM sources WHERE key=?',(r['key'],));excluded+=1
        elif r['kind'] in {'PDF','ROSTER'} and not r.get('legacy') and Path(r['path']).is_file():
            if not clinical_metadata_current(r) or stamp(r['path'])!=r.get('stamp'):
                stale.append(r['path'])
    # Retire only units explicitly removed by this filter; ordinary offline inputs stay active.
    for key,u in list(store.items('units')):
        if u.get('sources') and all(store.get('excluded_sources',s) for s in u['sources']) and not store.unit_sources(key):
            u.update(retired=True,retired_reason='Mac附带文件，不是临床资料');store.put('units',key,u,False)
    store.commit()
    if stale:ingest(store,stale,control)
    result={'parser_version':CLINICAL_PARSER_VERSION,'metadata_refreshed':len(stale),
        'newly_excluded':excluded,'excluded_total':sum(1 for _ in store.items('excluded_sources'))}
    store.put('meta','catalog_refresh',result)
    return result
