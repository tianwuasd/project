"""Read new/changed files independently. Sources and extracted pages are retained."""
import os
from pathlib import Path
import re

from common import sha256, write_json
from patient_pipeline.clinical import read_roster, normalize_text, document_ids, filename_name
from patient_pipeline.imaging import read_header, folder_id
from patient_pipeline.project_model import digest, issue
from patient_pipeline.runtime import fatal_io


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
        if root.is_file():found.add(root);continue
        def scan_error(exc):raise exc
        for base,dirs,files in os.walk(root,onerror=scan_error,followlinks=False):
            control.checkpoint()
            dirs[:]=[n for n in dirs if n not in {'.venv','.git','__pycache__'}
                     and not Path(base,n).is_symlink() and not Path(base,n).is_junction()]
            for name in files:
                p=Path(base,name)
                if p.suffix.lower() in {'.dcm','.dicom','.pdf','.xlsx'} and not name.startswith('~$'):
                    found.add(p)
    return sorted(found,key=lambda p:str(p).lower())


def extract_pdf(store,path,key,control):
    import pymupdf
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
    return dict(document_ids=sorted(ids),document_name=filename_name(path),pages=len(pages),
                low_text_pages=low,text_file=rel.as_posix(),sha256=content_hash)


def ingest(store,sources,control):
    control.progress('discover',0,0,'查找本批次资料')
    paths=discover(sources,store.root,control);changed=0;reused=0
    for n,path in enumerate(paths):
        control.progress('headers',n,len(paths),'登记资料并独立检查')
        key=source_key(path);old=store.source(key);before=stamp(path)
        if old and old.get('stamp')==before:
            reused+=1;continue
        kind={'.pdf':'PDF','.xlsx':'ROSTER'}.get(path.suffix.lower(),'DICOM')
        r={'key':key,'path':str(path),'kind':kind,'stamp':before,'pid':'','issues':[]}
        if store.get('meta','legacy_adopted'):r['new_since_legacy']=True
        try:
            if kind=='DICOM':r['header']=read_header(path,tolerate_timing=True)
            elif kind=='ROSTER':r['roster'],r['issues']=read_roster([path],control)
            else:r.update(extract_pdf(store,path,key,control))
            if stamp(path)!=before:raise OSError('读取过程中源文件发生变化；本次不接纳，请重试')
        except Exception as exc:
            if fatal_io(exc):raise
            r['read_error']=str(exc);r['pid']=folder_id(path) if kind=='DICOM' else ''
            r['issues']=[issue(kind+'_READ_FAILED',exc,r['pid'],path)]
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
