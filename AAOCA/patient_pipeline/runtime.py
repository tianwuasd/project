"""Cooperative controls, durable unit checkpoints and safe output ownership."""
from contextlib import AbstractContextManager
from datetime import datetime
import errno
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import threading
import time
import uuid

from common import sha256

RECIPE='patient-pipeline-resume-v1'


class StopRequested(BaseException):pass


def fatal_io(exc):
    return isinstance(exc,sqlite3.Error) or (isinstance(exc,OSError) and (exc.errno in {errno.ENOSPC,getattr(errno,'EDQUOT',122)} or getattr(exc,'winerror',None) in {39,112}))


class Control:
    def __init__(self,callback=None):
        self.callback=callback;self.gate=threading.Event();self.gate.set();self.stopping=threading.Event()
        self._paused=False;self._last_emit=0
    def emit(self,**event):
        if self.callback:self.callback(event)
    def pause(self):self.gate.clear();self.emit(state='pausing',message='暂停请求已收到，等待当前读写结束…')
    def resume(self):self.gate.set();self.emit(state='running',message='继续处理')
    def stop(self):self.stopping.set();self.gate.set();self.emit(state='stopping',message='正在安全停止，已完成进度会保留…')
    def checkpoint(self):
        if self.stopping.is_set():raise StopRequested('用户安全停止')
        if not self.gate.is_set():
            self._paused=True;self.emit(state='paused',message='已暂停，可继续，也可安全停止')
            while not self.gate.wait(.1):
                if self.stopping.is_set():raise StopRequested('用户安全停止')
            self._paused=False
        if self.stopping.is_set():raise StopRequested('用户安全停止')
    def progress(self,stage,done,total,message='',force=False):
        self.checkpoint()
        t=time.monotonic()
        if force or done==total or t-self._last_emit>.08:
            self.emit(stage=stage,done=done,total=total,message=message,state='running');self._last_emit=t


class RunLock(AbstractContextManager):
    def __init__(self,out):self.path=Path(out)/'.pipeline.lock';self.file=None
    def __enter__(self):
        self.file=self.path.open('a+b')
        try:
            if self.file.tell()==0:self.file.write(b'0');self.file.flush()
            self.file.seek(0)
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(self.file.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(self.file.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except (OSError,IOError) as exc:
            self.file.close();self.file=None
            raise RuntimeError('另一个任务正在使用此输出目录，请等它结束或选择其他目录。') from exc
        return self
    def __exit__(self,*args):
        if self.file:
            self.file.seek(0)
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(self.file.fileno(),msvcrt.LK_UNLCK,1)
            else:
                import fcntl
                fcntl.flock(self.file.fileno(),fcntl.LOCK_UN)
            self.file.close();self.file=None


class Runtime:
    def __init__(self,out,control=None,force=False):
        self.out=Path(out).resolve();self.control=control or Control();self.force=force
        (self.out/'details').mkdir(exist_ok=True)
        self.db=sqlite3.connect(self.out/'details/resume.sqlite')
        self.db.execute('CREATE TABLE IF NOT EXISTS units(key TEXT PRIMARY KEY, signature TEXT, result TEXT, artifacts TEXT)')
        self.db.commit();self.seen=set();self.protected_prefixes=set();self.images_done=0;self.images_total=0
        self.counts={'headers_read':0,'headers_reused':0,'images_converted':0,'images_reused':0,'images_adopted':0,'documents_reused':0}
        self.history=self.out/'back'/('superseded_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
        self.legacy_quality={}
    def close(self):self.db.close()
    def fingerprint(self,paths,extra=None):
        rows=[]
        for path in sorted(map(Path,paths),key=str):
            self.control.checkpoint();st=path.stat()
            rows.append([str(path.resolve()),st.st_size,st.st_mtime_ns])
        return hashlib.sha256(json.dumps([RECIPE,rows,extra],ensure_ascii=False,sort_keys=True,default=str).encode()).hexdigest()
    def inside(self,path):
        p=Path(path).resolve()
        if p==self.out or not p.is_relative_to(self.out):raise ValueError('拒绝操作输出目录之外的文件')
        return p
    def archive(self,path):
        original=Path(path);p=self.inside(original)
        if original.is_symlink() or original.is_junction():raise ValueError('不自动移动链接目录')
        if not p.exists():return
        target=self.history/p.relative_to(self.out)
        if target.exists():target=target.with_name(target.name+'.'+uuid.uuid4().hex[:8])
        target.parent.mkdir(parents=True,exist_ok=True)
        p.rename(target)
    def lookup(self,key,signature):
        self.control.checkpoint();self.seen.add(key)
        if self.force:return None
        row=self.db.execute('SELECT signature,result,artifacts FROM units WHERE key=?',(key,)).fetchone()
        if not row or row[0]!=signature:return None
        try:
            for item in json.loads(row[2]):
                self.control.checkpoint();path=self.inside(self.out/item['path'])
                if not path.is_file() or path.stat().st_size!=item['size'] or sha256(path)!=item['sha256']:return None
            return json.loads(row[1])
        except (OSError,ValueError,KeyError):return None
    def save(self,key,signature,result,artifacts=()):
        proof=[]
        for p in artifacts:
            path=self.inside(p)
            proof.append({'path':str(path.relative_to(self.out)),'size':path.stat().st_size,'sha256':sha256(path)})
        self.db.execute('INSERT OR REPLACE INTO units VALUES(?,?,?,?)',
            (key,signature,json.dumps(result,ensure_ascii=False,allow_nan=False),json.dumps(proof)))
        self.db.commit();self.seen.add(key)
    def retire(self,key):
        row=self.db.execute('SELECT artifacts FROM units WHERE key=?',(key,)).fetchone()
        if row:
            for artifact in json.loads(row[0]):self.archive(self.out/artifact['path'])
    def cleanup_stale(self):
        # Only registered artifacts absent from this completed run are archived.
        for key, in self.db.execute('SELECT key FROM units').fetchall():
            if key not in self.seen and not any(key.startswith(p) for p in self.protected_prefixes):
                self.retire(key);self.db.execute('DELETE FROM units WHERE key=?',(key,))
        self.db.commit()
    def header(self,path,loader):
        key='header:'+str(path.resolve());sig=self.fingerprint([path]);cached=self.lookup(key,sig)
        if cached is not None:self.counts['headers_reused']+=1;return cached
        value=loader(path)
        if self.fingerprint([path])!=sig:raise OSError('文件读取期间发生变化，请重试')
        self.save(key,sig,value);self.counts['headers_read']+=1;return value
    def advance(self,weight,message=''):
        self.images_done+=weight
        self.control.progress('images',self.images_done,self.images_total,message,True)
        self.control.emit(counts=self.counts.copy())
    def convert(self,kind,paths,dest,runner):
        key=kind+':'+str(Path(dest).relative_to(self.out));sig=self.fingerprint(paths)
        cached=self.lookup(key,sig)
        if cached is not None:
            self.counts['images_reused']+=1
            return cached['meta'],cached.get('metrics',[]),'校验通过，已跳过'
        dest=Path(dest)
        # Migrate an older completed output only after checking both original
        # source hashes and the actual saved image hash. Never trust existence.
        has_entry=self.db.execute('SELECT 1 FROM units WHERE key=?',(key,)).fetchone()
        if dest.exists() and not has_entry and not self.force:
            try:
                meta=json.loads((dest/'metadata.json').read_text(encoding='utf-8'))
                sources=meta.get('sources_in_slice_order') or [meta['source']]
                assert {str(Path(s['path']).resolve()) for s in sources}=={str(Path(p).resolve()) for p in paths}
                for s in sources:
                    self.control.checkpoint();assert sha256(s['path'])==s['sha256']
                data=dest/('image.nii.gz' if kind=='CT' else ('image.png' if meta['storage']=='PNG' else 'frames.npz'))
                assert sha256(data)==meta['output_sha256'] and meta.get('roundtrip_verified')
                assert (dest/'preview.png').is_file()
                if self.fingerprint(paths)!=sig:raise ValueError('源文件发生变化')
                patient_folder=dest.parent.parent
                if str(patient_folder) not in self.legacy_quality:
                    try:self.legacy_quality[str(patient_folder)]=json.loads((patient_folder/'图像抽查.json').read_text(encoding='utf-8'))
                    except (ValueError,OSError):self.legacy_quality[str(patient_folder)]=[]
                metrics=[]
                for prior in self.legacy_quality[str(patient_folder)]:
                    if kind=='CT' and prior.get('series')==dest.name.removeprefix('series_'):metrics=prior.get('sampled_slices',[])
                    if kind=='US' and prior.get('source')==str(paths[0]):metrics=prior.get('sampled_frames',[])
                self.save(key,sig,{'meta':meta,'metrics':metrics},[p for p in dest.iterdir() if p.is_file()])
                self.counts['images_adopted']+=1
                return meta,metrics,'旧版结果校验通过，已接纳'
            except (OSError,ValueError,KeyError,AssertionError):pass
        if dest.exists():self.archive(dest)
        metrics=[]
        from patient_pipeline.imaging import quality_metrics
        def checkpoint(done=0,total=1):
            self.control.progress('images',self.images_done+len(paths)*done/max(total,1),self.images_total,
                                  f'{kind}：当前 {done}/{total}',False)
        meta=runner(lambda a,i,n:metrics.append(quality_metrics(a,i,n)),checkpoint)
        if self.fingerprint(paths)!=sig:
            self.archive(dest);raise OSError('转换期间源文件发生变化，本次结果已隔离，需重试')
        self.save(key,sig,{'meta':meta,'metrics':metrics},[p for p in dest.iterdir() if p.is_file()])
        self.counts['images_converted']+=1
        return meta,metrics,'本次转换'
