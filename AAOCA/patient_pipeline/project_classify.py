"""Exclusive result categories and restartable, root-checked directory moves."""
import os
from pathlib import Path
from patient_pipeline.project_model import patient_folder

CATEGORIES={
 (True,True,True):'01_三类结果齐全', (True,True,False):'02_缺PDF',
 (False,True,True):'03_缺CT', (True,False,True):'04_缺超声',
 (False,True,False):'05_缺PDF和CT', (True,False,False):'06_缺PDF和超声',
 (False,False,True):'07_缺CT和超声', (False,False,False):'08_三类结果均缺'}


def subcategory(units):
    flags=set()
    for u in units:
        latest=u.get('last_attempt',u)
        if latest['status'] in {'failed','pending','unsupported'}:
            flags.add({'CT':'02_CT部分未成功','US':'07_超声部分未成功','PDF':'03_文档类型待补或关联待核实'}.get(u['kind'],'06_其他影像未转换'))
        for i in latest.get('issues',[]):
            if i.get('severity')=='info':continue
            if i['code'] in {'US_CALIBRATION_REVIEW','US_PHYSICAL_SCALE_UNKNOWN','US_SCALE_UNKNOWN'}:flags.add('04_超声标定待复核')
            elif latest['status']=='success':flags.add('08_其他待复核')
    if not flags:return '01_无其他异常'
    return next(iter(flags)) if len(flags)==1 else '05_多类问题并存'


def replace_prefix(value,old,new):
    if isinstance(value,str):
        v=value.replace('\\','/');o=old.replace('\\','/').rstrip('/')
        if v==o or v.startswith(o+'/'):return new+v[len(o):]
        return value
    if isinstance(value,list):return [replace_prefix(v,old,new) for v in value]
    if isinstance(value,dict):return {replace_prefix(k,old,new):replace_prefix(v,old,new) for k,v in value.items()}
    return value


def complete_move(store,job):
    old,new=job['old'],job['new'];src=store.path(old);dst=store.path(new)
    if src.exists() and dst.exists():raise ValueError(f'分类目标目录已存在，保留两处并停止移动：{dst}')
    if src.exists():
        if src.is_symlink() or src.is_junction():raise ValueError('不直接移动链接目录')
        dst.parent.mkdir(parents=True,exist_ok=True);src.rename(dst)
    elif not dst.exists():dst.mkdir(parents=True,exist_ok=True)
    for collection in ['units','folders']:
        for key,value in list(store.items(collection)):
            updated=replace_prefix(value,old,new)
            if updated!=value:store.put(collection,key,updated,commit=False)
    for r in list(store.sources(kind='PDF')):
        updated=replace_prefix(r,old,new)
        if updated!=r:store.put_source(updated,commit=False)
    # Keep legacy compatibility junctions aimed at the new physical patient directory.
    for rel in job.get('junctions',[]):
        link=store.root/rel
        if link.parent.resolve()!= (store.root/'patients').resolve():raise ValueError('兼容链接位置不合法')
        if link.is_junction():os.rmdir(link)  # Removes this junction only, never its target.
        if not link.exists():
            import _winapi
            _winapi.CreateJunction(str(dst),str(link))
    store.put('folders',job['pid'],new,commit=False)
    store.put('moves',job['pid'],{**job,'status':'complete'},commit=False);store.commit()


def classify(store,patients):
    if store.get('meta','external_legacy_root'):return
    for _,job in list(store.items('moves')):
        if job['status']=='pending':complete_move(store,job)
    for p in patients:
        pid=p['id'];parts=['patients',p['category']]
        if p['category']=='01_三类结果齐全':parts.append(p['subcategory'])
        parts.append(patient_folder(pid));new='/'.join(parts)
        old=store.get('folders',pid)
        if not old:
            store.path(new).mkdir(parents=True,exist_ok=True);store.put('folders',pid,new)
        elif old.replace('\\','/')!=new:
            oldpath=store.path(old);junctions=[]
            for item in (store.root/'patients').iterdir():
                if item.is_junction() and item.resolve()==oldpath:junctions.append(item.relative_to(store.root).as_posix())
            job={'pid':pid,'old':old,'new':new,'status':'pending','junctions':junctions}
            store.put('moves',pid,job);complete_move(store,job)
        p['folder']=store.get('folders',pid,new)
