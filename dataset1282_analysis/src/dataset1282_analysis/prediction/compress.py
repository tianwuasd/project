import argparse
from copy import deepcopy
from itertools import product
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import psutil

from ..contracts import atomic_json, digest, read_json
from .preprocessed import FORMAT, B2ND_FORMAT


def file_identity(path):
    stat=path.stat()
    return [stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns]


def unlink_npy(path):
    if os.name!='nt':
        path.unlink()
        return
    import ctypes
    from ctypes import wintypes
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    create=kernel.CreateFileW
    create.argtypes=[wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,wintypes.LPVOID,
                     wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE]
    create.restype=wintypes.HANDLE
    close=kernel.CloseHandle
    close.argtypes=[wintypes.HANDLE]
    close.restype=wintypes.BOOL
    # 只检查待删除的文件，允许本句柄删除文件，拒绝与其他读写句柄共享。
    handle=create(str(path),0x80000000,4,None,3,0x80,None)
    if handle==ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        path.unlink()
    finally:
        close(handle)


def delete_verified_npys(source_root,target,ready):
    source_root=source_root.resolve(strict=True)
    if os.name=='posix' and not source_root.is_relative_to(Path('/data4/zhangruichen/data/chd-segmentation/preprocessed')):
        raise ValueError('删除NPY仅允许指定的data4预处理目录')
    # 删除前重新检查全部压缩文件；只移除转换清单中已核验的image.npy和label.npy。
    pending=[]
    for paths in ready['verified'].values():
        for relative,identity in paths['compressed'].items():
            path=target/relative
            if path.is_symlink() or not path.resolve(strict=True).is_relative_to(target) or file_identity(path)!=identity:
                raise ValueError('核验后的压缩文件发生变化，拒绝删除NPY：'+str(path))
        for relative,identity in paths['npy'].items():
            path=source_root/relative
            if path.name not in ('image.npy','label.npy') or path.is_symlink():
                raise ValueError('删除对象超出已确认的NPY范围')
            if path.exists():
                if not path.resolve(strict=True).is_relative_to(source_root) or file_identity(path)!=identity:
                    raise ValueError('源NPY在核验后发生变化：'+str(path))
                pending.append((path,identity))
    me=psutil.Process()
    excluded={me.pid,*(p.pid for p in me.parents())}
    for process in psutil.process_iter(['pid','username','cmdline'],ad_value=None):
        if process.pid in excluded or process.info['username']!=me.username():
            continue
        if str(source_root) in ' '.join(process.info['cmdline'] or []):
            raise RuntimeError('仍有任务引用原NPY目录：'+str(process.pid))
        if os.name=='nt':
            continue
        if any(Path(f.path).is_relative_to(source_root) for f in process.open_files()):
            raise RuntimeError('仍有任务打开原NPY文件：'+str(process.pid))
        if any(Path(m.path).is_relative_to(source_root) for m in process.memory_maps(grouped=False)
               if m.path and not m.path.startswith('[')):
            raise RuntimeError('仍有任务映射原NPY文件：'+str(process.pid))
    removed=logical=0
    for path,identity in pending:
        if file_identity(path)!=identity:
            raise ValueError('删除前源文件身份发生变化')
        unlink_npy(path)
        removed+=1
        logical+=identity[2]
    ready.update(source_deleted=True,deleted_npy_files=removed,deleted_logical_bytes=logical)
    atomic_json(target/'compression-complete.json',ready)
    print(json.dumps({'stage':'npy_deleted','files':removed,'logical_bytes':logical,
                      'note':'硬链接的实际释放空间取决于其他链接是否保留'}),flush=True)


def verify_array(original, compressed, boundary, chunk_bytes):
    if original.shape!=compressed.shape or original.dtype!=compressed.dtype:
        raise ValueError('压缩前后的数组尺寸或类型不同')
    shape=original.shape
    chunks=list(shape)
    while np.prod(chunks)*original.dtype.itemsize>chunk_bytes:
        axis=max(range(len(chunks)),key=lambda i:chunks[i])
        chunks[axis]=max(1,(chunks[axis]+1)//2)
    for starts in product(*(range(0,n,c) for n,c in zip(shape,chunks))):
        boundary()
        section=tuple(slice(s,min(s+c,n)) for s,c,n in zip(starts,chunks,shape))
        before=np.ascontiguousarray(original[section])
        after=np.ascontiguousarray(compressed[section])
        if not np.array_equal(before.view(np.uint8),after.view(np.uint8)):
            raise ValueError('压缩前后逐字节不一致，拒绝提交转换结果')


def run(args):
    code=Path(args.cht_code).resolve(strict=True)
    sys.path.insert(0,str(code/'src'))
    from chd_segmentation.artifacts.store import operation_lock
    from chd_segmentation.preprocessing.inputs import OfflineDataset, resolve_preprocessed
    from chd_segmentation.preprocessing.offline_store import inspect_case, save_case
    source_path=Path(args.preprocessed).resolve(strict=True)
    prepared=read_json(source_path)
    if prepared['storage']['format']!=FORMAT or not prepared['storage']['complete']:
        raise ValueError('输入必须是已经完成的NPY预处理数据集')
    source_root=Path(prepared['storage']['root']).resolve(strict=True)
    if Path(args.output).is_symlink():
        raise ValueError('压缩输出不能使用软链接')
    target=Path(args.output).resolve()
    if target==source_root or target.is_relative_to(source_root) or source_root.is_relative_to(target):
        raise ValueError('压缩输出必须使用独立目录，禁止覆盖输入')
    if args.chunk_mib<1 or args.chunk_mib>32 or not 1<=args.level<=9:
        raise ValueError('压缩块需为1..32MiB，压缩级别需为1..9')
    if args.max_ram_gib<=0:
        raise ValueError('内存上限必须大于0')
    if target.exists() and (target.is_symlink() or not args.resume):
        raise ValueError('目标已存在；相同转换使用--resume继续')
    names=list(prepared['storage']['cases'])
    source=resolve_preprocessed(source_path,dataset=prepared['dataset'],case_ids=names)
    settings={'compression_chunk_bytes':args.chunk_mib*1024**2,'compression_level':args.level}
    recipe=deepcopy(source.catalog['recipe'])
    recipe.update(format=B2ND_FORMAT,compression=settings,
                  container_conversion={'source_recipe_identity':prepared['storage']['recipe_identity'],
                                        'method':'lossless-npy-to-b2nd-v1'})
    rid=digest(recipe)
    expected={'protocol':'chd1282-compression-v1','source':str(source_path),
              'source_identity':digest(prepared),'recipe_identity':rid}
    target.mkdir(parents=True,exist_ok=True)
    process=psutil.Process()
    peak=0
    def boundary():
        nonlocal peak
        rss=process.memory_info().rss
        peak=max(peak,rss)
        memory=psutil.virtual_memory()
        if rss>min(args.max_ram_gib*1024**3,memory.total*.8) or memory.available<memory.total*.2:
            raise MemoryError('达到转换内存上限；保留已提交病例，使用--resume接续')
    began=time.perf_counter()
    pointers={}
    verified={}
    with operation_lock(target):
        ledger=target/'conversion.json'
        if ledger.exists():
            if read_json(ledger)!=expected:
                raise ValueError('目标转换记录的输入或配方不同')
        elif any(p.name!='.run.lock' for p in target.iterdir()):
            raise ValueError('目标包含不属于本次转换的文件')
        else:
            atomic_json(ledger,expected)
        complete=target/'compression-complete.json'
        if complete.exists():
            ready=read_json(complete)
            if ready['conversion']!=expected or ready['prepared_identity']!=digest(read_json(target/'prepared.json')):
                raise ValueError('已完成转换的来源或输出描述不同')
            if args.delete_npy_after_verify and not ready['source_deleted']:
                delete_verified_npys(source_root,target,ready)
            print(json.dumps({'stage':'compression_complete','cases':len(names),'reused':True,
                              'source_deleted':ready['source_deleted']}),flush=True)
            return
        for index,name in enumerate(names,1):
            boundary()
            original=source.open(name,with_label=True)
            original_record=read_json(source_root/prepared['storage']['cases'][name]['key'][:16]/'case.json')
            source_stamps={kind:file_identity(source_root/prepared['storage']['cases'][name]['key'][:16]/(kind+'.npy'))
                           for kind in ('image','label')}
            key=digest({'recipe':rid,'source_case':prepared['storage']['cases'][name]['identity']})
            directory=target/key[:16]
            record=inspect_case(directory,key,rid)
            reused=record is not None
            if record is None:
                required=sum(original[k].nbytes for k in ('image','label') if original[k] is not None)
                if shutil.disk_usage(target).free<required+16*1024**2:
                    raise OSError('剩余空间不足以写入下一病例；保留已提交病例')
                entry=deepcopy(original_record['entry'])
                entry.update(image=original['image'],label=original['label'])
                entry['properties']['class_locations']=original['properties']['class_locations']
                record=save_case(directory,key,rid,entry,format=B2ND_FORMAT,
                                 compression=settings,boundary=boundary)
                del entry
            pointer={'case_id':name,'key':key,'identity':record['identity'],
                     'shape_czyx':record['files']['image.b2nd']['shape'],
                     'array_bytes':sum(v.get('array_bytes',0) for v in record['files'].values()),
                     'stored_bytes':sum(v['stamp'][0] for v in record['files'].values()),
                     'decode_workspace_bytes':record['decode_workspace_bytes'],
                     'locations_bytes':record['files']['locations.npz']['stamp'][0]}
            catalog={'format':B2ND_FORMAT,'recipe':recipe,'recipe_identity':rid,
                     'cases':{key:pointer},'case_index':{name:key}}
            # 逐例核验通过后登记；中断时已经提交的病例目录可重新打开并继续核验。
            converted=OfflineDataset(target,catalog,{name:pointer},prepared['dataset']).open(name,with_label=True)
            for kind in ('image','label'):
                verify_array(original[kind],converted[kind],boundary,settings['compression_chunk_bytes'])
            if digest(record['entry'])!=digest(original_record['entry']) or record['location_keys']!=original_record['location_keys']:
                raise ValueError('压缩前后的病例几何或监督元数据不同')
            for location,values in original['properties']['class_locations'].items():
                if not np.array_equal(values,converted['properties']['class_locations'][location]):
                    raise ValueError('压缩前后的前景采样坐标不同')
            if any(file_identity(source_root/prepared['storage']['cases'][name]['key'][:16]/(kind+'.npy'))!=stamp
                   for kind,stamp in source_stamps.items()):
                raise ValueError('源NPY在转换或核验期间发生变化')
            pointers[name]=pointer
            verified[name]={'npy':{str((source_root/prepared['storage']['cases'][name]['key'][:16]/(kind+'.npy')).relative_to(source_root)):
                                    source_stamps[kind]
                                    for kind in ('image','label')},
                            'compressed':{str((directory/file).relative_to(target)):file_identity(directory/file)
                                          for file in ('image.b2nd','label.b2nd','case.json','locations.npz')}}
            print(json.dumps({'stage':'compression','completed':index,'total':len(names),'case':name,
                              'reused':reused,'seconds':time.perf_counter()-began,'peak_rss_bytes':peak}),flush=True)
            del original,converted
        catalog={'format':B2ND_FORMAT,'recipe':recipe,'recipe_identity':rid,
                 'cases':{p['key']:p for p in pointers.values()},'case_index':{n:p['key'] for n,p in pointers.items()}}
        result=deepcopy(prepared)
        result['storage'].update(format=B2ND_FORMAT,root=str(target),base_root=str(target.parent),
                                 recipe_identity=rid,cases=pointers,complete=True)
        result['identity']=digest({k:v for k,v in result.items() if k!='identity'})
        atomic_json(target/'catalog.json',catalog)
        atomic_json(target/'prepared.json',result)
        ready={'conversion':expected,'prepared_identity':digest(result),'verified':verified,'source_deleted':False}
        atomic_json(target/'compression-complete.json',ready)
        if args.delete_npy_after_verify:
            delete_verified_npys(source_root,target,ready)
        print(json.dumps({'stage':'compression_complete','cases':len(names),'prepared':str(target/'prepared.json'),
                          'seconds':time.perf_counter()-began,'peak_rss_bytes':peak,'source_deleted':ready['source_deleted']}),flush=True)


def main(argv=None):
    parser=argparse.ArgumentParser(description='无损转换CHT的NPY预处理数据并逐块核验，可按病例接续')
    parser.add_argument('--preprocessed',type=Path,required=True)
    parser.add_argument('--cht-code',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--delete-npy-after-verify',action='store_true',help='全部病例核验通过且无活动引用后，仅删除源image.npy和label.npy')
    parser.add_argument('--chunk-mib',type=int,default=4)
    parser.add_argument('--level',type=int,default=3)
    parser.add_argument('--max-ram-gib',type=float,default=100)
    run(parser.parse_args(argv))


if __name__=='__main__':
    main()
