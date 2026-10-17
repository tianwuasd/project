"""One-off, restartable copying of registered successful CT outputs; no conversion."""
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
import argparse, csv, gzip, hashlib, io, json, re, shutil, zipfile
import nibabel as nib
import numpy as np

from workspace import workspace_root, work_directory, reference_directory, relocate
DEST = workspace_root(__file__)
WORK = work_directory(DEST)
REFERENCE = reference_directory(DEST)
SOURCES = {'AAOCA': DEST/'results/AAOCA_results', 'TAPVC': DEST/'results/TAPVC_result'}

def stamp():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()

def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)

def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def write_csv(path, rows, fields):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for row in rows:
            w.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict,list)) else v for k,v in row.items() if k in fields})

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def header(path):
    im = nib.load(str(path))
    spacing = list(map(float, im.header.get_zooms()))
    affine = np.asarray(im.affine)
    issues = []
    if len(im.shape) != 3: issues.append('不是三维单通道体积')
    if not np.isfinite(affine).all() or abs(np.linalg.det(affine[:3,:3])) < 1e-10: issues.append('空间变换不可用')
    if not all(np.isfinite(x) and x > 0 for x in spacing): issues.append('体素间距不可用')
    return {'shape':list(map(int,im.shape)), 'spacing_mm':spacing,
            'orientation': ''.join(nib.aff2axcodes(affine)), 'dtype':str(im.get_data_dtype()),
            'units':list(im.header.get_xyzt_units()), 'affine_ras':affine.tolist(),
            'sform_code':int(im.header['sform_code']), 'qform_code':int(im.header['qform_code']),
            'slope':float(im.dataobj.slope), 'intercept':float(im.dataobj.inter), 'header_issues':issues}

def audit():
    cases, patients, excluded, snapshots = [], [], [], {}
    for kind, root in SOURCES.items():
        roster = {r['id']:r for r in read_csv(root/'患者分类表.csv')}
        units = [r for r in read_csv(root/'处理单元表.csv') if r['kind']=='CT']
        all_units = defaultdict(list)
        for row in units: all_units[row['pid']].append(row)
        mapping = {}
        for pid, group in sorted(all_units.items()):
            success = [r for r in group if r['status']=='success']
            if not success: continue
            match = re.match(r'^\d+',pid)
            if not match: raise ValueError(f'编号没有数字前缀，须人工核对: {kind} {pid}')
            number = match.group()
            if number in mapping and mapping[number] != pid: raise ValueError(f'数字编号碰撞: {kind} {number}')
            mapping[number] = pid
            info = {'type':kind, 'folder_number':number, 'registered_patient_id':pid,
                    'source_patient_row':roster.get(pid,{}), 'ct_units':group,
                    'successful_original_series':len(success), 'volumes':sum(len(json.loads(r['outputs'])) for r in success),
                    'note':'数字前缀仅用于目录命名；登记编号、姓名及别名均保留。不因此执行患者合并。'}
            patients.append(info)
            for row in success:
                series = hashlib.sha256(row['key'].encode()).hexdigest()[:16]
                outputs = json.loads(row['outputs'])
                for index, out in enumerate(outputs,1):
                    src = (root/out).resolve(strict=True)
                    assert src.is_relative_to(root.resolve())
                    meta_path = src.parent/'metadata.json'
                    meta = json.loads(meta_path.read_text(encoding='utf-8-sig')) if meta_path.exists() else {}
                    case_id = f'{kind}_{number}_{series}_v{index:02d}'
                    rel = Path(kind)/number/f'序列_{series}'/f'体积_{index:02d}'/f'{case_id}_0000.nii.gz'
                    case = {'type':kind, 'number':number, 'registered_patient_id':pid,
                            'name':roster.get(pid,{}).get('name',''), 'case_id':case_id,
                            'unit_key':row['key'], 'series_folder':f'序列_{series}', 'volume_index':index,
                            'series_volume_count':len(outputs), 'source_file':str(src), 'destination':str(rel),
                            'source_bytes':src.stat().st_size, 'source_mtime_ns':src.stat().st_mtime_ns,
                            'source_output_sha256':meta.get('output_sha256'),
                            'source_phase_folder':src.parent.name,
                            'cardiac_phase_percent':meta.get('cardiac_phase_percent'),
                            'intensity_units_from_conversion':meta.get('intensity_units'),
                            'source_metadata':str(meta_path) if meta_path.exists() else None,
                            **header(src)}
                    cases.append(case)
            print(f'HEADERS {kind} {number} {len(cases)}',flush=True)
        ct_pids = {p['registered_patient_id'] for p in patients if p['type']==kind}
        excluded += [{'type':kind,'registered_patient_id':pid,'name':r['name'],'reason':'已有 CT 记录但没有成功体积' if all_units.get(pid) else '没有 CT 处理记录','ct_units':all_units.get(pid,[])} for pid,r in roster.items() if pid not in ct_pids]
        snapshots[kind] = {p.name:sha(p) for p in [root/'项目数据快照.json',root/'处理单元表.csv',root/'患者分类表.csv']}
    assert len({r['case_id'] for r in cases}) == len(cases)
    ref_files = sorted((REFERENCE/'formated_dataset1200_image_all').glob('*.nii.gz'))
    refs=[]
    for i,p in enumerate(ref_files,1):
        refs.append({'file':str(p),'bytes':p.stat().st_size,**header(p)})
        if i%100==0: print(f'CHD HEADERS {i}/{len(ref_files)}',flush=True)
    with zipfile.ZipFile(REFERENCE/'formated_dataset1200_label_less_then_11.zip') as z:
        names=[n for n in z.namelist() if n.endswith('.nii.gz')]
        label_by_name={Path(n).name:n for n in names}
        paired=sum(p.name.replace('_image.nii.gz','_label.nii.gz') in label_by_name for p in ref_files)
    def summary(rows):
        spacing=np.asarray([r['spacing_mm'][:3] for r in rows]); shapes=np.asarray([r['shape'][:3] for r in rows])
        return {'volumes':len(rows),'dtype':dict(Counter(r['dtype'] for r in rows)),
                'dimensions':dict(Counter(str(len(r['shape'])) for r in rows)),
                'orientation':dict(Counter(r['orientation'] for r in rows)),
                'units':dict(Counter(str(r['units']) for r in rows)),
                'spacing_min':spacing.min(axis=0).tolist(),'spacing_median':np.median(spacing,axis=0).tolist(),'spacing_max':spacing.max(axis=0).tolist(),
                'shape_min':shapes.min(axis=0).tolist(),'shape_median':np.median(shapes,axis=0).tolist(),'shape_max':shapes.max(axis=0).tolist(),
                'header_issue_count':sum(bool(r['header_issues']) for r in rows)}
    plan={'created':stamp(),'cases':cases,'patients':patients,'no_successful_ct':excluded,'source_table_hashes':snapshots}
    write_json(WORK/'copy_plan.json',plan)
    write_json(WORK/'chd_headers.json',refs)
    sums={k:summary([r for r in cases if r['type']==k]) for k in SOURCES}
    sums['CHD']=summary(refs);sums['CHD'].update(labels=len(names),paired_images=paired)
    for k in SOURCES:
        sums[k].update(patients=sum(p['type']==k for p in patients),original_series=sum(p['successful_original_series'] for p in patients if p['type']==k),bytes=sum(r['source_bytes'] for r in cases if r['type']==k))
    write_json(DEST/'格式核对汇总.json',sums)
    write_json(WORK/'header_audit_summary.json',sums)
    print(json.dumps(sums,ensure_ascii=False,indent=2),flush=True)

def copy_verified(src,dst,expected=None):
    dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.exists():
        source_hash=sha(src)
        if sha(dst)!=source_hash: raise ValueError(f'目标已存在且不同，停止以免覆盖: {dst}')
        return source_hash,'already_verified'
    partial=dst.with_name(dst.name+'.partial')
    h=hashlib.sha256()
    with src.open('rb') as f,partial.open('wb') as g:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):
            h.update(chunk);g.write(chunk)
    digest=h.hexdigest()
    if isinstance(expected,str) and re.fullmatch('[0-9a-fA-F]{64}',expected) and digest.lower()!=expected.lower():
        raise ValueError(f'原成果校验值不符: {src}')
    if sha(partial)!=digest: raise ValueError(f'复制后校验不符: {dst}')
    shutil.copystat(src,partial)
    partial.replace(dst)
    return digest,'copied_verified'

def copy_all():
    plan=relocate(json.loads((WORK/'copy_plan.json').read_text(encoding='utf-8')), DEST)
    cases=plan['cases'];completed=[];errors=[]
    for kind,root in SOURCES.items():
        for name,digest in plan['source_table_hashes'][kind].items():
            assert sha(root/name)==digest, '源登记表已改变，请重新核对'
    required=sum(c['source_bytes'] for c in cases if not (DEST/c['destination']).exists())
    assert shutil.disk_usage(DEST).free > required + 10*2**30,'磁盘空间不足'
    for i,c in enumerate(cases,1):
        try:
            src=Path(c['source_file']);dst=DEST/c['destination']
            assert src.stat().st_size==c['source_bytes'] and src.stat().st_mtime_ns==c['source_mtime_ns'],'源文件发生变化'
            digest,action=copy_verified(src,dst,c['source_output_sha256'])
            record={**c,'sha256':digest,'copy_status':action,'completed_at':stamp()}
            if c['source_metadata']:
                copy_verified(Path(c['source_metadata']),dst.parent/'原转换信息.json')
            preview=src.parent/'preview.png'
            if preview.exists(): copy_verified(preview,dst.parent/'影像预览.png')
            write_json(dst.parent/'体积信息.json',record)
            completed.append(record)
            with (WORK/'copy_log.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
        except Exception as e:
            errors.append({'case_id':c['case_id'],'source_file':c['source_file'],'error':str(e)})
        write_json(DEST/'整理进度.json',{'updated':stamp(),'stage':'复制并逐文件校验','processed':i,'total':len(cases),'success':len(completed),'failed':len(errors),'completed_bytes':sum(r['source_bytes'] for r in completed),'errors':errors})
        print(f'COPY {i}/{len(cases)} ok={len(completed)} failed={len(errors)} {c["case_id"]}',flush=True)
    for p in plan['patients']:
        rows=[r for r in completed if r['type']==p['type'] and r['registered_patient_id']==p['registered_patient_id']]
        p={**p,'copied_volumes':len(rows),'case_ids':[r['case_id'] for r in rows]}
        folder=DEST/p['type']/p['folder_number'];folder.mkdir(parents=True,exist_ok=True)
        write_json(folder/'编号信息.json',p)
        write_csv(folder/'CT清单.csv',rows,['case_id','registered_patient_id','name','unit_key','source_phase_folder','cardiac_phase_percent','shape','spacing_mm','dtype','orientation','destination','source_file','sha256'])
    fields=['type','number','registered_patient_id','name','case_id','unit_key','volume_index','series_volume_count','cardiac_phase_percent','source_phase_folder','shape','spacing_mm','dtype','orientation','intensity_units_from_conversion','source_bytes','destination','source_file','sha256','header_issues']
    write_csv(DEST/'CT总清单.csv',completed,fields)
    write_json(DEST/'CT总清单.json',completed)
    write_csv(DEST/'未纳入的编号.csv',plan['no_successful_ct'],['type','registered_patient_id','name','reason','ct_units'])
    write_csv(DEST/'编号总清单.csv',[{'type':p['type'],'number':p['folder_number'],'registered_patient_id':p['registered_patient_id'],'name':p['source_patient_row'].get('name',''),'original_series':p['successful_original_series'],'volumes':p['volumes'],'source_category':p['source_patient_row'].get('category',''),'source_subcategory':p['source_patient_row'].get('subcategory','')} for p in plan['patients']],['type','number','registered_patient_id','name','original_series','volumes','source_category','source_subcategory'])
    write_json(DEST/'整理进度.json',{'updated':stamp(),'stage':'复制完成' if not errors else '复制完成，部分待处理','total':len(cases),'success':len(completed),'failed':len(errors),'completed_bytes':sum(r['source_bytes'] for r in completed),'errors':errors,'nnunet_preprocessing':'尚未执行；待确定与目标模型一致的参数'})
    print('FINISHED',len(completed),'errors',errors,flush=True)

def samples():
    plan=relocate(json.loads((WORK/'copy_plan.json').read_text(encoding='utf-8')), DEST)
    refs=json.loads((WORK/'chd_headers.json').read_text(encoding='utf-8'))
    selected=[]
    for k in SOURCES:
        rows=[r for r in plan['cases'] if r['type']==k]
        indices=sorted({0,len(rows)//2,len(rows)-1,min(range(len(rows)),key=lambda i:rows[i]['spacing_mm'][2]),max(range(len(rows)),key=lambda i:rows[i]['spacing_mm'][2])})
        selected += [{'type':k,'file':rows[i]['source_file']} for i in indices]
    selected += [{'type':'CHD','file':refs[i]['file']} for i in [0,len(refs)//2,len(refs)-1]]
    selected += [{'type':'CHD','file':r['file']} for r in refs if r['dtype']=='int16']
    checkpoint=WORK/'voxel_samples.json'
    results=relocate(json.loads(checkpoint.read_text(encoding='utf-8')), DEST) if checkpoint.exists() else []
    done={r['file'] for r in results}
    with zipfile.ZipFile(REFERENCE/'formated_dataset1200_label_less_then_11.zip') as z:
        lookup={Path(n).name:n for n in z.namelist() if n.endswith('.nii.gz')}
        for row in selected:
            if row['file'] in done: continue
            img=nib.load(row['file']);arr=np.asarray(img.dataobj,dtype=np.float32)
            finite=np.isfinite(arr); vals=arr[finite]
            stat={'type':row['type'],'file':row['file'],'shape':list(arr.shape),'finite_fraction':float(finite.mean()),'min':float(vals.min()),'max':float(vals.max()),'mean':float(vals.mean()),'std':float(vals.std()),'percentiles_005_05_50_95_995':np.percentile(vals,[.5,5,50,95,99.5]).tolist(),'zero_fraction':float((arr==0).mean())}
            if row['type']=='CHD':
                member=lookup[Path(row['file']).name.replace('_image.nii.gz','_label.nii.gz')]
                if z.getinfo(member).flag_bits & 1:
                    stat['label']={'member':member,'status':'encrypted_password_required','content_checked':False}
                else:
                    label=nib.Nifti1Image.from_bytes(gzip.decompress(z.read(member)))
                    seg=np.asanyarray(label.dataobj)
                    stat['label']={'member':member,'shape':list(seg.shape),'values':np.unique(seg).tolist(),'same_shape':img.shape==label.shape,'same_affine':bool(np.allclose(img.affine,label.affine,atol=1e-4))}
            results.append(stat)
            write_json(WORK/'voxel_samples.json',results)
            print(json.dumps(stat,ensure_ascii=False),flush=True)
            del arr,vals,finite

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['audit','copy','samples']);args=p.parse_args()
    {'audit':audit,'copy':copy_all,'samples':samples}[args.action]()
