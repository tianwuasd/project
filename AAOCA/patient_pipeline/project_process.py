"""Independent conversion units; verified output revisions survive failed updates."""
from pathlib import Path
import shutil
import uuid

from common import ConversionError, sha256, staged_output, write_json
from convert_ct import convert_series, geometry
from convert_us import convert_file
from patient_pipeline.imaging import dataset_from_tags, BAD_PIXELS
from patient_pipeline.project_model import VERSION, digest, issue, patient_folder
from patient_pipeline.project_repairs import select_copies, phase_groups, phase_geometry
from patient_pipeline.runtime import fatal_io


def signature(records,options):
    return digest([VERSION,[(r['key'],r.get('stamp'),r.get('pid'),r.get('verified'),r.get('identity_issues'),
                             r.get('read_error')) for r in records],options.recover_copies,options.split_phases])


def available_sources(records,control):
    """An offline record can be represented only by a byte-identical available copy."""
    available=[r for r in records if Path(r['path']).is_file()]
    offline=[r for r in records if not Path(r['path']).is_file()]
    cached={}
    for r in offline:
        expected=r.get('sha256')
        if not expected:raise ConversionError('源文件离线且无已记录的哈希证明；已有成果保留，恢复源路径后可重试')
        found=False
        for candidate in available:
            if candidate.get('header',{}).get('tags',{}).get('SOPInstanceUID')!=r.get('header',{}).get('tags',{}).get('SOPInstanceUID'):continue
            control.checkpoint()
            if candidate['key'] not in cached:cached[candidate['key']]=sha256(candidate['path'])
            if cached[candidate['key']]==expected:found=True;break
        if not found:raise ConversionError('源文件离线，未找到哈希一致的完整副本；已有成果保留')
    return available


def error_code(message,kind):
    if message.strip().lower() in {'empty','missing','truncated','length_mismatch'}:return 'PIXEL_INVALID'
    checks=[('SOP_CONTENT_CONFLICT','SOP_CONTENT_CONFLICT'),('FrameTime','US_TIMING_INVALID'),
            ('像素','PIXEL_INVALID'),('Pixel','PIXEL_INVALID'),('pixel','PIXEL_INVALID'),
            ('dimensions','CT_DIMENSIONS_MISMATCH'),('At least two','CT_SINGLE_SLICE'),
            ('ImageOrientationPatient','CT_ORIENTATION_MISSING'),('orientation','CT_ORIENTATION_MISMATCH'),
            ('positions','CT_POSITION_CONFLICT'),('spacing','CT_SPACING_INVALID'),('期相','CT_PHASE_REVIEW')]
    return next((c for word,c in checks if word in message),kind+'_CONVERSION_FAILED')


def outputs_valid(store,unit,deep=False):
    if not unit.get('outputs'):return False
    for rel in unit['outputs']:
        path=Path(rel) if unit.get('external_root') else store.path(rel)
        if not path.is_file():return False
        proof=unit.get('proof',{}).get(rel)
        if proof and (path.stat().st_size!=proof['bytes'] or (deep and sha256(path)!=proof['sha256'])):return False
    return True


def patient_directory(store,pid):
    old=store.get('folders',pid)
    if old:return store.path(old)
    rel=Path('patients')/'待分类'/patient_folder(pid)
    store.put('folders',pid,rel.as_posix());return store.path(rel)


def process_one(store,key,records,options,control):
    kind=key.split(':',1)[0];pid=records[0].get('pid','')
    sig=signature(records,options);old=store.unit(key)
    input_signature=signature(records,type('InputOnly',(),{'recover_copies':False,'split_phases':False})())
    issues=[i for r in records for i in r.get('issues',[])+r.get('identity_issues',[])]
    unit={'kind':kind,'pid':pid,'signature':sig,'input_signature':input_signature,'sources':[r['key'] for r in records],
          'source_count':len(records),'status':'failed','outputs':[],'issues':issues,'repairs':[]}
    if old and old.get('retired'):
        old.pop('retired');store.put('units',key,old)
    if old and old.get('status')=='success' and not outputs_valid(store,old):
        unit['invalidate_previous']=True
    if old and old.get('status')=='success' and outputs_valid(store,old):
        # Legacy accepted recoveries have provenance, and are not invalidated by toggles or an offline source.
        if old.get('signature')==sig or old.get('input_signature')==input_signature or (old.get('legacy') and not any(r.get('new_since_legacy') for r in records)):
            return 'reused'
        last=old.get('last_attempt',{})
        if last.get('signature')==sig and options.action!='retry':return 'reused'
    elif old and old.get('signature')==sig and options.action not in {'retry'}:
        if old.get('status')!='success' and not (old.get('status')=='audited' and options.action!='audit'):return 'reused'
    try:
        if any(r.get('read_error') for r in records):
            raise ConversionError('源文件读取失败：'+next(r['read_error'] for r in records if r.get('read_error')))
        if kind=='PDF':
            r=records[0];unit['pages']=r.get('pages',0);unit['low_text_pages']=r.get('low_text_pages',[])
            unit['outputs']=[r['text_file']] if r.get('text_file') else []
            if len({r.get('pid') for r in records if r.get('pid')})>1:raise ConversionError('同一PDF副本关联到不同患者')
            if not all(r.get('verified') for r in records):unit['status']='pending';unit['invalidate_previous']=True
            else:unit['status']='success'
            if unit['low_text_pages']:
                unit['issues'].append(issue('PDF_TEXT_INCOMPLETE_NOTE','部分页面文字提取不足；仅作备注，不计错误',pid,r['path'],'info',pages=unit['low_text_pages']))
            unit['match_basis']=r.get('match_basis','')
            if unit['status']=='success' and not r.get('legacy'):
                dest=patient_directory(store,pid)/'clinical'/('document_'+digest(key)[:16])/('revision_'+uuid.uuid4().hex[:12])
                dest.parent.mkdir(parents=True,exist_ok=True)
                original=store.path(r['text_file'])
                with staged_output(dest) as stage:
                    shutil.copyfile(original,stage/'text.txt')
                    if original.with_name('pages.json').is_file():shutil.copyfile(original.with_name('pages.json'),stage/'pages.json')
                    write_json(stage/'association.json',{'patient_id':pid,'basis':unit['match_basis'],
                        'source':r['path'],'sha256':r.get('sha256'),'low_text_pages':unit['low_text_pages']})
                unit['outputs']=[(dest/'text.txt').relative_to(store.root).as_posix()]
            store.save_unit(key,unit);return 'processed'
        if not pid or len({r.get('pid') for r in records})!=1:
            unit['invalidate_previous']=True
            raise ConversionError('同一影像处理单元的患者归属不唯一')
        if any(i.get('severity','error')=='error' for i in issues):
            unit['invalidate_previous']=True
            raise ConversionError('编号关联未核实；仅暂停此影像单元')
        if not all(r.get('verified') for r in records):
            unit['invalidate_previous']=True;raise ConversionError('影像患者编号未核实')
        if kind not in {'CT','US'}:
            unit['status']='unsupported';unit['issues'].append(issue('MODALITY_NOT_CONVERTED','该影像类型仅登记，未执行转换',pid,records[0]['path'],'warning'))
            store.save_unit(key,unit);return 'processed'
        available=available_sources(records,control)
        approved_sops={r['sop'] for r in (old or {}).get('repairs',[]) if r.get('type')=='complete_copy' and r.get('sop')}
        selected,repairs=select_copies(available,True if options.recover_copies else approved_sops,control.checkpoint)
        if len(available)!=len(records):repairs.append({'type':'relocated_identical_copy','count':len(records)-len(available),'proof':'离线源已有SHA256与本次实际副本完全一致'})
        unit['repairs']=repairs;unit['unique_files']=len(selected)
        bad=[r for r in selected if r['header']['pixel_status'] in BAD_PIXELS]
        if bad:raise ConversionError(f'{len(bad)} 个文件像素缺失、为空或长度不符')
        groups=None
        if kind=='CT':
            try:geometry([dataset_from_tags(r['header']['tags']) for r in selected])
            except ConversionError:
                approved_phase=any(r.get('type') in {'phase_split','accepted_legacy_phase'} for r in (old or {}).get('repairs',[]))
                if not options.split_phases and not approved_phase:raise
                groups=phase_groups(selected)
            unit['slices']=len(selected)
        else:
            if len(selected)!=1:raise ConversionError('超声同一编号未能选出唯一文件')
            h=selected[0]['header'];t=h['tags']
            if h.get('timing_error'):raise ConversionError(h['timing_error'])
            unit.update(frames=int(t.get('NumberOfFrames') or 1),image_kind='video' if t.get('NumberOfFrames') else 'still')
            if not h.get('ultrasound_regions'):
                unit['issues'].append(issue('US_CALIBRATION_REVIEW','缺少区域物理标定；像素可转换，但不能直接据此计算毫米或速度',pid,selected[0]['path'],'warning'))
        if options.action=='audit':
            unit['status']='audited';store.save_unit(key,unit);return 'processed'
        dest=patient_directory(store,pid)/kind.lower()/('unit_'+digest(key)[:16])/('revision_'+uuid.uuid4().hex[:12])
        dest.parent.mkdir(parents=True,exist_ok=True)
        checkpoint=lambda *_:control.checkpoint()
        converted_sources=[]
        with staged_output(dest) as stage:
            if kind=='US':
                meta=convert_file(selected[0]['path'],stage/'result',checkpoint=checkpoint)
                converted_sources.append(meta['source'])
                output_names=[Path('result')/('frames.npz' if meta['storage']=='NPZ' else 'image.png')]
            elif groups:
                output_names=[]
                for n,(label,items,proof) in enumerate(groups,1):
                    name=f'phase_{n:02d}'
                    meta=convert_series([r['path'] for r in items],stage/name,checkpoint=checkpoint,
                                   geometry_validator=phase_geometry if proof['spacing_override'] else None)
                    converted_sources.extend(meta['sources_in_slice_order'])
                    output_names.append(Path(name)/'image.nii.gz');unit['repairs'].append({'type':'phase_split',**proof})
                unit['phase_count']=len(groups)
            else:
                meta=convert_series([r['path'] for r in selected],stage/'result',checkpoint=checkpoint)
                converted_sources.extend(meta['sources_in_slice_order'])
                output_names=[Path('result/image.nii.gz')]
            # Recheck every source that contributed before publishing the new version.
            for r in selected:
                st=Path(r['path']).stat()
                if r.get('stamp') and [st.st_size,st.st_mtime_ns]!=r['stamp']:
                    raise ConversionError('转换期间原文件改变，拒绝发布本次结果')
            write_json(stage/'repair_log.json',{'unit':key,'patient':pid,'repairs':unit['repairs'],
                'source_modified':False,'scope':'真实像素完整副本选择或明确期相拆分；无像素补全'})
        unit['outputs']=[(dest/p).relative_to(store.root).as_posix() for p in output_names]
        unit['proof']={rel:{'bytes':store.path(rel).stat().st_size,'sha256':sha256(store.path(rel))} for rel in unit['outputs']}
        from patient_pipeline.project_ingest import source_key
        hashes={source_key(s['path']):s['sha256'] for s in converted_sources}
        for r in selected:
            if r['key'] in hashes:r['sha256']=hashes[r['key']];store.put_source(r,False)
        store.commit()
        unit['status']='success';store.save_unit(key,unit);return 'processed'
    except Exception as exc:
        if fatal_io(exc):raise
        message=str(exc);unit['error']=message
        code=error_code(message,kind)
        if not any(i.get('severity','error')=='error' for i in unit['issues']):
            unit['issues'].append(issue(code,message,pid,records[0]['path']))
        store.save_unit(key,unit);return 'processed'


def process_units(store,options,control):
    keys=[r[0] for r in store.db.execute("SELECT DISTINCT unit_key FROM sources WHERE unit_key!='' ORDER BY unit_key")]
    stats={'images_converted':0,'images_reused':0,'documents_reused':0,'units_processed':0}
    for n,key in enumerate(keys):
        control.progress('images',n,len(keys),'逐序列 / 文件处理，保留其他成功结果')
        records=store.unit_sources(key)
        if options.patients and not any(r.get('pid') in options.patients for r in records):continue
        result=process_one(store,key,records,options,control)
        if result=='reused':stats['documents_reused' if key.startswith('PDF:') else 'images_reused']+=1
        else:
            stats['units_processed']+=1
            if store.unit(key)['status']=='success' and key.startswith(('CT:','US:')):stats['images_converted']+=1
        control.emit(counts=stats)
    # Retire only explicitly replaced source membership, never units whose old sources are offline.
    current=set(keys)
    for key,u in list(store.items('units')):
        if key not in current and u.get('sources'):
            u['retired']=True;store.put('units',key,u)
    control.progress('images',len(keys),len(keys),'独立处理完成',True)
    store.put('meta','last_processing',stats);return stats
