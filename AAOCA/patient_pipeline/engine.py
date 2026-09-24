"""Process one patient's indexed headers at a time."""
from collections import defaultdict
from pathlib import Path

import numpy as np

from common import token, write_json
from convert_ct import convert_series, geometry
from convert_us import convert_file
from patient_pipeline.imaging import (BAD_PIXELS, classify_ct, dataset_from_tags,
                                      deduplicate, quality_metrics, unique_values)
from patient_pipeline.runtime import fatal_io

CT_PARAMS=['Rows','Columns','PixelSpacing','SliceThickness','SpacingBetweenSlices','KVP','XRayTubeCurrent',
           'Exposure','ConvolutionKernel','ReconstructionDiameter','CTDIvol','RescaleSlope','RescaleIntercept',
           'Manufacturer','ManufacturerModelName','ProtocolName','SeriesDescription','ContrastBolusAgent',
           'TriggerTime','TemporalPositionIdentifier','StudyDate','ImageOrientationPatient']


def availability(total,good,audit):
    if not total:return '未发现文件'
    if audit:return '仅文件头检查_像素未全检'
    if good==total:return '全部转换验证通过_临床完整性未知'
    return '部分转换失败' if good else '未能转换'


def process_patient(ident,records,patient,out,audit=False,block_reason='',runtime=None):
    folder=out/'patients'/('patient_'+token(ident));folder.mkdir(parents=True,exist_ok=True)
    issues=[];ct_rows=[];us_rows=[];quality=[]
    def issue(code,message,source='',severity='warning'):
        issues.append({'patient_id':ident,'severity':severity,'code':code,'source':source,'message':message})
        if runtime and severity=='error':runtime.control.emit(level='error',message=f'已记录问题并继续：{message}')
    usable=[r for r in records if not r.get('error')]
    kept,duplicates,conflicts=deduplicate(usable)
    conflict_paths={r['path'] for r in conflicts}
    for dup in duplicates:issue('EXACT_DUPLICATE','字节完全一致，输出仅保留一份；原文件未删除。',dup['source'],'info')
    for r in conflicts:issue('SOP_CONTENT_CONFLICT','同一SOP编号对应不同内容，相关序列/文件不转换。',r['path'],'error')
    if block_reason:issue('PATIENT_IDENTITY_OR_SCAN_BLOCK',block_reason,severity='error')
    write_json(folder/'重复文件记录.json',duplicates)
    ct=[r for r in kept if r['tags'].get('Modality')=='CT']
    us=[r for r in kept if r['tags'].get('Modality')=='US']
    groups=defaultdict(list)
    for r in ct:groups[(r['tags'].get('StudyInstanceUID') or '',r['tags'].get('SeriesInstanceUID') or '')].append(r)
    cta_count=0;candidate_count=0;ct_success=0;us_success=0
    for (study,series),items in sorted(groups.items()):
        if runtime:runtime.control.checkpoint()
        label,evidence=classify_ct(items[0]['tags'])
        all_labels={classify_ct(r['tags'])[0] for r in items}
        if len(all_labels)>1:label='序列描述不一致_待确认CTA'
        if label=='CTA_描述明确':cta_count+=sum(int(r['tags'].get('NumberOfFrames') or 1) for r in items)
        if label=='增强心脏CT_疑似CTA':candidate_count+=sum(int(r['tags'].get('NumberOfFrames') or 1) for r in items)
        key=token(study+'|'+series);dest=folder/'ct'/('series_'+key)
        bad=[r for r in items if r['pixel_status'] in BAD_PIXELS]
        row={'患者编号':ident,'检查UID':study,'序列UID':series,'CTA判定':label,'CTA依据':evidence,
             '去重后文件数':len(items),'声明切片或帧数':sum(int(r['tags'].get('NumberOfFrames') or 1) for r in items),
             '像素长度异常文件数':len(bad),'几何检查':'','实际层间距mm':None,'位置覆盖mm':None,
             '预期切片数标签':unique_values(items,'NumberOfSeriesRelatedInstances'),'完整性说明':'临床扫描范围/原始导出是否齐全无法仅凭现有文件确认',
             '转换状态':'未转换','输出目录':'','错误':'','本次动作':'',
             **{k:unique_values(items,k) for k in CT_PARAMS}}
        failures=[]
        if block_reason:failures.append(block_reason)
        if any(r['identity_basis']!='PatientID' for r in items):failures.append('患者编号仅由文件夹推断，身份需核对')
        if any(r['path'] in conflict_paths for r in items):failures.append('同SOP内容冲突')
        if bad:failures.append(f'{len(bad)}个文件缺像素或像素长度异常')
        for r in bad:issue('CT_PIXEL_PAYLOAD',r['pixel_status'],r['path'],'error')
        if not study or not series:failures.append('缺少检查或序列UID')
        try:
            headers=[dataset_from_tags(r['tags']) for r in items]
            order,affine,positions,step=geometry(headers)
            row.update(几何检查='通过',实际层间距mm=step,位置覆盖mm=step*(len(items)-1))
            numbers=[r['tags'].get('InstanceNumber') for r in items]
            numbers=[int(n) for n in numbers if n is not None]
            if numbers and len(set(numbers))!=max(numbers)-min(numbers)+1:
                issue('INSTANCE_NUMBER_GAPS','InstanceNumber不连续，仅作提示；真实层间距另行检查。',items[0]['path'])
        except Exception as exc:
            row['几何检查']='未通过';failures.append(str(exc))
        declared=row['预期切片数标签']
        try:
            bad_declared=bool(declared) and (len(declared)!=1 or int(declared[0])!=len(items))
        except (ValueError,TypeError,OverflowError):
            bad_declared=True
        if bad_declared:
            issue('DECLARED_SERIES_COUNT_MISMATCH','文件数量与NumberOfSeriesRelatedInstances不一致，暂不输出该序列。',items[0]['path'],'error')
            failures.append('声明序列数量不一致')
        if failures:
            if runtime and dest.exists():runtime.archive(dest)
            row.update(转换状态='拒绝转换',错误='；'.join(dict.fromkeys(failures)))
            issue('CT_SERIES_REJECTED',row['错误'],items[0]['path'],'error')
        elif audit:
            row['转换状态']='仅检查文件头'
            if runtime:runtime.seen.add('CT:'+str(dest.relative_to(out)))
        else:
            metrics=[]
            try:
                paths=[r['path'] for r in items]
                if runtime:
                    meta,metrics,action=runtime.convert('CT',paths,dest,
                        lambda quality,check:convert_series(paths,dest,quality_callback=quality,checkpoint=check))
                    row['本次动作']=action
                else:
                    meta=convert_series(paths,dest,quality_callback=lambda a,i,n:metrics.append(quality_metrics(a,i,n)))
                row.update(转换状态='成功',输出目录=str(dest.relative_to(out)))
                ct_success+=1
                quality.append({'kind':'CT','series':key,'sampled_slices':metrics,'note':'Basic numerical checks, not clinical quality grading.'})
                if any(m['constant'] for m in metrics):issue('CT_CONSTANT_SAMPLE','抽查切片数值恒定，请复核。',items[0]['path'])
            except Exception as exc:
                if fatal_io(exc):raise
                row.update(转换状态='失败',错误=str(exc));issue('CT_CONVERSION_FAILED',str(exc),items[0]['path'],'error')
        ct_rows.append(row)
        if runtime:runtime.advance(len(items),'CT序列已处理，进度已保存')
    for r in us:
        if runtime:runtime.control.checkpoint()
        tags=r['tags'];nf=int(tags.get('NumberOfFrames') or 1);is_multi=tags.get('NumberOfFrames') is not None
        dest=folder/'us'/('item_'+token(r['path']))
        row={'患者编号':ident,'源文件':r['path'],'检查UID':tags.get('StudyInstanceUID'), '检查日期':tags.get('StudyDate'),
             '类型':'视频/多帧' if is_multi else '静态图','帧数':nf,'宽':tags.get('Columns'),'高':tags.get('Rows'),
             '颜色形式':tags.get('PhotometricInterpretation'),'位深':tags.get('BitsAllocated'),
             '时间信息':r['timing'],'区域标定':r['ultrasound_regions'],'传输语法':r['transfer_syntax'],
             '有损压缩历史':tags.get('LossyImageCompression'),'历史压缩比':tags.get('LossyImageCompressionRatio'),
             '像素检查':r['pixel_status'],'转换状态':'未转换','输出目录':'','错误':'','本次动作':''}
        failure=block_reason or ('同SOP内容冲突' if r['path'] in conflict_paths else '')
        if r['identity_basis']!='PatientID':failure=failure or '患者身份仅由目录推断，待核对'
        if r['pixel_status'] in BAD_PIXELS:failure=failure or r['pixel_status']
        if not r['ultrasound_regions']:issue('US_SCALE_UNKNOWN','缺少超声区域标定，不能推算真实组织尺寸。',r['path'])
        if is_multi and (not r['timing'] or r['timing']['source']=='unknown'):issue('US_TIMING_UNKNOWN','帧时间未知，保留帧但不虚构帧率。',r['path'])
        if failure:
            if runtime and dest.exists():runtime.archive(dest)
            row.update(转换状态='拒绝转换',错误=failure);issue('US_REJECTED',failure,r['path'],'error')
        elif audit:
            row['转换状态']='仅检查文件头'
            if runtime:runtime.seen.add('US:'+str(dest.relative_to(out)))
        else:
            metrics=[]
            try:
                if runtime:
                    _,metrics,action=runtime.convert('US',[r['path']],dest,
                        lambda quality,check:convert_file(r['path'],dest,quality_callback=quality,checkpoint=check))
                    row['本次动作']=action
                else:convert_file(r['path'],dest,quality_callback=lambda a,i,n:metrics.append(quality_metrics(a,i,n)))
                row.update(转换状态='成功',输出目录=str(dest.relative_to(out)));us_success+=1
                quality.append({'kind':'US','source':r['path'],'sampled_frames':metrics})
                if any(m['constant'] for m in metrics):issue('US_CONSTANT_SAMPLE','首/中/末抽查帧存在恒定图像，请复核。',r['path'])
            except Exception as exc:
                if fatal_io(exc):raise
                row.update(转换状态='失败',错误=str(exc));issue('US_CONVERSION_FAILED',str(exc),r['path'],'error')
        us_rows.append(row)
        if runtime:runtime.advance(1,'超声文件已处理，进度已保存')
    raw_ct=sum(r.get('tags',{}).get('Modality')=='CT' for r in records)
    raw_us=sum(r.get('tags',{}).get('Modality')=='US' for r in records)
    summary={'患者编号':ident,'姓名':patient.get('name',''),'住院号':patient.get('fields',{}).get('住院号',''),
             '病例关联':'患者表编号冲突' if patient.get('conflict') else ('患者表编号匹配' if patient.get('sources') else '未匹配患者表'),
             'DICOM文件数':len(records),'文件头失败数':sum(bool(r.get('error')) for r in records),
             'CT文件数':raw_ct,'CT去重后文件数':len(ct),'CT序列数':len(groups),'CT成功序列':ct_success,
             'CT总切片或帧数':sum(r['声明切片或帧数'] for r in ct_rows),
             'CTA描述明确切片数':cta_count,'疑似CTA切片数':candidate_count,
             'CT数据状态':availability(len(groups),ct_success,audit),
             '超声文件数':raw_us,'超声去重后文件数':len(us),'超声静态图数':sum(r['类型']=='静态图' for r in us_rows),
             '超声视频数':sum(r['类型']=='视频/多帧' for r in us_rows),'超声总帧数':sum(r['帧数'] for r in us_rows),
             '超声成功文件':us_success,'超声数据状态':availability(len(us),us_success,audit),
             '超声检查内容完整性':'未知_需核对所需切面与心动周期' if us else '输入中未发现超声',
             'CT扫描范围完整性':'未知_需原始检查目录或人工核对' if ct else '输入中未发现CT',
             '去除完全重复文件数':len(duplicates),'患者数据目录':str(folder.relative_to(out))}
    if any(r.get('error') for r in records) or (block_reason and (not ct or not us)):
        if not ct:summary['CT数据状态']='无法判定_文件头或身份需核对'
        if not us:summary['超声数据状态']='无法判定_文件头或身份需核对'
    if audit:
        if any(r['转换状态']=='拒绝转换' for r in ct_rows):summary['CT数据状态']='仅文件头检查_已发现异常'
        if any(r['转换状态']=='拒绝转换' for r in us_rows):summary['超声数据状态']='仅文件头检查_已发现异常'
    if not raw_ct:issue('CT_NOT_FOUND','输入范围内未发现CT；不代表患者从未做过CT。')
    if not raw_us:issue('US_NOT_FOUND','输入范围内未发现超声；不代表患者从未做过超声。')
    write_json(folder/'图像抽查.json',quality)
    return summary,ct_rows,us_rows,issues
