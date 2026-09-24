"""病例与影像一键处理。默认统计、检查、转换；原文件只读。"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sqlite3
import sys
import traceback

from common import token, write_json
from patient_pipeline.clinical import alias_map, normal_id, process_documents, read_roster
from patient_pipeline.engine import process_patient
from patient_pipeline.imaging import folder_id, read_header
from patient_pipeline.reports import write_csv, write_html
from patient_pipeline.runtime import Control, Runtime, RunLock, StopRequested, fatal_io

PROJECT=Path(__file__).resolve().parent
SKIP_NAMES={'.venv','__pycache__','.git','__MACOSX','back','test_outputs'}


def now():return datetime.now(timezone.utc).isoformat()


def validate_paths(source,output):
    source=Path(source).expanduser().resolve();output=Path(output).expanduser().resolve()
    if not source.is_dir():raise ValueError('输入必须是已解压的数据目录。')
    if source==output or source.is_relative_to(output):raise ValueError('输出不能等于输入或位于输入的上层目录。')
    if output.exists():
        if not output.is_dir():raise ValueError('输出路径必须是目录。')
        contents=[p for p in output.iterdir() if p.name!='.pipeline.lock']
        if contents and not (output/'processing_manifest.json').is_file():
            raise ValueError('此非空目录不是本程序的结果目录，不能自动覆盖。请选择空目录或之前的结果目录。')
    return source,output


def discover(source,output,control=None):
    result={'dicom':[],'xlsx':[],'pdf':[],'archives':[],'unsupported':[],'skipped_directories':[]}
    result['scan_errors']=[]
    for current,dirs,files in os.walk(source,followlinks=False,onerror=lambda exc:result['scan_errors'].append(str(exc))):
        if control:control.progress('discover',0,0,'正在查找文件…')
        current=Path(current)
        if current!=source and ((current/'processing_manifest.json').exists() or current==output or current==PROJECT):
            result['skipped_directories'].append(str(current));dirs[:]=[];continue
        allowed=[]
        for name in sorted(dirs):
            p=current/name
            if name in SKIP_NAMES or p.is_symlink() or p.is_junction() or p.resolve()==output or p.resolve()==PROJECT:
                result['skipped_directories'].append(str(p))
            else:allowed.append(name)
        dirs[:]=allowed
        for name in sorted(files):
            if name.startswith(('._','~$')):continue
            p=current/name
            if p.is_symlink():continue
            ext=p.suffix.lower()
            if ext=='.dcm':result['dicom'].append(p)
            elif ext=='.xlsx':result['xlsx'].append(p)
            elif ext=='.pdf':result['pdf'].append(p)
            elif ext in {'.zip','.7z','.rar','.change2zip'} or re.fullmatch(r'\.z\d+',ext):result['archives'].append(p)
            elif ext in {'.doc','.docx','.xls','.txt','.csv','.nii','.gz','.avi','.mp4'}:result['unsupported'].append(p)
    return result


def canonical(ident,aliases):
    hits=aliases.get(ident,set())
    return next(iter(hits)) if len(hits)==1 else ident


def selected_path(path,selected,aliases):
    ident=folder_id(path)
    if ident and canonical(ident,aliases) in selected:return True
    return any(re.search(r'(?<!\d)'+re.escape(i)+r'(?!\d)',path.name) for i in selected)


def run(source,output,patients=None,audit=False,control=None,force=False):
    source,out=validate_paths(source,output)
    out.mkdir(parents=True,exist_ok=True)
    with RunLock(out):
        previous=out/'processing_manifest.json'
        if previous.exists():
            try:old=json.loads(previous.read_text(encoding='utf-8'))
            except (ValueError,OSError) as exc:raise ValueError('旧运行清单无法读取，请保留该目录并选择新输出目录。') from exc
            if Path(old.get('input','')).resolve()!=source:raise ValueError('此结果目录属于其他输入路径，请选择新目录。')
            if sorted(old.get('selected_patient_ids') or [])!=sorted(patients or []):
                raise ValueError('续跑的患者范围须与上次一致；改变范围请选新输出目录。')
        runtime=Runtime(out,control,force)
        try:return _run(source,out,patients,audit,runtime)
        finally:runtime.close()


def _run(source,out,patients,audit,runtime):
    control=runtime.control;details=out/'details'
    manifest={'schema_version':2,'status':'running','started_utc':now(),'input':str(source),'output':str(out),
              'mode':'仅统计及文件头检查' if audit else '统计、检查、转换','selected_patient_ids':patients or [],
              'identity_note':'Patient-level association, not encounter-level pairing. No fuzzy name merge.',
              'privacy':'Local output is NOT formally deidentified. No upload or online AI used.',
              'source_modified':False,'resume_supported':True}
    manifest_path=out/'processing_manifest.json';write_json(manifest_path,manifest)
    database=None
    summaries=[];ct_rows=[];us_rows=[]
    try:
        write_html(out,[],manifest)
        inventory=discover(source,out,control)
        manifest['discovered_files']={k:len(v) for k,v in inventory.items()}
        write_json(details/'输入目录清单.json',{k:[str(p) for p in v] for k,v in inventory.items()})
        if inventory['scan_errors']:
            raise RuntimeError('输入目录扫描不完整，已保留原有结果和检查点。请恢复目录访问权限/磁盘连接后续跑。详情见 details/输入目录清单.json。')
        control.progress('headers',0,1,'读取患者表并检查文件头',True)
        roster,issues=read_roster(inventory['xlsx'],control);aliases=alias_map(roster)
        issues.extend({'code':'DIRECTORY_SCAN_FAILED','message':e,'severity':'error'} for e in inventory['scan_errors'])
        selected={canonical(normal_id(i),aliases) for i in (patients or [])}
        if selected:roster={i:r for i,r in roster.items() if i in selected}
        # Keep the full alias map to detect ID ambiguity even in a sample run.
        database=sqlite3.connect(details/'文件索引.sqlite')
        database.execute('DROP TABLE IF EXISTS files')
        database.execute('CREATE TABLE files(path TEXT PRIMARY KEY,pid TEXT,modality TEXT,study TEXT,series TEXT,sop TEXT,payload TEXT)')
        blocks=defaultdict(list);issuers=defaultdict(set);observed=set();header_reads=0;unknown_bad=False
        paths=[p for p in inventory['dicom'] if not selected or selected_path(p,selected,aliases)]
        for index,path in enumerate(paths,1):
            control.progress('headers',index-1,len(paths),f'检查文件头 {index}/{len(paths)}')
            try:
                record=runtime.header(path,read_header);raw_id=record['patient_id']
                ident=canonical(raw_id,aliases) if raw_id else 'unmatched_'+token(str(path.parent))
                tags=record['tags'];record['raw_patient_id']=raw_id;record['patient_id']=ident
                if len(aliases.get(raw_id,set()))>1:blocks[ident].append('患者编号在病例表中存在多个对应关系')
                if record['issuer']:issuers[ident].add(record['issuer'])
                declared_folder=record['folder_patient_id']
                if declared_folder and raw_id and canonical(declared_folder,aliases)!=ident:
                    blocks[ident].append('DICOM患者编号与父目录编号冲突')
                    issues.append({'patient_id':ident,'severity':'error','code':'FOLDER_PATIENT_ID_CONFLICT','source':str(path),'message':'DICOM编号与目录编号不一致，暂停此患者转换。'})
                if record['identity_basis']!='PatientID':blocks[ident].append('DICOM缺少患者编号，身份未验证')
                if not tags.get('Modality'):blocks[ident].append('存在缺少模态的DICOM，不能确认其归属')
                if record['warning_types']:
                    # Keep warning classes and source paths without dumping patient text into the console.
                    record['header_warning_note']='DICOM字段存在格式告警，原值未改写'
                if tags.get('Modality') not in {'CT','US'}:
                    issues.append({'patient_id':ident,'code':'MODALITY_NOT_CONVERTED','source':str(path),'message':'非CT/US或缺少模态，仅建立索引。'})
            except Exception as exc:
                if fatal_io(exc):raise
                fallback=folder_id(path);ident=canonical(fallback,aliases) if fallback else 'unmatched_'+token(str(path.parent))
                try:stat=path.stat();size,mtime=stat.st_size,stat.st_mtime_ns
                except OSError:size,mtime=None,None
                record={'path':str(path),'patient_id':ident,'error':str(exc),'bytes':size,'mtime_ns':mtime}
                tags={};blocks[ident].append('存在无法读取的DICOM文件头，无法可靠判断缺失属于哪一序列')
                if not fallback:unknown_bad=True
                issues.append({'patient_id':ident,'severity':'error','code':'DICOM_HEADER_FAILED','source':str(path),'message':str(exc)})
            if selected and ident not in selected:
                issues.append({'patient_id':ident,'severity':'error','code':'SAMPLE_ID_MISMATCH','source':str(path),'message':'所选路径内患者ID不属于样本名单；不转换。'})
                if record.get('folder_patient_id'):blocks[canonical(record['folder_patient_id'],aliases)].append('样本目录内出现其他患者ID')
                continue
            observed.add(ident)
            database.execute('INSERT INTO files VALUES(?,?,?,?,?,?,?)',
                (str(path),ident,tags.get('Modality',''),tags.get('StudyInstanceUID',''),tags.get('SeriesInstanceUID',''),tags.get('SOPInstanceUID',''),json.dumps(record,ensure_ascii=False)))
            if index%500==0:database.commit()
        database.commit();database.execute('CREATE INDEX by_patient ON files(pid)')
        control.progress('headers',len(paths),len(paths),'文件头检查完成',True)
        for sop, in database.execute("SELECT sop FROM files WHERE sop!='' GROUP BY sop HAVING COUNT(DISTINCT pid)>1"):
            for ident, in database.execute('SELECT DISTINCT pid FROM files WHERE sop=?',(sop,)):blocks[ident].append('同一SOP影像编号跨患者重复，需核对')
        for ident,values in issuers.items():
            if len(values)>1:blocks[ident].append('同一患者编号存在多个编号签发机构，不能自动合并')
        all_patients=set(roster)|observed|selected
        if not all_patients:
            raise ValueError('未找到可处理的患者。请确认输入为已解压目录，含.dcm和/或有患者编号列的.xlsx。')
        for ident in all_patients:
            roster.setdefault(ident,{'name':'','aliases':[ident],'fields':{},'sources':[],'conflict':False})
            if roster[ident].get('conflict'):blocks[ident].append('患者主表编号冲突')
            if unknown_bad:blocks[ident].append('输入存在无法归属的坏DICOM头，暂阻断转换，需修复或缩小输入范围')
        # Document links are resolved after DICOM-only patients have been added.
        print('正在关联病例，并提取所选患者的PDF文本',flush=True)
        documents,doc_issues,pdf_reads=process_documents(inventory['pdf'],roster,out,selected,runtime)
        issues.extend(doc_issues)
        runtime.images_total=database.execute("SELECT COUNT(*) FROM files WHERE modality IN ('CT','US')").fetchone()[0]
        for index,ident in enumerate(sorted(all_patients),1):
            control.checkpoint()
            print(f'正在处理患者 {index}/{len(all_patients)}',flush=True)
            control.emit(message=f'患者 {index}/{len(all_patients)}',state='running')
            records=[json.loads(row[0]) for row in database.execute('SELECT payload FROM files WHERE pid=? ORDER BY path',(ident,))]
            try:
                summary,ct,us,problems=process_patient(ident,records,roster[ident],out,audit,'；'.join(sorted(set(blocks[ident]))),runtime)
            except Exception as exc:
                if fatal_io(exc):raise
                # Preserve completed per-file checkpoints, report this patient,
                # and continue with independent patients.
                summary={'患者编号':ident,'姓名':roster[ident].get('name',''),'病例关联':'患者处理异常，需重试',
                    'DICOM文件数':len(records),'CT数据状态':'处理异常_部分结果已保留','超声数据状态':'处理异常_部分结果已保留',
                    '患者数据目录':str(Path('patients')/('patient_'+token(ident))),'处理错误':str(exc)}
                ct=[];us=[];problems=[{'patient_id':ident,'severity':'error','code':'PATIENT_PROCESSING_FAILED','message':str(exc)}]
                runtime.protected_prefixes.update({kind+':'+str(Path('patients')/('patient_'+token(ident))) for kind in ['CT','US']})
                control.emit(message=f'患者 {index} 遇到错误，已记录，继续下一位',state='running')
            issues.extend(problems)
            linked=[d for d in documents if d['patient_id']==ident and d.get('verified_link')]
            pending=[d for d in documents if d['patient_id']==ident and not d.get('verified_link')]
            summary.update(已核实PDF数=len(linked),已核实病例PDF数=sum(d['kind']=='病例' for d in linked),
                           已核实检验PDF数=sum(d['kind']=='检验' for d in linked),待确认PDF数=len(pending),
                           需OCR的PDF数=sum(bool(d['empty_pages']) for d in linked),
                           病例表标记=roster[ident].get('fields',{}),临床影像关联层级='患者级；同次就诊未确认')
            if not linked:issues.append({'patient_id':ident,'code':'NO_VERIFIED_DOCUMENT','message':'没有经内容编号核实的病例PDF，不能视为完整病例。'})
            for kind,column in [('病例','病历'),('检验','检验')]:
                expected=roster[ident].get('fields',{}).get(column)
                present=any(d['kind']==kind for d in linked)
                if expected in (1,'1') and not present:issues.append({'patient_id':ident,'code':'ROSTER_DOCUMENT_MISSING','message':f'病例表标记有{kind}，但未找到已核实的对应PDF。'})
            folder=out/summary['患者数据目录']
            try:
                folder.mkdir(parents=True,exist_ok=True)
                write_json(folder/'patient.json',{'summary':summary,'clinical_roster':roster[ident],'clinical_documents':linked,
                    'pending_documents':pending,'ct_series':ct,'ultrasound_files':us})
            except OSError as exc:
                if fatal_io(exc):raise
                issues.append({'patient_id':ident,'severity':'error','code':'PATIENT_REPORT_WRITE_FAILED','message':str(exc)})
            summaries.append(summary);ct_rows.extend(ct);us_rows.extend(us)
        # Cheap source guard: original size and mtime must remain unchanged. Pixel
        # converters also record SHA256. This is not claimed as a full source hash audit.
        changed=[]
        control.progress('reports',0,1,'汇总结果与保存报告',True)
        for raw, in database.execute('SELECT payload FROM files'):
            record=json.loads(raw)
            try:
                stat=Path(record['path']).stat()
                if (stat.st_size,stat.st_mtime_ns)!=(record['bytes'],record['mtime_ns']):changed.append(record['path'])
            except OSError:changed.append(record['path'])
        if changed:
            for p in changed:issues.append({'severity':'error','code':'SOURCE_CHANGED_DURING_RUN','source':p,'message':'运行期间源文件大小/修改时间变化，请重新处理。'})
        manifest.update(status='completed_with_issues' if issues else 'completed',completed_utc=now(),
            patient_count=len(summaries),header_files_read=runtime.counts['headers_read'],header_files_indexed=len(paths),**pdf_reads,
            ct_converted=sum(r['转换状态']=='成功' for r in ct_rows),us_converted=sum(r['转换状态']=='成功' for r in us_rows),
            issues_count=len(issues),source_stat_check_passed=not changed,
            resume_counts=runtime.counts,
            data_reading_note='Unchanged indexed headers and verified completed outputs may be reused. New image conversions decode their source pixels. PDF text extraction is local and matched to patients.',
            completeness_note='No inferred proof of clinical CTA coverage, ultrasound view coverage, or encounter pairing.')
        issue_rows=[{'患者编号':i.get('patient_id',''),'级别':i.get('severity','warning'),'问题代码':i.get('code',''),
                     '源文件':i.get('source',''),'说明':i.get('message','')} for i in issues]
        doc_rows=[{'患者编号':d['patient_id'],'文件类型':d['kind'],'源文件':d['source'],'关联依据':d['match_basis'],
                   '状态':d['status'],'总页数':d['pages'],'读取页数':d['pages_read'],'需OCR页码':d['empty_pages'],'输出文本':d['text_file']} for d in documents]
        write_csv(out/'患者总表.csv',summaries);write_csv(out/'CT序列表.csv',ct_rows)
        write_csv(out/'超声文件表.csv',us_rows);write_csv(out/'病例文档表.csv',doc_rows);write_csv(out/'问题清单.csv',issue_rows)
        write_json(details/'患者总表.json',summaries);write_json(details/'问题清单.json',issues)
        runtime.cleanup_stale()
        write_json(manifest_path,manifest);write_html(out,summaries,manifest)
        control.progress('reports',1,1,'处理完成',True);control.emit(state='completed',counts=runtime.counts.copy())
        print(f'完成：{len(summaries)}位患者；CT成功{manifest["ct_converted"]}个序列；超声成功{manifest["us_converted"]}个文件。',flush=True)
        print(f'结果：{out / "查看结果.html"}',flush=True)
        return 2 if issues else 0
    except BaseException as exc:
        manifest.update(status='interrupted' if isinstance(exc,(KeyboardInterrupt,StopRequested)) else 'failed',error=str(exc),completed_utc=now(),resume_counts=runtime.counts)
        try:
            write_json(manifest_path,manifest);write_html(out,summaries,manifest)
            (details/'error.log').write_text(traceback.format_exc(),encoding='utf-8')
        except OSError:pass  # A full/read-only disk must not mask the original failure.
        raise
    finally:
        if database:database.close()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,help='包含影像、病例表和PDF的已解压总目录')
    parser.add_argument('--output',type=Path,help='新建结果目录，不得已存在')
    parser.add_argument('--patients',nargs='+',help='可选：只处理指定患者编号；依据目录/文件名筛选，适合少量测试')
    parser.add_argument('--audit-only',action='store_true',help='可选：只统计文件头和提取病例，不转换影像，像素未全检')
    parser.add_argument('--force',action='store_true',help='强制重做；旧结果先归档，原始资料不变')
    args=parser.parse_args(argv)
    try:
        source=args.input or Path(input('请输入输入路径：').strip().strip('"'))
        output=args.output or Path(input('请输入输出路径（可续跑）：').strip().strip('"'))
        last_stage=['']
        def show(event):
            stage=event.get('stage')
            if stage and (stage!=last_stage[0] or event.get('done')==event.get('total')):
                print(event.get('message',''),flush=True);last_stage[0]=stage
        return run(source,output,args.patients,args.audit_only,Control(show),args.force)
    except KeyboardInterrupt:
        print('已中断。已完成进度已保存，再次使用相同路径可继续。');return 130
    except Exception as exc:
        print(f'未完成：{exc}',file=sys.stderr);return 1


if __name__=='__main__':sys.exit(main())
