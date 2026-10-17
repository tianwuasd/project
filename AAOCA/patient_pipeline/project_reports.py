"""One snapshot drives patient categories, overview, three issue families and history."""
from collections import Counter, defaultdict
import csv
import html
import io
import json
import os
from pathlib import Path
import uuid

from common import write_json
from patient_pipeline.project_classify import CATEGORIES, subcategory, classify
from patient_pipeline.project_model import digest, issue, now

LABELS={'PIXEL_INVALID':'像素缺失、为空或长度不符','CT_DIMENSIONS_MISMATCH':'图像宽高不一致',
        'CT_SINGLE_SLICE':'只有一张切片','CT_ORIENTATION_MISSING':'缺少有效图像方向',
        'CT_ORIENTATION_MISMATCH':'图像方向不一致','CT_SPACING_INVALID':'层间距或位置不规则',
        'CT_POSITION_CONFLICT':'位置重复或不同期相混组','CT_PHASE_REVIEW':'期相分组未通过验证',
        'SOP_CONTENT_CONFLICT':'同一影像编号对应不同内容','US_TIMING_INVALID':'超声帧时间记录不符合要求',
        'US_CALIBRATION_REVIEW':'超声物理标定待核实','US_SCALE_UNKNOWN':'超声物理标定待核实',
        'DOCUMENT_UNMATCHED':'PDF患者关联待核实','ROSTER_ID_CONFLICT':'患者主表编号冲突',
        'DIRECTORY_ID_CONFLICT':'目录编号与影像内部编号冲突','MODALITY_NOT_CONVERTED':'其他影像类型尚未转换',
        'CT_NOT_PROVIDED':'未提供CT资料','US_NOT_PROVIDED':'未提供超声资料','PDF_NOT_PROVIDED':'未提供PDF资料'}


def reason_label(code):return LABELS.get(code,code)


def report_text(value):
    """Clarify display wording without changing saved evidence or filesystem paths."""
    text=str(value)
    for old,new in [('无其他异常','无其他待处理问题'),
                    ('像素长度异常','像素长度与声明尺寸不符'),
                    ('时间向量异常','时间记录不符合要求'),
                    ('异常视频','未成功转换的视频')]:
        text=text.replace(old,new)
    return text


def issue_description(item):
    if item['code']=='PIXEL_INVALID' and item.get('message')=='empty':
        return '像素数据字段为空；文件中有影像描述信息，但没有可解码的图像像素，无法生成影像输出。'
    return report_text(item['message'])


def family(code):
    if code.endswith('_NOT_PROVIDED'):return '资料缺失'
    if any(word in code for word in ['ROSTER','DIRECTORY_ID','PATIENT_ID','DOCUMENT_UNMATCHED','DOCUMENT_PAGE_ID','ID_RESOLVED','IDENTITY']):return '编号关联'
    return '数据本身'


def remedy(code):
    if code.endswith('_NOT_PROVIDED'):return '如有资料，后续添加对应文件夹；补充后自动更新。'
    if 'ROSTER' in code:return '核对主表重复行；确认同一患者后导入明确的编号/姓名别名规则。'
    if family(code)=='编号关联':return '核对编号与姓名；可启用唯一姓名匹配，同名多患者保留待核实。'
    if 'SOP' in code:return '可启用完整副本选择；身份、几何和有效像素不一致时仍需重新导出。'
    if 'PIXEL' in code:return '寻找同一影像的完整副本或重新导出；不补造缺失像素。'
    if 'PHASE' in code or 'POSITION' in code:return '可尝试按明确期相标签分组；各组须覆盖相同且规则的空间网格。'
    if 'TIMING' in code:return '核对设备导出的帧数和时间标签；不自动补0或猜测时间。'
    if 'CALIBRATION' in code or 'SCALE' in code:return '可查看像素，定量测量前需核实物理标定。'
    if 'SINGLE' in code:return '重新导出完整序列；单张图像不计为三维CT。'
    if 'ORIENTATION' in code or 'DIMENSIONS' in code:return '检查定位图/派生重建图和原始断层图是否混组；不能安全组成同一体积时保留原文件。'
    if code=='PDF_TEXT_INCOMPLETE_NOTE':return '仅备注；不影响已核实的PDF关联，不计错误。'
    return '查看具体文件与原始错误，补充可信来源或重新导出后重试。'


def build_snapshot(store):
    units=[{**v,'key':k} for k,v in store.items('units') if not v.get('retired')]
    existing={u['key'] for u in units}
    # A selected-patient run still lists registered but unprocessed material accurately.
    for key,pid,kind in store.db.execute("SELECT unit_key,pid,kind FROM sources WHERE unit_key!='' GROUP BY unit_key,pid,kind"):
        if key not in existing:
            units.append({'key':key,'pid':pid,'kind':key.split(':',1)[0],
                          'status':'registered','outputs':[],'issues':[]});existing.add(key)
    bypatient=defaultdict(list);issues=[];overview={}
    for u in units:bypatient[u.get('pid','')].append(u)
    raw=dict(store.db.execute('SELECT kind,count(*) FROM sources GROUP BY kind'))
    raw_modalities=dict(store.db.execute("SELECT modality,count(*) FROM sources WHERE kind='DICOM' GROUP BY modality"))
    roster_rows={(src.get('path'),src.get('sheet'),src.get('row')) for r in store.sources(kind='ROSTER')
                 for p in r.get('roster',{}).values() for src in p.get('sources',[])}
    for kind in ['CT','US','PDF']:
        rows=[u for u in units if u['kind']==kind]
        success=[u for u in rows if u['status']=='success']
        failures=[u for u in rows if u['status']=='failed']
        reasons=Counter(next((i['code'] for i in u.get('issues',[]) if i.get('severity','error')!='info'),kind+'_CONVERSION_FAILED') for u in failures)
        overview[kind]={'total':len(rows),'success':len(success),'failed':len(failures),
            'pending':sum(u['status']=='pending' for u in rows),'audited':sum(u['status']=='audited' for u in rows),
            'registered':sum(u['status']=='registered' for u in rows),
            'output_files':sum(len(u.get('outputs',[])) for u in success),
            'update_failed_kept':sum(u.get('last_attempt',{}).get('status')=='failed' for u in success),
            'failed_reasons':dict(reasons),'patients_success':len({u['pid'] for u in success if u.get('pid')})}
    for u in units:
        for i in u.get('last_attempt',u).get('issues',[]):issues.append({**i,'unit':u['key'],'kind':u['kind']})
    issues.extend(store.get('meta','registry_issues',[]))
    # Header failures, roster problems or unsupported files are not discarded from the report.
    issues.extend(store.get('meta','legacy_extra_issues',[]))
    for r in store.sources(kind='ROSTER'):
        for i in r.get('issues',[]):
            patient=store.get('patients',i.get('patient_id',''),{})
            if i['code']=='ROSTER_ID_CONFLICT' and patient.get('confirmed') and not patient.get('conflict'):
                issues.append({**i,'code':'ROSTER_ID_RESOLVED','severity':'info','message':'此前主表冲突已按明确确认规则合并；原记录保留'})
            else:issues.append(i)
    patients=[]
    for pid,p in store.items('patients'):
        rows=bypatient[pid];counts={k:sum(u['kind']==k and u['status']=='success' for u in rows) for k in ['CT','US','PDF']}
        availability={}
        for k in counts:
            specific=[u for u in rows if u['kind']==k]
            if counts[k]:availability[k]='已有成功结果'+('，仍有部分失败/待核实' if any(u['status']!='success' or u.get('last_attempt') for u in specific) else '')
            elif any(u['status']=='pending' for u in specific):availability[k]='已有资料，关联待核实'
            elif any(u['status']=='audited' for u in specific):availability[k]='已检查，尚未转换'
            elif any(u['status']=='registered' for u in specific):availability[k]='已登记，尚未处理'
            elif specific:availability[k]='已有资料，未成功'
            else:
                availability[k]='尚未提供'
                issues.append(issue(k+'_NOT_PROVIDED',f'当前项目尚未登记此患者的{k}资料',pid,severity='missing',kind=k))
        presence=tuple(counts[k]>0 for k in ['CT','US','PDF'])
        patients.append({'id':pid,'name':p.get('name',''),'counts':counts,'category':CATEGORIES[presence],
            'subcategory':subcategory(rows),'availability':availability,'folder':store.get('folders',pid,''),
            'aliases':p.get('aliases',[]),'origin':p.get('origin','legacy')})
    unique={}
    for i in issues:
        i={**i};i.setdefault('severity','error');i.setdefault('patient_id','');i.setdefault('source','');i.setdefault('unit','')
        i['family']=family(i['code']);i['remedy']=remedy(i['code'])
        i['id']=digest([i['patient_id'],i['code'],i['source'],i['unit']])[:24];unique[i['id']]=i
    issues=list(unique.values());previous=store.get('meta','last_snapshot',{})
    before={i['id']:i for i in previous.get('issues',[])}
    active={i['id']:i for i in issues if i['severity']!='info'}
    snapshot={'generated':now(),'patients':patients,'overview':overview,'raw_sources':raw,'raw_modalities':raw_modalities,
        'roster_row_count':len(roster_rows),'issues':issues,
        'category_counts':dict(Counter(p['category'] for p in patients)),
        'roster_patients':sum(p.get('origin') in {'roster','legacy'} for _,p in store.items('patients')),
        'changed':{'new_patients':len(set(p['id'] for p in patients)-set(p['id'] for p in previous.get('patients',[]))),
                   'new_issues':len(set(active)-set(before)),
                   'resolved_issues':[v for k,v in before.items() if k not in unique and v.get('severity')!='info']},
        'processing':store.get('meta','last_processing',{}),'ingest':store.get('meta','last_ingest',{}),
        'previous_overview':previous.get('overview',{}),'manual_recoveries':store.get('meta','legacy_manifest',{}).get('manual_recoveries',[])}
    return snapshot


def atomic_text(path,text):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp-'+uuid.uuid4().hex[:8])
    with temp.open('w',encoding='utf-8-sig',newline='') as f:f.write(text);f.flush();os.fsync(f.fileno())
    temp.replace(path)


def csv_text(rows,columns):
    out=io.StringIO(newline='');w=csv.DictWriter(out,fieldnames=columns);w.writeheader()
    for r in rows:w.writerow({k:(json.dumps(r.get(k,''),ensure_ascii=False) if isinstance(r.get(k), (list,dict)) else r.get(k,'')) for k in columns})
    return out.getvalue()


def render(snapshot,store):
    marker=json.loads((store.root/'project.json').read_text(encoding='utf-8'))
    report_title=str(marker.get('report_title') or ('TAPVC 数据项目总览' if store.root.name.casefold().startswith('tapvc') else 'AAOCA 数据项目总览'))
    e=lambda v:html.escape(str(v),quote=True)
    def table(headers,rows):
        return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+e(h)+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'
    def link(rel,label):
        if not rel:return ''
        from urllib.parse import quote
        url=Path(rel).as_uri() if Path(rel).is_absolute() else quote(rel.replace('\\','/'),safe='/')
        return '<a href="'+e(url)+'">'+e(label)+'</a>'
    cards=''.join(f'<article><b>{value}</b><span>{label}</span></article>' for value,label in [
        (len(snapshot['patients']),'已登记患者编号'),(snapshot['category_counts'].get('01_三类结果齐全',0),'三类结果齐全'),
        (snapshot['overview']['CT']['output_files'],'CT NIfTI 文件'),(snapshot['overview']['US']['success'],'超声成功文件'),
        (snapshot['overview']['PDF']['success'],'已核实 PDF')])
    overview=[]
    overview.append(['患者主表',e(str(snapshot['raw_sources'].get('ROSTER',0))+' 份表；'+str(snapshot.get('roster_row_count',0))+' 条主表记录；'+str(snapshot['roster_patients'])+' 个主表/历史登记编号'),
                     e('关联后共 '+str(len(snapshot['patients']))+' 个编号'),e('缺少主表也可处理；具体编号差异见下表')])
    for k,label in [('CT','CT（原始序列）'),('US','超声（去重后文件）'),('PDF','PDF（内容单元）')]:
        v=snapshot['overview'][k]
        reasons='；'.join(reason_label(code)+' '+str(n) for code,n in v['failed_reasons'].items()) or '无'
        raw_count=snapshot['raw_sources'].get('PDF',0) if k=='PDF' else snapshot.get('raw_modalities',{}).get(k,0)
        quantity=f'{raw_count} 个已识别源文件；{v["total"]} 个处理单元'
        if k=='US' and v['failed_reasons'].get('US_TIMING_INVALID'):
            quantity+='（包含独立记录的帧时间不符合要求的文件）'
        overview.append([e(label),e(quantity),e(f'成功 {v["success"]}；失败 {v["failed"]}；待关联 {v["pending"]}；仅检查 {v["audited"]}；未处理 {v["registered"]}'),e('失败原因：'+reasons+f'。更新失败但保留旧成果 {v["update_failed_kept"]} 项')])
    categories=table(['分类文件夹','患者数','判定依据'],[[e(v),snapshot['category_counts'].get(v,0),e('CT/超声有成功输出，PDF已核实关联')] for v in CATEGORIES.values()])
    patientrows=[]
    for p in snapshot['patients']:
        patientrows.append([e(p['id']),e(p['name']),e(report_text(p['category']+' / '+p['subcategory'])),
            e(p['availability']['CT']),e(p['availability']['US']),e(p['availability']['PDF']),link(p['folder'],'患者目录')])
    issuesections=''
    for group in ['编号关联','数据本身','资料缺失']:
        relevant=[i for i in snapshot['issues'] if i['family']==group and i['severity']!='info']
        grouped=defaultdict(list)
        for i in relevant:grouped[i['code']].append(i)
        summary=[]
        for code,items in sorted(grouped.items()):
            summary.append([e(reason_label(code))+'<br><small>'+e(code)+'</small>',len(items),e('、'.join(sorted({i['patient_id'] for i in items if i['patient_id']})) or '归属待核实'),e(remedy(code))])
        details=table(['患者编号','问题 / 所在环节','详细原因','来源文件 / 单元'],[[e(i['patient_id']) or '待关联',e(i['code']),e(issue_description(i)),e(i.get('source') or i.get('unit'))] for i in relevant])
        issuesections+=f'<section><h2>{e(group)}</h2>'+table(['小类原因','提醒条目数','涉及编号','可能的处理方法'],summary)+f'<details><summary>展开逐文件明细（{len(relevant)} 条）</summary>{details}</details></section>'
    notes=[i for i in snapshot['issues'] if i['severity']=='info']
    us_errors=[i for i in snapshot['issues'] if i.get('kind')=='US' and i['severity']=='error']
    us_scale=[i for i in snapshot['issues'] if i['code'] in {'US_SCALE_UNKNOWN','US_CALIBRATION_REVIEW'} and i['severity']!='info']
    us_missing={i['patient_id'] for i in snapshot['issues'] if i['code']=='US_NOT_PROVIDED'}
    us_details=table(['患者编号','涉及文件','具体原因'],[
        [e(i['patient_id']),e(str(i.get('source') or i.get('unit')).replace('\\','/').rsplit('/',1)[-1]),e(issue_description(i))]
        for i in sorted(us_errors,key=lambda i:(i['patient_id'],i.get('source','')))])
    us_explanation='<section><h2>超声文件：未成功转换与使用限制</h2><p>当前未成功转换 '+str(snapshot['overview']['US']['failed'])+' 个处理单元，具体文件及原因如下。帧时间记录不符合要求时，当前转换器不能按既定规则确认视频时间轴；这不等于已经证明图像像素损坏。像素数据为空则没有可解码的图像内容。</p>'+us_details+'<p>物理标定待核实：'+str(len(us_scale))+' 条提醒，涉及 '+str(len({i['patient_id'] for i in us_scale}))+' 个编号。此提醒本身不等于转换失败；可以查看已成功输出的图像，但用像素距离换算真实组织尺寸前，需要核实区域、单位及比例。提醒条目不与失败文件数相加。</p><p>尚未提供超声资料：'+str(len(us_missing))+' 个编号，属于资料缺失，与已有文件未成功转换分开统计。</p></section>'
    history=table(['数据','上次成功','当前成功','变化'],[[k,snapshot['previous_overview'].get(k,{}).get('success','—'),v['success'],
            v['success']-snapshot['previous_overview'].get(k,{}).get('success',v['success'])] for k,v in snapshot['overview'].items()])
    return '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'''+html.escape(report_title)+'''</title><style>
body{font-family:"Microsoft YaHei",system-ui,sans-serif;background:#f3f6fa;color:#203449;margin:0;line-height:1.6}main{max-width:1480px;margin:auto;padding:32px}h1{margin:0;font-size:30px}h2{font-size:21px;color:#126d81}p{color:#4f6376}.cards{display:flex;gap:14px;flex-wrap:wrap;margin:24px 0}article{background:#123f54;color:white;border-radius:10px;padding:18px 24px;flex:1}article b{display:block;font-size:32px}article span{white-space:nowrap}section{background:white;padding:24px;margin:22px 0;border:1px solid #dbe4ed;border-radius:12px}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}th{background:#e6f0f4;text-align:left}th,td{border:1px solid #dce5ed;padding:10px 12px;vertical-align:top;word-break:break-word}tr:nth-child(even){background:#f8fafc}a{color:#087e9d}summary{cursor:pointer;padding:14px 0;font-weight:600}input{padding:12px;width:min(600px,90%);border:1px solid #abc3d1;border-radius:6px}.note{border-left:4px solid #e5b957;padding:12px;background:#fff8e6}small{color:#607388}@media print{main{padding:0}details{display:block}input{display:none}section{break-inside:avoid}}</style><main>'''+f'''
<h1>{e(report_title)}</h1><p>数据统计时间：{e(snapshot['generated'])} · 患者级关联；不代表同次就诊匹配或临床检查完整。</p>
<div class="cards">{cards}</div><p class="note">三类结果齐全表示有可用 CT、超声和已核实 PDF，并不表示所有序列均成功。PDF 文字提取不足只作备注。失败按原始序列或文件统计，期相输出单独计数。</p>
<section><h2>表 1 · 原始资料与处理情况</h2>{table(['数据类型 / 统计单位','登记数量','成功 / 失败 / 待处理','原因与保留成果'],overview)}<p>累计输入文件：{e(snapshot['raw_sources'])}。完全相同的副本只处理一次；提醒条目数不等于失败序列数或患者数。</p></section>
{us_explanation}
<section><h2>表 2 · 结果分类与患者详情</h2>{categories}<p>“缺”按处理结果判断；下面另列尚未提供、已有但失败、关联待核实。后续补充资料后自动调整分类。</p><input id="search" placeholder="筛选患者编号、姓名、分类或问题"><div id="patients">{table(['编号','姓名','大类 / 小类','CT','超声','PDF','结果'],patientrows)}</div></section>
{issuesections}<section><h2>本次变化与修复记录</h2>{history}<p>本次新增编号 {snapshot['changed']['new_patients']}；新增提醒 {snapshot['changed']['new_issues']}；已解除提醒 {len(snapshot['changed']['resolved_issues'])}。</p><p>{link('项目数据快照.json','完整快照')} · {link('患者分类表.csv','患者分类表')} · {link('详细问题表.csv','详细问题表')} · {link('处理单元表.csv','各序列/文件输出及修复证据')}</p><p>原始资料未改写。已确认的历史修复 {len(snapshot['manual_recoveries'])} 项保留。失败更新不会覆盖旧的成功成果。</p></section>
<section><h2>备注（不计错误）</h2><details><summary>{len(notes)} 条备注，包括低文字页与重复副本说明</summary>{table(['编号','备注','原因'],[[e(i['patient_id']),e(i['code']),e(issue_description(i))] for i in notes])}</details></section>
<small>报告含患者资料，请保存在授权位置。原始资料目录离线时仍可查看报告；新转换只依赖本次涉及的源文件。</small></main>
<script>document.querySelector('#search').addEventListener('input',e=>{{let q=e.target.value.toLowerCase();document.querySelectorAll('#patients tbody tr').forEach(r=>r.hidden=!r.textContent.toLowerCase().includes(q));}});</script></html>'''


def publish_project(store,snapshot):
    classify(store,snapshot['patients'])
    # Paths may have moved; snapshot metrics remain the same, output rows use updated registry.
    units=[dict(key=k,**{a:b for a,b in u.items() if a!='key'}) for k,u in store.items('units') if not u.get('retired')]
    if not store.get('meta','external_legacy_root'):
        bypatient=defaultdict(list)
        for u in units:bypatient[u.get('pid','')].append(u)
        for p in snapshot['patients']:
            folder=store.path(p['folder']);links=[]
            from urllib.parse import quote
            for u in bypatient[p['id']]:
                for rel in u.get('outputs',[]):
                    path=store.path(rel);url=quote(os.path.relpath(path,folder).replace('\\','/'),safe='/')
                    links.append('<li><a href="'+html.escape(url,quote=True)+'">'+html.escape(u['kind']+' · '+path.name)+'</a> · '+html.escape(u['status'])+'</li>')
            content='<meta charset="utf-8"><title>患者资料索引</title><h1>'+html.escape(p['id']+' '+p['name'])+'</h1><p>'+html.escape(report_text(p['category']+' / '+p['subcategory']))+'</p><ul>'+''.join(links)+'</ul>'
            atomic_text(folder/'资料索引.html',content)
    batch=store.get('meta','current_batch',uuid.uuid4().hex[:12])
    history=store.root/'report_history'/batch;history.mkdir(parents=True,exist_ok=True)
    write_json(history/'snapshot.json',snapshot)
    write_json(store.root/'项目数据快照.json',snapshot)
    atomic_text(store.root/'患者分类表.csv',csv_text(snapshot['patients'],['id','name','category','subcategory','counts','availability','folder','aliases']))
    atomic_text(store.root/'详细问题表.csv',csv_text(snapshot['issues'],['family','patient_id','severity','code','message','source','unit','remedy']))
    atomic_text(store.root/'处理单元表.csv',csv_text(units,['key','pid','kind','status','source_count','unique_files','slices','frames','outputs','error','last_attempt','repairs']))
    page=render(snapshot,store)
    atomic_text(store.root/'查看结果.html',page)
    # Historical HTML links are relative to the project root.
    atomic_text(history/'report.html',page.replace('<meta charset="utf-8">','<meta charset="utf-8"><base href="../../">',1))
    store.put('meta','last_snapshot',snapshot)
    store.put('snapshots',batch,{'time':snapshot['generated'],'path':str(history.relative_to(store.root))})
    return store.root/'查看结果.html'
