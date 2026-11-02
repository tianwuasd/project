"""Local-only descriptive report. Every table keeps its denominator and missing values."""
from collections import Counter,defaultdict
import csv
import html
import json
import math
import os
import time
import uuid
from pathlib import Path
from .contracts import atomic_json,read_json,sha256_file


def write_csv(path,rows,fieldnames=None):
    rows=list(rows)
    fields=fieldnames or list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for row in rows:
            clean={}
            for key,value in row.items():
                if isinstance(value,(list,dict)): value=json.dumps(value,ensure_ascii=False)
                if isinstance(value,str) and value.startswith(('=','+','-','@','\t','\r')): value="'"+value
                clean[key]='' if value is None else value
            writer.writerow(clean)


def aggregate_results(results,expected_total,stage='full'):
    cases=[];labels=[];issues=[];origins=Counter();image_types=Counter();label_types=Counter();shapes=Counter()
    classes=defaultdict(lambda:{'cases_present':0,'voxels':0,'touching_boundary_cases':0,'component_total':0})
    for result in results:
        image=result.get('image',{});label=result.get('label',{});g=result.get('geometry',{})
        row={'case_id':result['case_id'],'source_key':result.get('source_key'),'collection':result.get('collection'),
             'status':result['status'],'metadata_link_status':result.get('metadata_link_status'),
             'readability_status':result.get('readability_status'),
             'array_geometry_status':g.get('array_geometry_status'),'physical_geometry_status':g.get('physical_geometry_status'),
             'label_semantics_status':g.get('label_semantics_status','unverified_numeric_labels'),
             'shape':g.get('image',{}).get('shape'),'image_dtype':g.get('image',{}).get('dtype'),
             'label_dtype':g.get('label',{}).get('dtype'),'spacing_header':g.get('image',{}).get('spacing'),
             'units_header':g.get('image',{}).get('units'),'origin_delta_header_units':g.get('origin_delta_header_units'),
             'image_min':image.get('min'),'image_max':image.get('max'),'image_mean':image.get('mean'),
             'image_std':image.get('std'),'image_nonfinite_count':image.get('nonfinite_count'),
             'image_percentiles_approximate':image.get('percentiles_approximate'),
             'label_nonfinite_count':label.get('nonfinite_count'),'label_noninteger_count':label.get('noninteger_count'),
             'foreground_voxels':label.get('foreground_voxels'),
             'foreground_fraction':label.get('foreground_voxels',0)/label['voxel_count'] if label.get('voxel_count') else None,
             'label_values':[c['value'] for c in label.get('classes',[])],
             'seconds':result.get('seconds'),'source_hashes_match_manifest':result.get('source_expected_hash_match')}
        row.update({'image_p'+key:value for key,value in image.get('percentiles',{}).items()})
        cases.append(row)
        if g:
            origins[json.dumps(g.get('origin_delta_header_units'))]+=1
            if g.get('image'):
                image_types[g['image']['dtype']]+=1;shapes[str(g['image']['shape'])]+=1
            if g.get('label'): label_types[g['label']['dtype']]+=1
        def issue(code,detail): issues.append({'case_id':result['case_id'],'code':code,'detail':detail,'action':'review only; input retained'})
        for error in result.get('errors',[]): issue('read_failure',error)
        if result.get('metadata_link_status')!='matched': issue('metadata_link_unresolved',result.get('metadata_link_status'))
        if g.get('physical_geometry_status')=='unverified': issue('physical_geometry_unverified','Unknown units and/or header disagreement; no automatic repair')
        if g and not g.get('header_affine_match',True): issue('header_affine_mismatch',g.get('origin_delta_header_units'))
        for conflict in g.get('metadata_shape_conflicts',[]): issue('metadata_shape_conflict',conflict)
        if image.get('nonfinite_count',0): issue('nonfinite_image_voxels',image['nonfinite_count'])
        if image.get('constant'): issue('constant_image','All finite intensities are equal')
        if label.get('noninteger_count',0): issue('noninteger_label_values',label['noninteger_count'])
        if label.get('nonfinite_count',0): issue('nonfinite_label_voxels',label['nonfinite_count'])
        if label.get('empty'): issue('empty_label','No nonzero finite label voxels')
        if label.get('per_class_statistics_status')=='not_computed_cardinality_limit':
            issue('label_cardinality_exceeded','Per-class statistics not computed; complete read, invalid-value counts and hash retained')
        if False in result.get('source_expected_hash_match',{}).values(): issue('historical_digest_mismatch','Current bytes differ from manifest evidence')
        for item in label.get('classes',[]):
            labels.append({'case_id':result['case_id'],**item})
            aggregate=classes[str(item['value'])];aggregate['cases_present']+=1;aggregate['voxels']+=item['voxels']
            aggregate['touching_boundary_cases']+=bool(item.get('touches_boundary'))
            aggregate['component_total']+=item.get('components') or 0
    denominators={'expected_pairs':expected_total,'terminal_results':len(results),'pending':max(0,expected_total-len(results)),
                  'completed_cases':sum(r['status']=='complete' for r in results),'failed_cases':sum(r['status']!='complete' for r in results),
                  'headers_available':sum('geometry' in r for r in results),
                  'image_statistics_available':sum('image' in r for r in results),
                  'label_statistics_available':sum('label' in r for r in results),
                  'full_current_file_hashes':sum(len(r.get('source_sha256',{})) for r in results)}
    denominators['voxel_analysis_not_requested']=expected_total if stage!='full' else 0
    denominators['metadata_only_completed_pairs']=expected_total if stage=='metadata' else 0
    if stage=='metadata': denominators['pending']=0
    return {'denominators':denominators,'case_rows':cases,'label_rows':labels,'issues':issues,
            'issue_counts':dict(Counter(i['code'] for i in issues)), 'label_summary':dict(classes),
            'image_dtypes':dict(image_types),'label_dtypes':dict(label_types),'origin_deltas_header_units':dict(origins),'shapes':dict(shapes)}


def _plots(output,summary,metadata):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    rows=summary['case_rows']
    arrays=[([r.get('age_at_acquisition_years') for r in metadata],'Age at acquisition (years)'),
            ([r['shape'][2] if r.get('shape') and len(r['shape'])>2 else None for r in rows],'Image third dimension (voxels)'),
            ([r['image_mean'] for r in rows],'Mean proxy-scaled intensity (HU unverified)'),
            ([r['foreground_fraction'] for r in rows],'Nonzero label voxel fraction')]
    for ax,(values,title) in zip(axes.flat,arrays):
        clean=[v for v in values if isinstance(v,(int,float)) and math.isfinite(v)]
        ax.hist(clean,bins=min(30,max(1,len(clean))),color='#346e91',edgecolor='white');ax.set_title(title,fontsize=10)
        ax.set_ylabel('Records / image pairs');ax.text(.98,.95,f'n={len(clean)}',ha='right',va='top',transform=ax.transAxes)
    models=Counter(str(r.get('ManufacturerModelName','missing')) for r in metadata)
    axes[1,1].barh(list(models),list(models.values()),color='#609f93');axes[1,1].set_title('Scanner model: Excel records',fontsize=10)
    occurrence=summary['label_summary'];keys=sorted(occurrence,key=float)
    axes[1,2].bar(keys,[occurrence[k]['cases_present'] for k in keys],color='#a88857')
    axes[1,2].set_title('Numeric label occurrence; anatomy unverified',fontsize=10);axes[1,2].set_ylabel('Readable labels')
    path=output/'overview.png';fig.savefig(path,dpi=150);plt.close(fig)
    return path


def _choose_examples(results,metadata,links,limit):
    valid=[r for r in results if r.get('status')=='complete' and 'image' in r and 'label' in r and 'geometry' in r]
    by_id={r['case_id']:r for r in valid};chosen=[];reasons=defaultdict(list)
    def add(row,reason):
        if row['case_id'] not in chosen and len(chosen)<limit: chosen.append(row['case_id'])
        if row['case_id'] in chosen: reasons[row['case_id']].append(reason)
    for kind in ('image','label'):
        groups=defaultdict(list)
        for row in valid: groups[row['geometry'][kind]['dtype']].append(row)
        for dtype,group in sorted(groups.items(),key=lambda kv:len(kv[1])):
            for row in sorted(group,key=lambda r:r['case_id'])[:3]: add(row,kind+' dtype '+dtype)
    for row in valid:
        if row['image'].get('mean') is None: add(row,'no finite image values; rendering skipped explicitly')
    for collection in sorted(set(r.get('collection','') for r in valid)):
        add(next(r for r in valid if r.get('collection')==collection),'collection '+collection)
    for row in valid:
        if row['geometry']['metadata_shape_conflicts']: add(row,'Excel shape conflict')
        if row.get('metadata_link_status')!='matched': add(row,'metadata link unresolved')
    matched={l['case_id']:metadata[l['excel_index']] for l in links['case_rows'] if l['excel_index'] is not None}
    models=set()
    for row in valid:
        model=matched.get(row['case_id'],{}).get('ManufacturerModelName')
        if model not in models: add(row,'scanner '+str(model));models.add(model)
    patient_counts=Counter(m.get('PatientID') for m in matched.values() if m.get('PatientID'))
    for row in valid:
        patient=matched.get(row['case_id'],{}).get('PatientID')
        if patient and patient_counts[patient]>1: add(row,'repeated PatientID record; no automatic merge')
    for key in ('mean','std','min','max'):
        group=sorted([r for r in valid if isinstance(r['image'].get(key),(int,float)) and math.isfinite(r['image'][key])],
                     key=lambda r:(r['image'][key],r['case_id']))
        if group: add(group[0],'minimum '+key);add(group[-1],'maximum '+key)
    for row in sorted(valid,key=lambda r:r['case_id']):
        if len(chosen)>=limit: break
        add(row,'deterministic remaining example')
    return [(by_id[c],reasons[c]) for c in chosen]


def _qc_examples(output,results,metadata,links,limit):
    import numpy as np
    import nibabel as nib
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from .voxels import NiftiStream
    target=output/'qc_examples';target.mkdir(exist_ok=True)
    records=[]
    for row,reasons in _choose_examples(results,metadata,links,limit):
        if row['image'].get('mean') is None:
            records.append({'case_id':row['case_id'],'status':'skipped_no_finite_image','reasons':reasons});continue
        if row['label'].get('per_class_statistics_status')=='not_computed_cardinality_limit':
            records.append({'case_id':row['case_id'],'status':'skipped_label_cardinality_limit','reasons':reasons});continue
        # Stream the image once into three plane buffers; do not load another full float64 volume.
        source=NiftiStream(row['input_paths']['image'],8);shape=source.shape;center=[s//2 for s in shape]
        planes=[np.empty((shape[1],shape[2])),np.empty((shape[0],shape[2])),None]
        for start,block in source.blocks():
            stop=start+block.shape[2];planes[0][:,start:stop]=block[center[0],:,:];planes[1][:,start:stop]=block[:,center[1],:]
            if start<=center[2]<stop: planes[2]=np.array(block[:,:,center[2]-start])
        if source.sha256!=row['source_sha256']['image']: raise ValueError('Input changed before QC rendering')
        label_source=NiftiStream(row['input_paths']['label'],8)
        if label_source.shape!=shape:
            records.append({'case_id':row['case_id'],'status':'skipped_shape_mismatch','reasons':reasons});continue
        labels=[np.empty((shape[1],shape[2])),np.empty((shape[0],shape[2])),None]
        for start,block in label_source.blocks():
            stop=start+block.shape[2]
            labels[0][:,start:stop]=block[center[0],:,:];labels[1][:,start:stop]=block[:,center[1],:]
            if start<=center[2]<stop: labels[2]=np.array(block[:,:,center[2]-start])
        if label_source.sha256!=row['source_sha256']['label']: raise ValueError('Label changed before QC rendering')
        lo=row['image']['percentiles'].get('1');hi=row['image']['percentiles'].get('99')
        if lo is None or hi is None or hi<=lo: lo,hi=row['image']['min'],row['image']['max']
        fig,axes=plt.subplots(1,3,figsize=(13,5),layout='constrained')
        values=[c['value'] for c in row['label']['classes']]
        label_min=min([0.,*values]);label_max=max([1.,*values])
        for i,ax in enumerate(axes):
            ax.imshow(planes[i].T,cmap='gray',origin='lower',vmin=lo,vmax=hi,aspect='auto')
            ax.imshow(np.ma.masked_where(labels[i].T==0,labels[i].T),cmap='tab20',vmin=label_min,vmax=label_max,
                      origin='lower',alpha=.42,interpolation='nearest',aspect='auto')
            ax.set_title(f'Array axis {i} = {center[i]}');ax.set_xlabel('voxel index');ax.set_ylabel('voxel index')
        fig.suptitle(row['case_id']+' | ARRAY INDEX overlay only; physical alignment and anatomy unverified',fontsize=10)
        path=target/(row['case_id']+'.png');fig.savefig(path,dpi=120);plt.close(fig)
        records.append({'case_id':row['case_id'],'status':'rendered','file':path.relative_to(output).as_posix(),
                        'reasons':reasons,'indices':center,'header_origin_delta':row['geometry']['origin_delta_header_units']})
        del planes,labels
    atomic_json(output/'qc_selection.json',records)
    return records


def _build_report(run_dir,output,render_examples=True):
    run_dir=Path(run_dir);provenance=read_json(run_dir/'provenance.json');config=read_json(run_dir/'resolved_config.json')
    expected=provenance['basis']['case_ids'];results=[]
    for case_id in expected:
        path=run_dir/'cases'/(case_id+'.json');marker=path.with_suffix('.complete.json')
        if not path.exists() or not marker.exists(): continue
        evidence=read_json(marker)
        if evidence.get('result_sha256')!=sha256_file(path) or evidence.get('contract')!=provenance['contract']:
            raise ValueError('Case result commit validation failed')
        results.append(read_json(path))
    stage=provenance['basis']['stage']
    summary=aggregate_results(results,len(expected),stage=stage)
    metadata=read_json(run_dir/'metadata_records.json');links=read_json(run_dir/'metadata_links.json')
    metadata_summary=read_json(run_dir/'metadata_summary.json')
    summary['metadata']={'excel_records':len(metadata),'strict_matches':links['matched_count'],
                         'unmatched_excel_records':len(links['unmatched_excel_indices']),
                         'ambiguous_excel_records':len(links['ambiguous_excel_indices']),
                         'distinct_patient_ids':metadata_summary['distinct_patient_ids'],
                         'duplicate_id_groups':len(metadata_summary['duplicate_id_groups'])}
    summary['run_id']=run_dir.name;summary['analysis_contract']=provenance['contract']
    summary['physical_units']='unverified; no mm/mL or HU conclusions';summary['clinical_label_names']='not established'
    from .metadata import describe
    summary['numeric_distributions']={key:describe([r.get(key) for r in summary['case_rows']])
        for key in ('image_min','image_max','image_mean','image_std','foreground_voxels','foreground_fraction')}
    summary['slice_count_distribution']=describe([r['shape'][2] for r in summary['case_rows'] if r.get('shape') and len(r['shape'])>2])
    summary['header_spatial_units']=dict(Counter(r['geometry']['image']['units'][0] for r in results if 'geometry' in r))
    summary['metadata_categories']=metadata_summary['categories']
    summary['metadata_numeric']=metadata_summary['numeric']
    summary['stage']=stage
    write_csv(output/'cases.csv',summary['case_rows']);write_csv(output/'label_classes.csv',summary['label_rows'])
    write_csv(output/'issues.csv',summary['issues'],['case_id','code','detail','action'])
    write_csv(output/'metadata_records.csv',metadata);write_csv(output/'metadata_links.csv',links['case_rows'])
    write_csv(output/'unmatched_excel.csv',[metadata[i] for i in links['unmatched_excel_indices']],list(metadata[0]) if metadata else [])
    write_csv(output/'metadata_categories.csv',[{'field':key,'value':value,'records':count,'denominator_excel_records':len(metadata)}
                for key,categories in metadata_summary['categories'].items() for value,count in categories.items()])
    write_csv(output/'numeric_summary.csv',[{'source':'Excel','field':key,**value} for key,value in metadata_summary['numeric'].items()]+
              [{'source':'NIfTI cases','field':key,**value} for key,value in summary['numeric_distributions'].items()])
    write_csv(output/'slice_statistics.csv',[{'case_id':r['case_id'],**sl} for r in results for sl in r.get('image',{}).get('slices',[])])
    write_csv(output/'label_summary.csv',[{'numeric_label':k,**v,'denominator_readable_labels':summary['denominators']['label_statistics_available']}
                                        for k,v in summary['label_summary'].items()])
    atomic_json(output/'summary.json',{k:v for k,v in summary.items() if k not in ('case_rows','label_rows','issues')})
    _plots(output,summary,metadata)
    examples=_qc_examples(output,results,metadata,links,config['visual_examples']) if render_examples else []
    den=summary['denominators'];meta=summary['metadata'];escape=html.escape
    text=f'''# dataset1282 本地数据检查与分析报告

运行：{run_dir.name}；分析协议：{config['protocol']}。

本次请求范围：{stage}；未请求体素分析的文件对：{den['voxel_analysis_not_requested']}。仅元数据运行以metadata_only_completed_pairs记录完成范围，不把未请求的影像计算列为待完成。

本报告只描述本次文件读取与统计结果。原始影像、标签和Excel均未修改；没有执行自动修复、重采样、排除病例或训练。

## 完成情况与分母

- 清单研究文件对：{den['expected_pairs']}；已生成终态：{den['terminal_results']}；待完成：{den['pending']}。
- 完成分析：{den['completed_cases']}；读取或计算失败：{den['failed_cases']}。
- 影像统计可用：{den['image_statistics_available']}；标签统计可用：{den['label_statistics_available']}；本轮完整文件摘要：{den['full_current_file_hashes']}。
- Excel记录：{meta['excel_records']}；严格对应：{meta['strict_matches']}；未匹配Excel：{meta['unmatched_excel_records']}；关联歧义：{meta['ambiguous_excel_records']}。
- 不同PatientID：{meta['distinct_patient_ids']}；重复ID组：{meta['duplicate_id_groups']}。此数字不等于已核实的独立患者数，未自动合并记录。

## 描述性统计

- 影像类型：{json.dumps(summary['image_dtypes'],ensure_ascii=False)}。
- 标签类型：{json.dumps(summary['label_dtypes'],ensure_ascii=False)}。
- 头空间单位：{json.dumps(summary['header_spatial_units'],ensure_ascii=False)}。
- NIfTI第三维切片数：{json.dumps(summary['slice_count_distribution'],ensure_ascii=False)}。
- 性别按Excel记录统计：{json.dumps(metadata_summary['categories']['PatientSex'],ensure_ascii=False)}。
- 扫描设备按Excel记录统计：{json.dumps(metadata_summary['categories']['ManufacturerModelName'],ensure_ascii=False)}。
- 采集时年龄（年）：{json.dumps(metadata_summary['numeric']['age_at_acquisition_years'],ensure_ascii=False)}。
- 标签减影像的原点差（header units）：{json.dumps(summary['origin_deltas_header_units'],ensure_ascii=False)}。

以上数值的valid_count为有效记录分母，missing_or_invalid保留缺失或无效记录数量。不同病例图像均值的分布属于病例级描述，不是把全部数据体素合在一起的均值。来源间距、层厚与其他数量的范围/中位数见numeric_summary.csv；各标签出现次数及体素数见label_summary.csv。

## 当前证据与解释范围

图像全体素计数、有限值最值、均值、总体标准差和逐切片统计使用全部体素。分位数取固定线性下标的至多{config['percentile_samples']:,}个样本，属于近似值，不能当作精确分位数。

当前影像灰度按NIfTI代理的slope/intercept缩放；HU尚无来源证据。头间距和Excel记录间距分别呈现，未回填或替换。空间单位未证实的记录不输出毫米距离、毫升体积或真实视野结论。数组尺寸一致不证明空间配准正确。

标签保留原数值，未将编号指定为心脏解剖名称。连通域使用{config['connectivity']}阶邻接（1阶为3D六邻域），包围盒和触边按数组下标描述。多连通域、触边或文件名class分组都不直接解释为疾病或错误标签。

## 需复核的项目

'''
    for key,value in sorted(summary['issue_counts'].items()): text+=f'- {key}：{value}条问题记录。\n'
    text+='''
每个文件对的关联、可读性、数组网格、物理空间和标签语义分别记录。一个病例可有多个问题；问题条数不能相加作为异常病例数。流程完成不表示所有数据质量通过。读取失败不会删除原数据，也不会改变未来纳入决定。

## 交付文件

cases.csv 为病例级汇总；metadata_records.csv 保留Excel字段；metadata_links.csv 与 unmatched_excel.csv 保留关联去向；issues.csv 记录复核事项；label_classes.csv 和 label_summary.csv 给出标签统计；slice_statistics.csv 为逐切片统计。

overview.png 展示主要分布。qc_examples/ 是按固定规则选取的三平面数组叠加图，选取理由和下标见qc_selection.json。显示位置没有应用空间修复，不代表物理配准已通过；少量切面不能替代逐例人工核查。

## 后续所需证据

需补充原始转换流程中的空间标定与灰度保真证据、标签编号对应表，并人工核实未匹配记录及切片计数差异。重复PatientID在后续划分训练/验证数据前需核实，避免同一患者跨集合；本轮不制定训练划分。

## 方法参照

- [NiBabel NIfTI header / qform / sform / scaling](https://nipy.org/nibabel/nifti_images.html)：用于区分头信息与缩放后数值。
- [NiBabel images and memory](https://nipy.org/nibabel/images_and_memory.html)：参考代理读取与内存缓存边界，本工程使用顺序分块读取。
- [SciPy ndimage.label](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.label.html)：连接关系显式指定，每个非零数值类分别统计。
- [MRIQC metrics](https://mriqc.readthedocs.io/en/latest/measures.html)：只借鉴逐例指标与报告组织方式，未将MRI专属指标或阈值应用于CT。

所有真实病例表、图片和本报告仅保存在本地，不属于GitHub代码发布内容。
'''
    (output/'report.md').write_text(text,encoding='utf-8')
    def table(mapping):
        return '<table><tbody>'+''.join('<tr><th>'+escape(str(k))+'</th><td>'+escape(str(v))+'</td></tr>' for k,v in mapping.items())+'</tbody></table>'
    cards=''.join('<figure><img loading="lazy" src="'+escape(r['file'],quote=True)+'"><figcaption>'+escape(r['case_id']+' | '+', '.join(r['reasons']))+'</figcaption></figure>' for r in examples if r['status']=='rendered')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>dataset1282 本地分析报告</title><style>body{font:16px/1.7 system-ui,sans-serif;max-width:1200px;margin:35px auto;padding:0 24px;color:#20313d}h1{color:#205977}table{border-collapse:collapse;width:100%;margin:20px 0}td,th{border:1px solid #ced9df;text-align:left;padding:8px}th{background:#edf3f6}img{width:100%;height:auto}pre{white-space:pre-wrap;font:inherit;background:#f4f7f8;padding:24px}figure{margin:24px 0}a{color:#216990}</style><h1>dataset1282 本地数据检查与分析</h1>'''
    page+='<p>只读审计 · 未修改原始数据 · 流程完成与数据质量结论分别报告</p>'+table(den)+table(meta)
    page+='<img src="overview.png" alt="数据分布概览"><h2>问题类型</h2>'+table(summary['issue_counts'])
    page+='<p><a href="report.md">完整Markdown报告</a> · <a href="cases.csv">病例表</a> · <a href="issues.csv">问题表</a> · <a href="metadata_links.csv">关联表</a> · <a href="summary.json">汇总JSON</a></p>'
    page+='<h2>解释与方法</h2><pre>'+escape(text)+'</pre><h2>代表病例数组切面</h2>'+cards+'</html>'
    (output/'report.html').write_text(page,encoding='utf-8')
    return {'report':str(output/'report.html'),'markdown':str(output/'report.md'),'summary':str(output/'summary.json'),
            'denominators':den,'qc_examples':len(examples)}


def build_report(run_dir,render_examples=True,home=None):
    """Publish an immutable complete report version; failures leave the previous pointer intact."""
    from .resources import snapshot,budget,admit_case
    from .observability.status import operation_lock
    from .engine.runner import code_identity
    run_dir=Path(run_dir)
    config=read_json(run_dir/'resolved_config.json');provenance=read_json(run_dir/'provenance.json')
    code_root=Path(__file__).resolve().parents[2]
    home=Path(home or os.environ.get('DATASET1282_HOME',provenance['code_source']))
    if home.parent.name=='code_versions': home=home.parent.parent
    with operation_lock(home,'report',run_dir):
        facts=snapshot();applied=budget(facts,config)
        json_bytes=sum(p.stat().st_size for p in (run_dir/'cases').glob('*.json'))
        # Parsed JSON/dicts + CSV aggregation + plotting; QC itself uses only planes and slabs.
        estimate=json_bytes*10+768*1024**2
        decision=admit_case(applied,applied,[],estimate)
        if not decision['allowed']: raise MemoryError('Report resource admission refused: '+decision['reason'])
        version=time.strftime('%Y%m%dT%H%M%S',time.gmtime())+'-'+uuid.uuid4().hex[:8]
        output=run_dir/'report_versions'/version;output.mkdir(parents=True)
        atomic_json(output/'report-provenance.json',{'analysis_contract':provenance['contract'],
                    'report_code_files':code_identity(code_root),'resources':applied,'estimated_bytes':estimate})
        result=_build_report(run_dir,output,render_examples)
        pointer={'version':version,'report':str(Path(result['report']).relative_to(run_dir)),
                 'summary':str(Path(result['summary']).relative_to(run_dir)),'analysis_contract':provenance['contract']}
        atomic_json(run_dir/'report-current.json',pointer)
        stable=run_dir/'report';stable.mkdir(exist_ok=True)
        relative='../'+Path(result['report']).relative_to(run_dir).as_posix()
        landing='<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0;url='+html.escape(relative,quote=True)+'"><a href="'+html.escape(relative,quote=True)+'">打开已完成的分析报告</a>'
        temporary=stable/('.report-'+uuid.uuid4().hex+'.tmp')
        temporary.write_text(landing,encoding='utf-8');os.replace(temporary,stable/'report.html')
        result['stable_report']=str(stable/'report.html')
        return result
