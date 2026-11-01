"""Freeze a private input descriptor from audited data and the prior OOF split."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'05_src'))
from imagechd_cohort import PROJECT_ID, file_hash, make_splits
from imagechd_excel import read_spacing_excel


def build(raw, models, audit, output):
    raw, models, audit, output = map(Path,(raw,models,audit,output))
    if output.exists():
        raise FileExistsError('输入描述目录必须是新目录；禁止覆盖冻结描述')
    model_record=json.loads(models.read_text(encoding='utf-8-sig'))
    qc=json.loads(audit.read_text(encoding='utf-8-sig'))
    spacing_path=raw/'imagechd_dataset_image_info.xlsx'
    spacings=read_spacing_excel(spacing_path)
    folds=model_record['split_plan']['folds']
    if [f['index'] for f in folds]!=[0,1,2]:
        raise ValueError('需要原实验完整三折划分')
    roles={}
    for fold in folds:
        for part in ('train','val','test'):
            for cid in fold['cases'][part]:
                if str(fold['index']) in roles.setdefault(cid,{}):
                    raise ValueError('同折病例重复')
                roles[cid][str(fold['index'])]=part
    cases=[]
    for row in qc['cases']:
        cid=row['case_id'];image=raw/(cid+'_image.nii.gz');label=raw/(cid+'_label.nii.gz')
        source_hash=file_hash(image)
        # Audit and frozen model are evidence inputs, not reconstructed HU.
        frozen_hash=row['source_sha256']
        expected=frozen_hash.get('image',frozen_hash.get('image_sha256')) if isinstance(frozen_hash,dict) else frozen_hash
        if expected is not None and source_hash!=expected:
            raise ValueError('源图与QC记录SHA256不符: '+cid)
        if row['included'] and (set(roles.get(cid,{}))!={'0','1','2'} or not set(range(8))<=set(row['label_ids'])):
            raise ValueError('纳入病例缺七结构标注或三折角色')
        cases.append(dict(case_id=cid,group='ImageCHD',modality='CT',patient_id=None,
            image=str(image.resolve()),label=str(label.resolve()),image_sha256=source_hash,
            status='eligible' if row['included'] else 'quarantined',
            exclusion='missing_MYO_annotation' if not row['included'] else None,
            source_qc=['extreme_class_constant_intensity'] if cid=='ct_1170' else [],
            spacing_mm=spacings[cid]['spacing_mm'],spacing_source='imagechd_author_excel',
            spacing_excel_row=spacings[cid]['excel_row'],extra_label_ids=[x for x in row['label_ids'] if x>7],roles=roles.get(cid,{})))
    make_splits(cases)
    record=dict(protocol=PROJECT_ID,patient_identity_verified=False,cases=cases,
        source_provenance={p.name:file_hash(p) for p in (models,audit,spacing_path)})
    output.mkdir(parents=True)
    (output/'imagechd_input.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'source_pairs':len(cases),'eligible':sum(c['status']=='eligible' for c in cases),'patient_identity_verified':False}))
    return record


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('raw','models','audit','output'):p.add_argument('--'+key,required=True)
    args=p.parse_args();build(args.raw,args.models,args.audit,args.output)
