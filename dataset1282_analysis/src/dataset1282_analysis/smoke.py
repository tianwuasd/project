import argparse
import json
import os
from pathlib import Path
from .config import resolve_config
from .contracts import atomic_json,sha256_file
from .short_workspace import ShortWorkspace


def synthetic_fixture(root):
    import numpy as np
    import nibabel as nib
    import openpyxl
    from .metadata import FIELDS
    root=Path(root);data=root/'CHD';rows=[]
    workbook=openpyxl.Workbook();sheet=workbook.active;sheet.title='dicom2nii_info';sheet.append(FIELDS)
    for index in range(3):
        name=f'SYN_{index:03d}'; group='native' if index==0 else 'review_required'
        image=data/group/'images'/(name+'_0000.nii.gz');label=data/group/'labels'/(name+'.nii.gz')
        image.parent.mkdir(parents=True,exist_ok=True);label.parent.mkdir(parents=True,exist_ok=True)
        x=np.arange(12*14*8,dtype=np.int16).reshape(12,14,8)
        y=np.zeros_like(x,dtype=np.int8);y[3:8,4:10,2:6]=1;y[0,0,0]=2
        a=np.eye(4);b=np.eye(4);b[:3,3]=index
        nib.save(nib.Nifti1Image(x,a),image);nib.save(nib.Nifti1Image(y,b),label)
        if index==2: image.write_bytes(image.read_bytes()[:-30])
        key='synthetic_source_'+str(index)
        rows.append({'case_id':name,'source_key':key,'collection':group,
                     'outputs':[{'path':str(p),'sha256':sha256_file(p)} for p in (image,label)]})
        record=dict(idx=key if index!=2 else 'unmatched_synthetic',PatientName='Synthetic only',PatientID='1234567890123456' if index<2 else '0002',
                    PatientBirthDate='20000101',PatientSex='F',AcquisitionDate='20200101',ManufacturerModelName='Synthetic scanner',
                    PixelSpacingX=.5,PixelSpacingY=.5,SliceThickness=1,Rows=12,Cols=14,all_file_z=8,all_read_file_z=8,valid_z=8,
                    calc_zSpacing=1,dicom_folder='synthetic')
        sheet.append([record.get(k) for k in FIELDS])
    atomic_json(data/'manifest.json',rows)
    excel=root/'metadata.xlsx';workbook.save(excel)
    return data,excel


def main(argv=None):
    p=argparse.ArgumentParser(description='Logs-only disposable smoke profiles')
    p.add_argument('--profile',choices=['synthetic','quick','capacity'],default='synthetic')
    p.add_argument('--run-id')
    args=p.parse_args(argv)
    root=Path(os.environ['DATASET1282_CODE_ROOT']);home=Path(os.environ['DATASET1282_HOME'])
    config=resolve_config({},home/'project.local.json',root)
    from .engine.runner import run_analysis
    protected=[value for value in (config['data_root'],config['metadata_excel']) if value]
    with ShortWorkspace(Path(config['work_root'])/'smoke',protected_paths=protected,
                        prefix=args.profile) as workspace:
        workspace.record('configuration',profile=args.profile,config=config,
                         input_identities={str(value):sha256_file(value) for value in protected
                                           if Path(value).is_file()})
        if config['data_root']:
            manifest=Path(config['data_root'])/'manifest.json'
            workspace.record('input_manifest',path=manifest,sha256=sha256_file(manifest))
        result=run_profile(args,config,root,home,workspace,run_analysis)
        workspace.record('result',**result)
        if not result['ok']:
            raise RuntimeError(f'Smoke acceptance failed; see {workspace.logs}')
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0


def run_profile(args,config,root,home,workspace,run_analysis):
    config=dict(config,work_root=str(workspace.scratch/'work'))
    home=workspace.scratch/'home'
    home.mkdir()
    if args.profile=='synthetic':
        data,excel=synthetic_fixture(workspace.scratch/'input')
        config.update(data_root=str(data),metadata_excel=str(excel))
        workspace.record('synthetic_configuration',config=config)
        result=run_analysis(config,root,home,run_id='synthetic')
        resumed=run_analysis(config,root,home,run_id='synthetic',resume=True)
        ok=result['complete']==2 and result['failed']==1 and resumed['reused']==2
        result={'ok':ok,'first':result,'resumed':resumed,'expected_failure':'deliberately truncated synthetic gzip'}
    else:
        import nibabel as nib
        from .inputs import load_cases
        cases=load_cases(config)
        chosen={}
        if args.profile=='quick':
            for case in cases:
                chosen.setdefault(case.collection,case.case_id)
        else:
            maximum=(0,None)
            for case in cases:
                image=nib.load(case.image)
                chosen.setdefault(str(image.get_data_dtype()),case.case_id)
                label=nib.load(case.label)
                label_key='label_'+str(label.get_data_dtype())
                chosen.setdefault(label_key,case.case_id)
                if image.shape[2]>maximum[0]: maximum=(image.shape[2],case.case_id)
            chosen['largest']=maximum[1]
        result=run_analysis(config,root,home,run_id=args.run_id,case_ids=sorted(set(chosen.values())))
        result['ok']=result['failed']==0
    return result
