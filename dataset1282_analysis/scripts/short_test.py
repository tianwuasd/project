import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from dataset1282_analysis.contracts import atomic_json, read_json, sha256_file
from dataset1282_analysis.short_workspace import ShortWorkspace


def fixture(base):
    import nibabel as nib
    import numpy as np
    from openpyxl import Workbook
    from dataset1282_analysis.metadata import FIELDS,read_metadata
    from dataset1282_analysis.inputs import load_cases
    from dataset1282_analysis.preparation import allocate
    data=base/'data'
    (data/'native/images').mkdir(parents=True)
    (data/'native/labels').mkdir()
    workbook=Workbook()
    sheet=workbook.active
    sheet.title='dicom2nii_info'
    sheet.append(FIELDS)
    manifest=[]
    shape=(40,40,32)
    xyz=np.indices(shape)
    mask=((xyz[0]>3)&(xyz[0]<36)&(xyz[1]>3)&(xyz[1]<36)&(xyz[2]>3)&(xyz[2]<29))
    arr=np.where(mask,100+xyz[0]*8+xyz[2],-1000).astype(np.int16)
    seg=np.where(mask,1+(xyz[0]//3)%10,0).astype(np.uint8)
    for i in range(16):
        name=f'SYN_{i:03d}'
        device='LightSpeed VCT' if i<8 else 'SOMATOM Definition Flash'
        row=dict(idx=name,PatientName='synthetic',PatientID=f'p{i:03d}',PatientBirthDate='20200101',
                 PatientSex='M',AcquisitionDate='20210101',ManufacturerModelName=device,
                 PixelSpacingX=.8,PixelSpacingY=.8,SliceThickness=1.,Rows=40,Cols=40,
                 all_file_z=32,all_read_file_z=32,valid_z=32,calc_zSpacing=1.,dicom_folder='synthetic')
        sheet.append([row[k] for k in FIELDS])
        item=dict(case_id=name,source_key=name,collection='native',outputs=[])
        for kind,suffix,array in [('images','_0000',arr),('labels','',seg)]:
            path=data/'native'/kind/(name+suffix+'.nii.gz')
            affine=np.eye(4)
            if kind=='labels': affine[:3,3]=1
            # Missing structure is deliberately retained as ambiguous background.
            array=np.where(array==10,0,array) if kind=='labels' and i%2==0 else array
            nib.save(nib.Nifti1Image(array,affine),path)
            item['outputs'].append(dict(path=str(path),sha256=sha256_file(path)))
        manifest.append(item)
    excel=base/'metadata.xlsx'
    workbook.save(excel)
    atomic_json(data/'manifest.json',manifest)
    config=dict(data_root=str(data),metadata_excel=str(excel),work_root=str(base/'work'),environment=sys.prefix)
    atomic_json(base/'config.json',config)
    cohort=allocate(load_cases(config),read_metadata(excel))
    cohort['inputs']=dict(manifest=sha256_file(data/'manifest.json'),excel=sha256_file(excel))
    atomic_json(base/'cohort.json',cohort)
    selected=[]
    for role,device in [('train','LightSpeed VCT'),('train','SOMATOM Definition Flash'),('development','LightSpeed VCT')]:
        eligible=[n for n,r in cohort['cases'].items() if r['role']==role and r['device']==device]
        selected.append(sorted(eligible,key=lambda n:(int(n[-3:])%2,n))[0])
    policy=dict(patient_aliases=[],excluded_cases=[],HU_policy=dict(additional_offset=0,additional_scale=1),
                partial_supervision=dict(include_training_partition_missing_class_cases=True),
                label_mapping=dict(values={str(i):'background' if i==0 else f'class_{i}' for i in range(11)},renumber_voxels=False))
    atomic_json(base/'policy.json',policy)
    return base/'config.json',base/'policy.json',base/'cohort.json',selected


def main():
    parser=argparse.ArgumentParser(description='Logs-only synthetic or real-subset acceptance')
    parser.add_argument('--work-root',required=True)
    parser.add_argument('--research-python',required=True)
    parser.add_argument('--real-config')
    parser.add_argument('--policy')
    parser.add_argument('--frozen-cohort')
    parser.add_argument('--case',action='append',dest='cases')
    args=parser.parse_args()
    protected=[]
    if args.real_config:
        if not all((args.policy,args.frozen_cohort,args.cases)):
            parser.error('Real test requires policy, frozen cohort and explicit case IDs')
        cfg=read_json(args.real_config)
        protected=[args.real_config,args.policy,args.frozen_cohort,cfg['data_root'],cfg['metadata_excel']]
    with ShortWorkspace(args.work_root,protected_paths=protected) as workspace:
        workspace.record('configuration',arguments=vars(args),
                         inputs={str(path):sha256_file(path) for path in protected if Path(path).is_file()})
        run_checks(args,workspace)
    print(workspace.logs,flush=True)


def run_checks(args,workspace):
    base=workspace.scratch
    home=base/'home'
    home.mkdir()
    env=dict(os.environ,PYTHONPATH=str(ROOT/'src'),PYTHONUTF8='1',TMP=str(base),TEMP=str(base),
             CUDA_VISIBLE_DEVICES='-1',DATASET1282_HOME=str(home),DATASET1282_CODE_ROOT=str(ROOT))
    evidence=[]
    def command(name,cmd,success=True,timeout=1800,expected_error=None):
        start=time.monotonic()
        log=workspace.logs/(name+'.log')
        workspace.record('command',check=name,argv=cmd)
        with log.open('wb') as stream:
            result=subprocess.run(cmd,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=timeout)
        evidence.append(dict(check=name,exit=result.returncode,seconds=time.monotonic()-start,
                             passed=(result.returncode==0)==success))
        if expected_error is not None:
            evidence[-1]['passed'] &= expected_error in log.read_text(encoding='utf-8',errors='replace')
        workspace.record('check',**evidence[-1])
        print(name,evidence[-1],flush=True)
        if not evidence[-1]['passed']:
            raise RuntimeError('Failed '+name+'; see '+str(log))
    command('workflow-unit',[sys.executable,'-B','-m','unittest','discover','-s',str(ROOT/'tests'),'-p','test_workflow.py','-v'])
    command('marginal-unit',[args.research_python,'-B','-m','unittest','discover','-s',str(ROOT/'tests'),'-p','test_marginal.py','-v'])
    if args.real_config:
        config,policy,cohort,names=Path(args.real_config),Path(args.policy),Path(args.frozen_cohort),args.cases
        # Redirect outputs, retaining the real immutable inputs and local environment.
        cfg=read_json(config)
        cfg['work_root']=str(base/'work')
        atomic_json(base/'config.json',cfg)
        config=base/'config.json'
    else:
        config,policy,cohort,names=fixture(base)
    workspace.record('input_configuration',config=read_json(config),cases=names,
                     identities={str(path):sha256_file(path) for path in (config,policy,cohort)},
                     synthetic=not bool(args.real_config))
    common=[sys.executable,'-B','-c',
            'from dataset1282_analysis.cli import main; raise SystemExit(main())',
            '--config',str(config),'prepare',
            '--workflow',str(policy),'--frozen-cohort',str(cohort),'--research-python',args.research_python]
    for name in names: common+=['--case',name]
    serial=common+['--run','serial','--workers','1','--staged-extension','.nii.gz']
    parallel=common+['--run','adaptive','--workers','2','--staged-extension','.nii']
    command('serial',serial)
    command('adaptive',parallel)
    outputs=[base/'work/prepared'/r/'hu-marginal' for r in ('serial','adaptive')]
    command('numerical-and-trainer',[args.research_python,'-B',str(ROOT/'scripts/verify_prepared.py'),
                                    str(outputs[0]),str(outputs[1]),str(base/'numerical.json')])
    markers={str(p):p.stat().st_mtime_ns for p in (outputs[1]/'commits').glob('*.json')}
    command('resume',parallel+['--resume'])
    assert all(Path(p).stat().st_mtime_ns==stamp for p,stamp in markers.items()),'Resume rewrote completed jobs'
    request_path=outputs[1]/'request.json'
    original_request=request_path.read_bytes()
    changed=read_json(request_path)
    changed['numerical_runtime']['packages']['numpy']='deliberate-test-mismatch'
    atomic_json(request_path,changed)
    try:
        command('runtime-change-rejected',parallel+['--resume'],success=False,
                expected_error='Existing run requires --resume')
    finally:
        request_path.write_bytes(original_request)
    target=next((outputs[1]/'preprocessed').rglob('*_seg.b2nd'))
    with target.open('r+b') as stream:
        original=stream.read(1)
        stream.seek(0)
        stream.write(bytes([original[0]^1]))
    try:
        command('tamper-rejected',parallel+['--resume'],success=False,
                expected_error='Committed artifact is missing or changed')
    finally:
        with target.open('r+b') as stream: stream.write(original)
    # Remove only one generated completion marker to model interruption before commit.
    marker=outputs[1]/'commits'/('process-'+names[0]+'.json')
    marker.unlink()
    command('incomplete-case-recovery',parallel+['--resume'])
    command('recovery-equivalence',[args.research_python,'-B',str(ROOT/'scripts/verify_prepared.py'),
                                    str(outputs[0]),str(outputs[1]),str(base/'recovered.json')])
    serial_result=read_json(outputs[0]/'complete.json')
    adaptive_seconds=next(x['seconds'] for x in evidence if x['check']=='adaptive')
    serial_seconds=next(x['seconds'] for x in evidence if x['check']=='serial')
    report=dict(passed=True,kind='real_subset' if args.real_config else 'synthetic',checks=evidence,
                directory=str(base),cases=len(names),serial_seconds=serial_seconds,adaptive_seconds=adaptive_seconds,
                speedup=serial_seconds/adaptive_seconds,serial_stages=serial_result['stages'],
                scope='technical acceptance, not whole-cohort preprocessing or research training')
    atomic_json(base/'report.json',report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__': main()
