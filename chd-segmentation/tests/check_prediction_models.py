import argparse
import ast
import json
from pathlib import Path
import sys
from time import perf_counter

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def check(prepared_path=None):
    started = perf_counter()
    import psutil
    from chd_segmentation.models.workflow import options
    from chd_segmentation.models.contracts import CAPABILITIES
    paths = list((ROOT/'src/chd_segmentation/models').rglob('*.py'))
    paths += [ROOT/'src/chd_segmentation/models_cli.py',ROOT/'src/chd_segmentation/prediction/metrics.py',
        ROOT/'src/chd_segmentation/preprocessing/prepare.py',Path(__file__)]
    for path in paths:
        ast.parse(path.read_text(encoding='utf-8-sig'),filename=str(path))
    configs = [options(path) for path in sorted((ROOT/'configs/models').glob('*.yaml'))]
    assert CAPABILITIES['totalsegmentator']==CAPABILITIES['medsam3']=={'predict'}
    for config in configs:
        if config['model'] in ('totalsegmentator','medsam3'):
            assert config['execution']['memory_limit_bytes']<=100_000_000_000
            assert config['execution']['ram_fraction']<=.8
    result = dict(syntax_files=len(paths),models=[config['model'] for config in configs],files_created=0)
    if prepared_path:
        import nibabel as nib
        import numpy as np
        from chd_segmentation.models.standalone_prediction import working_image,case_workspace
        prepared = json.loads(Path(prepared_path).read_text(encoding='utf-8'))
        rows = prepared['manifest']['cases'][:2]
        headers = []
        for row in rows:
            image = nib.load(row['image'])
            working = working_image(image,row)
            assert np.allclose(np.linalg.norm(working.affine[:3,:3],axis=0),row['spacing_xyz'])
            headers.append(dict(case_id=row['case_id'],shape=list(image.shape),dtype=str(image.get_data_dtype()),
                source_affine=image.affine.tolist(),working_affine=working.affine.tolist(),
                spacing_xyz=row['spacing_xyz'],mask_bytes=int(np.prod(image.shape)),
                workspace={model:case_workspace(image,row,model) for model in ('totalsegmentator','medsam3')}))
        result['real_headers'] = headers
    result['seconds'] = perf_counter()-started
    memory = psutil.Process().memory_info()
    result['rss_bytes'] = memory.rss
    result['peak_rss_bytes'] = getattr(memory,'peak_wset',None)
    result['gpu_memory'] = '未测量；未初始化模型或CUDA'
    return result


def real(config_path,prepared_path,cases,output):
    from chd_segmentation.models.workflow import options
    from chd_segmentation.models.prediction import predict
    if not 1 <= len(cases) <= 2 or len(set(cases))!=len(cases):
        raise ValueError('真实短流程只允许1至2例、0次训练更新')
    config = options(config_path)
    started = perf_counter()
    first = predict(config,prepared_path,None,output,cases)
    second = predict(config,prepared_path,None,output,cases)
    assert second['reused']
    records = json.loads((Path(output)/'predictions.json').read_text(encoding='utf-8'))['cases']
    assert set(cases) <= records.keys()
    for name in cases:
        assert (Path(output)/records[name]['file']).is_file()
        assert records[name]['model']==config['model']
        if config['model']=='totalsegmentator':
            assert len(records[name]['covered_classes'])==7
    return dict(first=first,repeated=second,seconds=perf_counter()-started,updates=0,
        tested_cases=cases,scope='真实预测及逐例续跑；未验证全量效果或正式性能')


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared')
    parser.add_argument('--config')
    parser.add_argument('--case',action='append')
    parser.add_argument('--output')
    parser.add_argument('--run-real',action='store_true')
    args = parser.parse_args()
    result = real(args.config,args.prepared,args.case or [],args.output) if args.run_real else check(args.prepared)
    print(json.dumps(result,ensure_ascii=False,allow_nan=False,indent=2))
