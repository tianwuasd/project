import gc
import json
import os
from pathlib import Path
import shutil
import sys
from time import perf_counter
from unittest import TestCase

import numpy as np
import psutil
import torch


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT/'src'))


def run(root, report):
    root, report = root.resolve(), report.resolve()
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    (root/'.owner').write_text('compressed-real-two-cases', encoding='utf-8')
    (root/'.gitignore').write_text('*\n', encoding='utf-8')
    os.environ['CHD_RUNTIME_DIR'] = str(root/'runtime')
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
        os.environ[name] = '2'
    torch.set_num_threads(2)
    from chd_segmentation.artifacts.store import read_json, write_json, digest
    from chd_segmentation.config import load_config
    from chd_segmentation.preprocessing.prepare import prepare
    from chd_segmentation.preprocessing.offline import preprocess, case_pointer
    from chd_segmentation.preprocessing.offline_store import save_case, begin_case, inspect_case
    from chd_segmentation.preprocessing.arrays import B2ND_FORMAT, CompressedArray
    from chd_segmentation.data.dataset import CaseDataset, sample_patch
    from chd_segmentation.models.q_route1.geometry import crop_local_input
    from chd_segmentation.models.q_route1.training_inputs import prepare_inputs, pooled_image
    from chd_segmentation.models.stunet.adapter import StunetAdapter
    from chd_segmentation.models.workflow import options, save_state, load_state
    from chd_segmentation.models.q_route1.adapter import QAdapter
    from chd_segmentation.training.checkpoint import tensors_equal
    case = TestCase()
    began = perf_counter()
    result = dict(state='running', optimizer_updates=0, samples=['ct_1003','ct_1012'],
                  scope='at most 64 cubed real CT; one STU and one Q base CPU update; zero GPU')
    timings, checks = {}, []
    try:
        manifest = read_json('D:/projects/back/deliverables/先心分割处理/server-inputs/imagechd_manifest.json')
        ids = result['samples']
        rows = [dict(next(r for r in manifest['cases'] if r['case_id']==name), roles=[role])
                for name, role in zip(ids, ('train','validation'))]
        data = Path('D:/data/archive/ImageCHD_dataset/ImageCHD_dataset')
        for row in rows:
            for kind in ('image','label'):
                row[kind] = str(data/(row['case_id']+'_'+kind+'.nii.gz'))
        manifest.update(cases=rows, scope='technical', data_root=str(data), plan_scope='technical', binding_required=False)
        course = root/'course'
        write_json(course/'manifest.json', manifest)
        target = np.max([r['spacing_xyz'] for r in rows],axis=0).tolist()
        weight = 'D:/projects/先心/results/model_weights/stunet-small/small_ep4k.model'
        settings = dict(dataset='imagechd',scope='technical',manifest=str(course/'manifest.json'),
            pretrained=weight,work_root=str(course),run_dir=str(course/'runs/STU'),result_root=str(root/'result'),
            planning=dict(mode='auto',samples_per_case=1000,max_samples=1000,target_spacing_xyz=target),
            technical=dict(roi_xyz=[[224,256],[224,256],[80,112]]),
            preprocessing=dict(root=str(root/'data'),execution='whole',compression_chunk_bytes=16*1024),
            training=dict(patch=[32]*3,batch=1,attempts=1,epochs=None,steps_per_epoch=1,validation_steps=1),
            resources=dict(device='cpu',cpu_threads=2,workers=0,prefetch=1,cache_bytes=0,
                           max_workers=1,max_readers=1,reserve_ram_bytes=512*1024**2,memory_limit_bytes=16*1024**3),
            logging=dict(heartbeat_seconds=15,case_details=False))
        write_json(course/'config.json',settings)
        config = load_config(course/'config.json')
        mark = perf_counter()
        source = prepare(config,course/'source.json')
        compressed = preprocess(config,course/'source.json',course/'compressed.json')
        timings['prepare_compress_seconds'] = perf_counter()-mark
        case.assertEqual(compressed['storage']['format'],B2ND_FORMAT)
        again = preprocess(config,course/'source.json',course/'compressed.json')
        case.assertEqual(again,compressed)
        checks.append('actual default preprocessing and complete-case reuse')
        dataset = CaseDataset(compressed,dict(cache_bytes=0))
        entries = [dataset.get(name) for name in ids]
        from chd_segmentation.models.stunet.preprocess import StunetTransform
        for row, entry in zip(rows,entries):
            expected = StunetTransform(source,512*1024**2)(row)
            expected['processed_spacing'] = source['plan']['configurations']['3d_fullres']['spacing']
            case.assertTrue(isinstance(entry['image'],CompressedArray))
            for key in ('image','label'):
                np.testing.assert_array_equal(np.asarray(entry[key]),expected[key])
            directory = Path(compressed['storage']['root'])/compressed['storage']['cases'][row['case_id']]['key'][:16]
            case.assertFalse((directory/'image.npy').exists())
            case.assertFalse((directory/'label.npy').exists())
            for seed in (0,91):
                for a,b in zip(sample_patch(entry,[32]*3,np.random.default_rng(seed),.33),
                               sample_patch(expected,[32]*3,np.random.default_rng(seed),.33)):
                    np.testing.assert_array_equal(a,b)
            mark = perf_counter()
            for a,b in zip(prepare_inputs(entry,'imagechd',1024**3),prepare_inputs(expected,'imagechd',1024**3)):
                torch.testing.assert_close(a,b,atol=0,rtol=0)
            for a,b in zip(pooled_image(entry,1024**3),pooled_image(expected,1024**3)):
                torch.testing.assert_close(a,b,atol=0,rtol=0)
            probability = torch.softmax(torch.from_numpy(np.asarray(entry['image']))[None].repeat(1,11,1,1,1),1)
            for start in ((-7,0,2),(2,3,4)):
                a = crop_local_input(entry['image'],probability,start,32)
                b = crop_local_input(torch.from_numpy(expected['image'])[None],probability,start,32)
                for x,y in zip(a,b):
                    torch.testing.assert_close(x,y,atol=0,rtol=0)
            timings[row['case_id']+'_q_inputs_seconds'] = perf_counter()-mark
        checks.append('lossless source values, deterministic patches, Q heatmap/pooling and clipped local crops')
        old = dict(settings,run_dir=str(course/'runs/old'),preprocessing=dict(settings['preprocessing'],storage_format='npy'))
        write_json(course/'old-config.json',old)
        uncompressed = preprocess(load_config(course/'old-config.json'),course/'source.json',course/'old.json')
        old_dataset = CaseDataset(uncompressed,dict(cache_bytes=0))
        for name,entry in zip(ids,entries):
            np.testing.assert_array_equal(old_dataset.get(name)['image'],np.asarray(entry['image']))
        checks.append('manual npy export and existing npy reader remain supported')
        streamed_settings = dict(settings,run_dir=str(course/'runs/blocks'),
            preprocessing=dict(settings['preprocessing'],root=str(root/'stream-data'),execution='blocks'))
        write_json(course/'blocks-config.json',streamed_settings)
        streamed = preprocess(load_config(course/'blocks-config.json'),course/'source.json',course/'streamed.json')
        streamed_dataset = CaseDataset(streamed,dict(cache_bytes=0))
        for name,entry in zip(ids,entries):
            other = streamed_dataset.get(name)
            np.testing.assert_allclose(np.asarray(other['image']),np.asarray(entry['image']),atol=1e-5,rtol=1e-5)
            np.testing.assert_array_equal(np.asarray(other['label']),np.asarray(entry['label']))
        case.assertFalse(list(Path(streamed['storage']['root']).rglob('*.npy')))
        del other,streamed_dataset
        checks.append('bounded preprocessing transfers compressed results and removes temporary work arrays')
        # 实际取消一个压缩写出，下一次沿原病例身份清理未完成目录并重写。
        interrupted = root/'interrupted'
        def cancel():
            raise InterruptedError('bounded test cancellation')
        with case.assertRaises(InterruptedError):
            save_case(interrupted,'actual-case','actual-recipe',entries[0],format=B2ND_FORMAT,boundary=cancel)
        case.assertFalse(interrupted.exists())
        record = save_case(interrupted,'actual-case','actual-recipe',entries[0],format=B2ND_FORMAT)
        case.assertEqual(inspect_case(interrupted,'actual-case','actual-recipe'),record)
        checks.append('cancelled compression never commits; next call resumes by clean bounded rewrite')
        # 两种容器的预测窗口使用同一真实网络、同一权重，仅比较前向，不增加更新。
        spec = StunetAdapter.spec(config,source['plan'])
        model = StunetAdapter.build(spec)
        StunetAdapter.load_pretrained(model,weight)
        model.eval()
        image,label = sample_patch(entries[0],[32]*3,np.random.default_rng(7),.33)
        dense_image,dense_label = sample_patch(old_dataset.get(ids[0]),[32]*3,np.random.default_rng(7),.33)
        with torch.no_grad():
            first,second = model(torch.from_numpy(image)[None]),model(torch.from_numpy(dense_image)[None])
        for a,b in zip(first,second):
            torch.testing.assert_close(a,b,atol=0,rtol=0)
        del first,second
        model.train()
        optimizer = torch.optim.SGD(model.parameters(),lr=.001,momentum=.99)
        mark = perf_counter()
        loss = StunetAdapter.training_step(model,dict(image=torch.from_numpy(image)[None],
            label=torch.from_numpy(label)[None],unknown_classes=[entries[0]['unknown_classes']]),'imagechd')
        loss.backward()
        optimizer.step()
        result['optimizer_updates'] += 1
        save_state(root/'stu.pt',dict(model=model.state_dict(),optimizer=optimizer.state_dict()))
        restored = load_state(root/'stu.pt')
        case.assertTrue(tensors_equal(model.state_dict(),restored['model']))
        optimizer.load_state_dict(restored['optimizer'])
        timings['stu_update_save_read_seconds'] = perf_counter()-mark
        del restored,model,optimizer,loss
        gc.collect()
        q = dict(model='q_route1',run_dir=str(course/'runs/Q'),work_root=str(course),device='cpu',precision='fp32',
                 technical=True,updates=1,patch=32,local_size=32,accumulation=1,pretrained=weight,
                 execution=dict(workers=0,cpu_processes=1),
                 technical_roles=dict(base=dict(train=[ids[0]],validation=[ids[1]],test=[])))
        write_json(course/'q.json',q)
        bundle = QAdapter(options(course/'q.json'),compressed,'base')
        optimizer = bundle.optimizer()
        mark = perf_counter()
        loss = bundle.loss(entries[0],np.random.default_rng(9),0)
        case.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        optimizer.step()
        result['optimizer_updates'] += 1
        save_state(root/'q.pt',dict(model=bundle.model.state_dict(),optimizer=optimizer.state_dict()))
        restored = load_state(root/'q.pt')
        case.assertTrue(tensors_equal(restored['model'],bundle.model.state_dict()))
        optimizer.load_state_dict(restored['optimizer'])
        timings['q_update_save_read_seconds'] = perf_counter()-mark
        checks.append('same model outputs for npy/b2nd; one STU and one Q update, saved model and optimizer readback')
        result.update(state='passed',timings=timings,checks=checks,
                      compression_bytes=sum(p.stat().st_size for p in Path(compressed['storage']['root']).rglob('*.b2nd')),
                      uncompressed_bytes=sum(p.stat().st_size for p in Path(uncompressed['storage']['root']).rglob('*.npy')),
                      stored_shapes=[list(e['image'].shape) for e in entries])
        del restored,bundle,optimizer,entries,dataset,old_dataset,entry,expected
        gc.collect()
    finally:
        if sys.exception() is not None:
            result.update(state='failed',error=str(sys.exception()))
        result.update(seconds=perf_counter()-began,timings=timings,checks=checks,
                      process_peak_rss_bytes=getattr(psutil.Process().memory_info(),'peak_wset',None),vram_peak_bytes=None)
        files = [p for p in root.rglob('*') if p.is_file()]
        result.update(file_count=len(files),bytes=sum(p.stat().st_size for p in files))
        write_json(report,result)
        if root != Path(os.environ['CHD_COMPRESSED_TEST_ROOT']).resolve() or root.is_symlink() or (root/'.owner').read_text()!='compressed-real-two-cases':
            raise ValueError('测试目录归属不同，停止清理')
        shutil.rmtree(root)
        result['cleanup'] = dict(deleted_files=len(files),released_bytes=result['bytes'],root_exists=root.exists())
        write_json(report,result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    run(Path(os.environ['CHD_COMPRESSED_TEST_ROOT']),Path(os.environ['CHD_COMPRESSED_TEST_REPORT']))
