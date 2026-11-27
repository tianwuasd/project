import json
import os
from pathlib import Path
import shutil
import sys
from time import perf_counter

import nibabel as nib
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT/'src'))


def main():
    root = Path(os.environ['CHD_COMPRESSED_TEST_ROOT']).resolve()
    report = Path(os.environ['CHD_COMPRESSED_TEST_REPORT']).resolve()
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    os.environ['CHD_RUNTIME_DIR'] = str(root/'runtime')
    (root/'.owner').write_text('compression-io-real', encoding='utf-8')
    (root/'.gitignore').write_text('*\n', encoding='utf-8')
    from chd_segmentation.preprocessing.arrays import save_compressed, CompressedArray
    from chd_segmentation.artifacts.store import write_json
    result = dict(state='running',optimizer_updates=0,scope='two real CT/label 32 cubed regions; compression I/O only')
    began = perf_counter()
    try:
        data = Path('D:/data/archive/ImageCHD_dataset/ImageCHD_dataset')
        for case in ('ct_1003','ct_1012'):
            for kind in ('image','label'):
                source = nib.load(data/(case+'_'+kind+'.nii.gz'))
                value = np.asarray(source.dataobj[224:256,224:256,80:112], dtype=np.float32 if kind=='image' else np.int16)
                target = root/(case+'_'+kind+'.b2nd')
                save_compressed(target,value,dict(compression_chunk_bytes=16*1024))
                loaded = CompressedArray(target)
                np.testing.assert_array_equal(np.asarray(loaded),value)
                np.testing.assert_array_equal(loaded[2:17,3:20,1:9],value[2:17,3:20,1:9])
                del loaded
        result['state'] = 'passed'
    finally:
        if sys.exception() is not None:
            result.update(state='failed',error=str(sys.exception()))
        result['seconds'] = perf_counter()-began
        files = [p for p in root.rglob('*') if p.is_file()]
        result.update(file_count=len(files),bytes=sum(p.stat().st_size for p in files))
        write_json(report,result)
        if root != Path(os.environ['CHD_COMPRESSED_TEST_ROOT']).resolve() or root.is_symlink() or (root/'.owner').read_text()!='compression-io-real':
            raise ValueError('测试目录归属不同')
        shutil.rmtree(root)
        result['cleanup'] = dict(deleted_files=len(files),released_bytes=result['bytes'],root_exists=root.exists())
        write_json(report,result)
    print(json.dumps(result,ensure_ascii=False))


if __name__ == '__main__':
    main()
