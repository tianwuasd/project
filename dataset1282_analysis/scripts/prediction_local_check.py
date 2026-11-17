import argparse
import io
import json
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import nibabel as nib
import numpy as np
import torch

from dataset1282_analysis.xu_geometry import CaseGeometry, _resize
from dataset1282_analysis.xu_models import build_model
from dataset1282_analysis.xu_training import load_predictor
from dataset1282_analysis.prediction.heldout import select_cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    torch.set_num_threads(2)
    image = nib.load(args.image)
    values = np.asarray(image.dataobj[::8, ::8, ::4], dtype=np.float32)
    grid = _resize(values, (64, 64, 64), 1)
    grid = (grid - grid.mean()) / max(float(grid.std()), 1e-6)
    model = build_model('crop64', 'smoke')
    state = dict(contract=dict(branch='crop64', profile='smoke'), model=model.state_dict())
    buffer = io.BytesIO()
    torch.save(state, buffer)
    buffer.seek(0)
    predictor = load_predictor(buffer, branch='crop64', profile='smoke',
                               data_identity=None, validate_identity=False)
    predicted = predictor(grid)
    assert predicted.shape == grid.shape and predicted.dtype == np.uint8
    geo = CaseGeometry(values.shape, (1., 1., 2.), (0, 1, 2),
                       tuple((0, n) for n in values.shape[::-1]), grid.shape)
    restored = geo.restore_label(predicted)
    assert restored.shape == values.shape
    # 真实病例仅验证技术路径，不参与科研效果评价。
    name = args.image.name.removesuffix('.nii.gz')
    cohort = dict(cases={name: dict(role='source_test', included=True, index_scan=True, patient=name)})
    assert list(select_cases(cohort, ['source_test'])) == [name]
    print(json.dumps(dict(image=str(args.image), sampled_shape=list(values.shape),
                          network_shape=list(predicted.shape), restored_shape=list(restored.shape),
                          checkpoint_bytes=buffer.getbuffer().nbytes, seconds=time.monotonic()-started,
                          generated_files=0, training_updates=0, device='cpu',
                          scope='real CT read, in-memory technical model save/load, forward and geometry; no trained-model accuracy'),
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
