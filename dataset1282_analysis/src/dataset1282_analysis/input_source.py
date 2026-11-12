from pathlib import Path
import pickle

import nibabel as nib
import numpy as np

from .xu_geometry import PreparedCache


class LegacyInput(PreparedCache):
    online = False

    def properties(self, name):
        path = self._artifact(f'process-{name}', f'{self._folder}/{name}.pkl')
        with path.open('rb') as stream:
            return pickle.load(stream)

    def native_image_path(self, name):
        return self._artifact(f'stage-{name}', f'staged/{name}/image{self.staged_extension}')

    def read_native_image(self, name):
        image = nib.load(self.native_image_path(name))
        geometry = self.geometry(name)
        if image.shape != geometry.native_shape_xyz or not np.allclose(
                image.affine, np.diag([*geometry.spacing_xyz, 1]), rtol=1e-6, atol=1e-7):
            raise ValueError('原图空间信息与预处理合同不同')
        values = image.get_fdata(dtype=np.float32, caching='unchanged')
        if not np.isfinite(values).all():
            raise ValueError('原图包含非有限数值')
        return values

    def verify_image(self, name):
        return self._artifact(f'process-{name}', f'{self._folder}/{name}.b2nd')


def open_prepared(run_dir, *, allow_technical_subset=False):
    if (Path(run_dir)/'online.json').is_file():
        from .online_input import OnlineInput
        return OnlineInput(run_dir, allow_technical_subset=allow_technical_subset)
    return LegacyInput(run_dir, allow_technical_subset=allow_technical_subset)
