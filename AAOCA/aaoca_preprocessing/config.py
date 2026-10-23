"""Fixed intensity contract and validated, explicit resource settings."""
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class Config:
    target_spacing: tuple[float, float, float] | None = None
    workers: int = 1
    chunk_depth: int = 8
    max_case_gib: float = 8.0

    def __post_init__(self):
        if self.target_spacing is not None:
            spacing = tuple(float(v) for v in self.target_spacing)
            if len(spacing) != 3 or not all(math.isfinite(v) and v > 0 for v in spacing):
                raise ValueError('target spacing must contain three positive finite millimetre values')
            object.__setattr__(self, 'target_spacing', spacing)
        for name, maximum in [('workers', 32), ('chunk_depth', 1024)]:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
                raise ValueError(f'{name} must be an integer in [1,{maximum}]')
        if not math.isfinite(self.max_case_gib) or self.max_case_gib <= 0:
            raise ValueError('max_case_gib must be positive and finite')

    def recipe(self):
        return {
            'schema': 1, 'algorithm': 'aaoca-linear-cell-fov-v1',
            'input_units': 'HU', 'clip_hu': [-1000, 1000], 'output_range': [-1, 1],
            'formula': 'clip(HU, -1000, 1000) / 1000',
            'intensity_order': 'linear resampling in HU, then clipping and normalization',
            'nominal_spacing_mm': self.target_spacing,
            'orientation': 'RAS axis order, preserve obliquity and shear',
            'grid': 'rounded shape; preserve voxel-cell FOV and centre; actual spacing in affine',
            'interpolation': 'scipy affine_transform order=1 mode=nearest prefilter=False',
            'crop': None, 'registration': None, 'dtype': 'float32',
            'packages': {p: importlib.metadata.version(p) for p in ('numpy', 'scipy', 'nibabel')},
            'implementation_sha256': hashlib.sha256(b''.join(
                path.name.encode() + path.read_bytes().replace(b'\r\n', b'\n')
                for path in sorted(Path(__file__).parent.glob('*.py')))).hexdigest(),
        }

    def signature(self):
        return hashlib.sha256(json.dumps(self.recipe(), sort_keys=True).encode()).hexdigest()
