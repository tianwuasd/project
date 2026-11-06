"""Numerical environment identity is part of the resume contract, not just a log."""
import json
import subprocess

PROBE = r'''
from importlib import metadata
import hashlib, json, platform, sys
packages = ['nnunetv2','torch','numpy','scipy','nibabel','scikit-image','blosc2']
files = ['preprocessing/preprocessors/default_preprocessor.py',
         'preprocessing/normalization/default_normalization_schemes.py',
         'preprocessing/resampling/default_resampling.py',
         'experiment_planning/dataset_fingerprint/fingerprint_extractor.py',
         'experiment_planning/experiment_planners/default_experiment_planner.py']
distribution = metadata.distribution('nnunetv2')
print(json.dumps(dict(python=platform.python_version(),platform=sys.platform,
    packages={p:metadata.version(p) for p in packages},
    official_modules={f:hashlib.sha256(distribution.locate_file('nnunetv2/'+f).read_bytes()).hexdigest() for f in files})))
'''


def numerical_runtime(python):
    result=subprocess.run([str(python),'-I','-c',PROBE],capture_output=True,text=True,encoding='utf-8',timeout=60)
    if result.returncode:
        raise RuntimeError('Research environment probe failed: '+result.stderr[-2000:])
    identity=json.loads(result.stdout)
    if identity['packages']['nnunetv2']!='2.8.1':
        raise ValueError('Requires nnunetv2==2.8.1')
    return identity
