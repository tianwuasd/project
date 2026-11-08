from pathlib import Path
import argparse
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from dataset1282_analysis.contracts import atomic_json, read_json, sha256_file


def main():
    parser = argparse.ArgumentParser(description='Prediction acceptance using completed technical training')
    parser.add_argument('--training',type=Path,required=True)
    parser.add_argument('--models',type=Path,help='Optional completed four-branch forward-test fixture')
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    training = args.training.resolve(strict=True)
    if read_json(training/'run.json')['scope'] != 'technical_subset':
        raise ValueError('Acceptance requires synthetic technical training, with formal held-out data sealed')
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('Acceptance requires a fresh output directory')
    out.mkdir(parents=True,exist_ok=True)
    scratch = out/'temporary'
    scratch.mkdir()
    env = dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
               PYTHONPATH=os.pathsep.join([str(ROOT/'src'),str(ROOT/'tests'),os.environ.get('PYTHONPATH','')]),
               CUDA_VISIBLE_DEVICES='-1',PYTHONIOENCODING='utf-8',TEMP=str(scratch),TMP=str(scratch),
               DATASET1282_PREDICTION_TEST_TRAINING=str(training))
    modules = ['test_prediction_selection','test_prediction_preparation','test_prediction_contracts',
               'test_prediction_existing']
    if args.models:
        env['DATASET1282_PREDICTION_TEST_MODELS'] = str(args.models.resolve(strict=True))
        modules.append('test_prediction_forward')
    checks = []
    for name in modules:
        started = time.monotonic()
        log = out/(name+'.log')
        with log.open('w',encoding='utf-8') as stream:
            result = subprocess.run([sys.executable,'-B','-u','-m','unittest',name,'-v'],
                                    cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=900)
        row = dict(check=name,seconds=time.monotonic()-started,exit=result.returncode,log_sha256=sha256_file(log))
        checks.append(row)
        atomic_json(out/'checks.json',checks)
        print(row,flush=True)
        if result.returncode:
            raise RuntimeError(f'{name}: see {log}')
    atomic_json(out/'acceptance.json',dict(scope='synthetic CPU prediction acceptance',
                training_protocol=read_json(training/'run.json')['protocol'],checks=checks,cuda_tested=False))
    print(out/'acceptance.json',flush=True)


if __name__ == '__main__':
    main()
