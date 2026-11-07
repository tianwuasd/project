"""Run Xu segmentation experiments with the research Python from an isolated code checkout."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from dataset1282_analysis.xu_workflow import main

if __name__ == '__main__':
    import os
    if any(os.environ.get(k) for k in ('XU_HOST_BUDGET_BYTES','XU_BRANCH_DEVICES','XU_CUDA_ADMISSIONS')):
        from dataset1282_analysis.xu_resources import apply_env_limit
        device = sys.argv[sys.argv.index('--device') + 1] if os.environ.get('XU_CUDA_ADMISSION') else 'cpu'
        print('Applied fresh resource limits:', apply_env_limit(device), flush=True)
    main()
