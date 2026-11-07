"""Validate a two-GPU Xu branch launch with synthetic formal-size inputs."""
from pathlib import Path
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from dataset1282_analysis.xu_pair_probe import main

if __name__ == '__main__':
    from dataset1282_analysis.xu_gpu_isolation import configure_coordinator, validate_entry
    if len(sys.argv)>1 and sys.argv[1]=='_worker':
        validate_entry([],os.environ,default='cuda:0')
    elif '--help' not in sys.argv and '-h' not in sys.argv:
        configure_coordinator()
    sys.exit(main())
