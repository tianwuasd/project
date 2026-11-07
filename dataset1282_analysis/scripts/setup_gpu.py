"""Check/reuse a research runtime; install only into a separate owned venv."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from dataset1282_analysis.gpu_environment import main

if __name__ == '__main__':
    raise SystemExit(main())
