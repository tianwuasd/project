"""Validate a two-GPU Xu branch launch with synthetic formal-size inputs."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from dataset1282_analysis.xu_pair_probe import main

if __name__ == '__main__':
    sys.exit(main())
