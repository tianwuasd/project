"""Run one bounded, full-width Xu training step on the selected device."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from dataset1282_analysis.xu_capacity import main

if __name__ == '__main__':
    sys.exit(main())
