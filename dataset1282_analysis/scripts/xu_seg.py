"""Run Xu segmentation experiments with the research Python from an isolated code checkout."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from dataset1282_analysis.xu_workflow import main

if __name__ == '__main__':
    main()
