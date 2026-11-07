"""Run or inspect the fixed-split Xu segmentation pipeline with the research Python."""
from pathlib import Path
import signal
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from dataset1282_analysis.xu_pipeline import main

if __name__ == '__main__':
    if len(sys.argv)>1 and sys.argv[1]=='run':
        from dataset1282_analysis.xu_gpu_isolation import configure_coordinator
        cpu = '--device' in sys.argv and sys.argv[sys.argv.index('--device')+1]=='cpu'
        configure_coordinator(cpu_only=cpu)
    def stop(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    sys.exit(main())
