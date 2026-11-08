from pathlib import Path
import os
import signal
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))

if __name__ == '__main__':
    from dataset1282_analysis.xu_gpu_isolation import configure_coordinator,validate_entry
    if len(sys.argv)>1 and sys.argv[1]=='run':
        cpu = '--device' in sys.argv and sys.argv[sys.argv.index('--device')+1]=='cpu'
        configure_coordinator(cpu_only=cpu)
    elif len(sys.argv)>1 and sys.argv[1]=='worker':
        validate_entry(sys.argv[1:],os.environ,default='cpu')
    else:
        configure_coordinator(cpu_only=True)
    def stop(signum,frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    from dataset1282_analysis.prediction.cli import main
    main()
