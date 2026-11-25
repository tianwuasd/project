from pathlib import Path
import signal
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))

if __name__=='__main__':
    def stop(signum,frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    from dataset1282_analysis.xu_gpu_isolation import configure_coordinator
    configure_coordinator(cpu_only='--device' in sys.argv and sys.argv[sys.argv.index('--device')+1]=='cpu'
                          or not {'predict','short'}.intersection(sys.argv))
    from dataset1282_analysis.prediction.data4 import main
    main()
