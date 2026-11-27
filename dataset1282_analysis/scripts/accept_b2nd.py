import argparse
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))

if __name__=='__main__':
    from dataset1282_analysis.prediction.read_acceptance import accept_xu
    p=argparse.ArgumentParser()
    for name in ('source','compressed','cht-code','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    print(accept_xu(a.source,a.compressed,a.cht_code,a.output))
