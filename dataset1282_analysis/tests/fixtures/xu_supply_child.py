from pathlib import Path
import sys
import numpy as np

from dataset1282_analysis.xu_supply import BatchSupply
from dataset1282_analysis.contracts import atomic_json

supply = BatchSupply([0,1,2,3],1,4,lambda b:np.full((4,)*3,b[0],dtype='float32'),lambda _:64)
with supply.batches() as batches:
    values = [float(row.mean()) for _,row in batches]
atomic_json(Path(sys.argv[1]),dict(values=values,stats=supply.stats))
