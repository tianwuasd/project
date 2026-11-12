from pathlib import Path
import sys

source = Path(__file__).resolve().parent / 'tools' / 'coarse_segmentation'
sys.path.insert(0, str(source))

from aaoca_seg.direct import main

if __name__ == '__main__':
    raise SystemExit(main())
