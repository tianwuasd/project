"""Standalone AAOCA segmentation panel; no desktop or preprocessing imports."""
from pathlib import Path
import argparse
import json
import os
import sys

source = Path(__file__).resolve().parent / 'tools' / 'coarse_segmentation'
os.environ['AAOCA_SEG_SOURCE_CHECKOUT'] = str(source)
bootstrap = argparse.ArgumentParser(add_help=False)
bootstrap.add_argument('--runtime', default=os.environ.get('AAOCA_SEG_RUNTIME', '~/.local/state/aaoca-seg-terminal'))
options, _ = bootstrap.parse_known_args()
runtime = Path(options.runtime).expanduser().resolve()
try:
    release = json.loads((runtime / 'release.json').read_text(encoding='utf-8'))
    target = Path(release['path']).resolve()
    if target.is_relative_to(runtime / 'releases') and (target / 'aaoca_seg' / 'terminal.py').is_file():
        source = target
except (OSError, ValueError, KeyError, TypeError):
    pass
sys.path.insert(0, str(source))
from aaoca_seg.terminal import main

if __name__ == '__main__':
    raise SystemExit(main())
