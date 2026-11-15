import argparse
from pathlib import Path
import sys


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source',required=True)
    parser.add_argument('--config',required=True)
    parser.add_argument('--prepared',required=True)
    parser.add_argument('--stage',choices=('base','localizer','local'),required=True)
    args = parser.parse_args()
    source = Path(args.source).resolve()/'src'
    sys.path.insert(0,str(source))
    from chd_segmentation.models.workflow import options,train
    import chd_segmentation.models.workflow as workflow
    if not Path(workflow.__file__).resolve().is_relative_to(source):
        raise ValueError('旧版短测未使用指定的完整源码副本')
    train(options(args.config),args.prepared,args.stage,stop_after=1)
