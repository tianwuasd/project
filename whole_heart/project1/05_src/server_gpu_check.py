"""在独立进程中检查指定的一张GPU，退出后释放CUDA上下文再开始任务队列。"""
import argparse
from pathlib import Path
from server_launch import runtime_check

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('work')
    parser.add_argument('gpu_memory', type=float)
    args = parser.parse_args()
    runtime_check(Path(args.work), args.gpu_memory)
