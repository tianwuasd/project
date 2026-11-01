"""CPU regression for the exact nnU-Net 2.8.1 VRAM-estimation API.

No CT, CUDA, training or complete planning run. By default use installed 2.8.1;
--upstream-source-dir accepts the three official source files for an isolated
API check, with actual Torch and dynamic-network-architectures installed.
"""
import argparse
import ast
import hashlib
import inspect
import json
import os
from pathlib import Path
import pydoc
import subprocess
import sys
from typing import Tuple, Union
import warnings
from unittest.mock import patch

CODE = Path(__file__).resolve().parents[1]


def load_function(source, name, namespace):
    node = next(n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<official nnU-Net 2.8.1 API>', 'exec'), namespace)
    return namespace[name], hashlib.sha256(ast.unparse(node).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream-source-dir')
    parser.add_argument('--dependencies', help='Optional isolated dependency directory')
    parser.add_argument('--report-file')
    args = parser.parse_args()
    if args.dependencies:
        sys.path.insert(0, str(Path(args.dependencies).resolve()))
    import importlib.metadata as metadata
    import torch
    from dynamic_network_architectures.architectures.unet import PlainConvUNet
    namespace = dict(torch=torch, os=os, subprocess=subprocess, Tuple=Tuple, Union=Union,
                     pydoc=pydoc, warnings=warnings, join=os.path.join)
    hashes = {}
    if args.upstream_source_dir:
        root = Path(args.upstream_source_dir)
        for filename, name in [('default_n_proc_DA.py', 'get_allowed_n_proc_DA'),
                               ('get_network_from_plans.py', 'get_network_from_plans'),
                               ('default_experiment_planner.py', 'static_estimate_VRAM_usage')]:
            _, hashes[name] = load_function((root / filename).read_text(encoding='utf-8'), name, namespace)
        estimate = namespace['static_estimate_VRAM_usage']
        source_mode = 'official 2.8.1 method bodies, isolated imports'
    else:
        if metadata.version('nnunetv2') != '2.8.1':
            raise ValueError('This regression targets nnU-Net 2.8.1')
        from nnunetv2.experiment_planning.experiment_planners.default_experiment_planner import ExperimentPlanner
        estimate = ExperimentPlanner.static_estimate_VRAM_usage
        hashes['static_estimate_VRAM_usage'] = hashlib.sha256(inspect.getsource(estimate).encode()).hexdigest()
        source_mode = 'installed nnU-Net 2.8.1'
    helper = (CODE / '05_src/prepare_nnunet.py').read_text(encoding='utf-8')
    configure, _ = load_function(helper, 'configure_paths', dict(os=os, ROOT=CODE / 'synthetic-path-not-created'))
    architecture = dict(n_stages=3, features_per_stage=[4, 8, 16],
                        conv_op='torch.nn.modules.conv.Conv3d', kernel_sizes=[[3, 3, 3]] * 3,
                        strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2]], n_conv_per_stage=[2, 2, 2],
                        n_conv_per_stage_decoder=[2, 2], conv_bias=True,
                        norm_op='torch.nn.modules.instancenorm.InstanceNorm3d',
                        norm_op_kwargs=dict(eps=1e-5, affine=True), dropout_op=None,
                        dropout_op_kwargs=None, nonlin='torch.nn.LeakyReLU', nonlin_kwargs=dict(inplace=True))
    def call():
        return int(estimate((16, 16, 16), 1, 8, PlainConvUNet.__module__ + '.PlainConvUNet',
                            architecture, ('conv_op', 'norm_op', 'dropout_op', 'nonlin')))
    original_threads = torch.get_num_threads()
    checks, values = [], []
    try:
        torch.set_num_threads(2)
        with patch.dict(os.environ, {'nnUNet_n_proc_DA': '0'}):
            try:
                call()
            except RuntimeError as error:
                if 'positive integer' not in str(error):
                    raise
                checks.append('original_zero_thread_runtime_error_reproduced')
            else:
                raise AssertionError('Original planner failure was not reproduced')
        for budget in [4, 1]:
            with patch.dict(os.environ, {'WHOLE_HEART_CPU_THREADS': str(budget), 'nnUNet_n_proc_DA': '0'}):
                configure()
                assert int(os.environ['nnUNet_n_proc_DA']) == budget
                values.append(call())
                assert values[-1] > 0 and torch.get_num_threads() == 2
                checks.append(f'actual_3d_unet_feature_map_estimate_with_{budget}_threads')
        assert values[0] == values[1]
        checks.append('positive_estimate_unchanged_by_thread_count_and_threads_restored')
        assert not torch.cuda.is_initialized()
    finally:
        torch.set_num_threads(original_threads)
    report = dict(status='passed', source_mode=source_mode, torch=torch.__version__,
                  dynamic_network_architectures=metadata.version('dynamic-network-architectures'),
                  upstream_methods_sha256=hashes, device='cpu', cuda_initialized=False,
                  real_ct=False, complete_planning=False, full_gpu_preflight=False,
                  feature_map_estimate=values[0], checks=checks)
    if args.report_file:
        Path(args.report_file).write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
