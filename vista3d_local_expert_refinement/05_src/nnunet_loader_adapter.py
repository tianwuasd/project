"""Project-local, fail-closed adapter for nnU-Net 2.8.1's inner thread limit.

Only the literal ``threadpool_limits(limits=1)`` argument is changed. Crop,
oversampling, transforms, dtypes, assembly and RNG operations are upstream code.
No upstream module or installed file is mutated.
"""
import ast
import hashlib
import importlib.metadata
import inspect
import textwrap


def configured_loader_class():
    from nnunetv2.training.dataloading import data_loader
    upstream = data_loader.nnUNetDataLoader
    version = importlib.metadata.version('nnunetv2')
    if version != '2.8.1':
        raise RuntimeError(f'loader adapter requires nnunetv2 2.8.1, got {version}')
    source = textwrap.dedent(inspect.getsource(upstream.generate_train_batch))
    tree = ast.parse(source)
    matches = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
               and isinstance(node.func, ast.Name) and node.func.id == 'threadpool_limits']
    if len(matches) != 1:
        raise RuntimeError('unsupported upstream loader: expected one inner threadpool limit')
    limits = [kw for kw in matches[0].keywords if kw.arg == 'limits']
    if len(limits) != 1 or not isinstance(limits[0].value, ast.Constant) or limits[0].value.value != 1:
        raise RuntimeError('unsupported upstream loader: inner limit changed')
    limits[0].value = ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()),
                                    attr='worker_threads', ctx=ast.Load())
    ast.fix_missing_locations(tree)
    namespace = dict(vars(data_loader))
    exec(compile(tree, inspect.getsourcefile(upstream) + ':project-thread-adapter', 'exec'), namespace)
    adapted = namespace['generate_train_batch']

    class ConfiguredNNUNetDataLoader(upstream):
        def generate_train_batch(self):
            # threadpoolctl alone does not update PyTorch's explicit thread
            # setting on every runtime. Serial calls restore the rank's setting;
            # persistent workers already have this value and avoid setters.
            torch = data_loader.torch
            previous = torch.get_num_threads()
            try:
                if previous != self.worker_threads: torch.set_num_threads(self.worker_threads)
                return adapted(self)
            finally:
                if previous != self.worker_threads: torch.set_num_threads(previous)

        def __init__(self, *args, worker_threads=1, **kwargs):
            if isinstance(worker_threads, bool) or not isinstance(worker_threads, int) or worker_threads < 1:
                raise ValueError('worker_threads must be a positive integer')
            self.worker_threads = worker_threads
            super().__init__(*args, **kwargs)

    ConfiguredNNUNetDataLoader.adapter_provenance = dict(
        nnunet_version=version, source_file=inspect.getsourcefile(upstream),
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        operation='single AST substitution: limits=1 -> limits=self.worker_threads; scoped torch threads')
    return ConfiguredNNUNetDataLoader
