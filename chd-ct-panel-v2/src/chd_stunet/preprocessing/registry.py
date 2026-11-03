"""Method registry: expose only implemented, versioned scientific contracts.

The CLI/scheduler know no label rules. A future model may register another
config and adapter here without changing the current method's cache identity.
"""
import json
import importlib
from ..config import ROOT

CONFIGS = {name: ROOT / 'configs/preprocessing' / (name + '.json')
           for name in ('stunet-s', 'stunet-s-partial')}
ADAPTERS = {name: 'chd_stunet.preprocessing.stunet_s' for name in CONFIGS}


def get_method(name):
    if name not in CONFIGS:
        raise ValueError('Unknown preprocessing method: ' + name)
    return json.loads(CONFIGS[name].read_text(encoding='utf-8'))


def list_methods():
    return [get_method(name) for name in CONFIGS]


def get_adapter(name):
    """Resolve method leaves only when execution begins, not during CLI help."""
    if name not in CONFIGS or name not in ADAPTERS:
        raise ValueError('Unknown/unimplemented preprocessing adapter: '+name)
    return importlib.import_module(ADAPTERS[name])
