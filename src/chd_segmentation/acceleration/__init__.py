from .contracts import (RuntimeRequest, ResourceDemand, ResourceHandle, ResourceSnapshot,
                        Allocation, Reservation, BoundaryDecision, CaseEstimate,
                        CasePolicy, CaseExecution, BlockJob, SafeBoundary)
from importlib import import_module


_EXPORTS = {
    'ResourceClient': 'client', 'ResourceSession': 'runtime', 'attach': 'runtime',
    'choose_execution': 'cpu', 'OrderedSupply': 'supply', 'HostCoordinator': 'sync',
    'transfer_batch': 'transfer', 'TransferHandle': 'transfer',
    'TimingBook': 'timing', 'BlockExecutor': 'blocks',
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module('.'+_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value
