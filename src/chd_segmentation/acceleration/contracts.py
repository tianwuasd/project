from dataclasses import dataclass, field
from typing import Any


@dataclass
class ResourceDemand:
    fixed_bytes: int = 0
    workspace_bytes: int = 0
    sample_bytes: int = 0
    batch_buffer_bytes: int = 0
    cache_bytes: int = 0
    cpu_slots: int = 1
    gpu_count: int = 0
    gpu_ids: Any = None
    local_batches: Any = None

    def __post_init__(self):
        for name in ('fixed_bytes','workspace_bytes','sample_bytes','batch_buffer_bytes','cache_bytes'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(name+'必须为非负整数')
        if type(self.cpu_slots) is not int or self.cpu_slots < 1:
            raise ValueError('cpu_slots必须为正整数')
        if type(self.gpu_count) is not int or not 0 <= self.gpu_count <= 4:
            raise ValueError('gpu_count必须在0至4之间')
        if self.local_batches is not None and (not self.local_batches or
                any(type(value) is not int or value < 1 for value in self.local_batches)):
            raise ValueError('local_batches必须为非空正整数份额')

    @property
    def memory_bytes(self):
        return sum((self.fixed_bytes, self.workspace_bytes, self.sample_bytes,
                    self.batch_buffer_bytes, self.cache_bytes))


@dataclass
class RuntimeRequest:
    task_id: str
    stage: str
    registry_root: Any
    run_dir: Any
    resources: dict
    demand: ResourceDemand


@dataclass
class ResourceHandle:
    owner: dict
    condition: Any
    state: Any
    cancel: Any
    resources: dict
    members: Any


@dataclass
class ResourceSnapshot:
    sampled_at: str
    sampled_monotonic: float
    sample_age_seconds: float
    host: dict
    project_accounted_bytes: int
    headroom_bytes: int
    paused: bool
    cancelled: bool
    sampling_seconds: float
    accounting_method: str = 'rss_sum'


@dataclass
class Allocation:
    owner: dict
    gpus: list
    workers: list
    reader_slots: int
    prefetch: int
    cpu_threads: int
    memory_bytes: int
    requested: dict = field(default_factory=dict)
    applied: dict = field(default_factory=dict)


@dataclass
class Reservation:
    amount_bytes: int
    purpose: str


@dataclass
class BoundaryDecision:
    action: str
    reason: str
    waited_seconds: float = 0.


@dataclass
class CaseEstimate:
    case_id: str
    whole_peak_bytes: int
    fixed_bytes: int
    minimum_block_bytes: int
    output_shape: Any


@dataclass
class CasePolicy:
    execution: str = 'auto'
    block_bytes: Any = None
    max_block_bytes: Any = None
    max_threads: int = 1


@dataclass
class CaseExecution:
    mode: str
    reason: str
    workspace_bytes: int
    block_bytes: int
    concurrency: int


@dataclass
class BlockJob:
    selection: Any
    additional_bytes: int
    voxels: int


@dataclass
class SafeBoundary:
    kind: str
    consumed_index: int = 0
    checkpoint_path: Any = None
