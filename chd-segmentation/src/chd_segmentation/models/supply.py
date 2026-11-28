from ..acceleration.model_supply import CpuReady, OrderedSupply
from ..data.dataset import CaseDataset


class AdmittedCases(CaseDataset):
    def __init__(self, *args, rank, gate, device=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.rank, self.gate, self.device = rank, gate, device
        self.position, self.mode = 0, 'candidates'

    def get(self, name):
        gate, rank = self.gate, self.rank
        if self.mode == 'candidates':
            gate.ready.wait(rank, f'q-candidate:{self.position}:{name}', self.device)
        with gate.build(rank, gate.batch.value, self.position, ordered=self.mode not in ('validation','candidate_shard')):
            gate.reserve(rank, name)
            entry = super().get(name)
        gate.mark_ready(rank)
        gate.deliver(rank)
        self.position += 1
        return entry
