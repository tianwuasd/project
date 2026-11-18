from ..data.dataset import CaseDataset
from ..data.input_binding import medsam_case


class ModelCases(CaseDataset):
    def __init__(self, prepared, config, *, with_label=True, resources=None):
        self.model_config = config
        super().__init__(prepared, resources or dict(cache_bytes=0), with_label=with_label)

    def get(self, name):
        entry = super().get(name)
        if self.model_config['model'] == 'medsam2':
            entry = medsam_case(entry, self.model_config.get('input_source', {}))
        return entry
