from collections import Counter,defaultdict
from dataclasses import dataclass
import copy
import math
import numpy as np


@dataclass(frozen=True)
class CandidateRecord:
    case: str
    patient: str
    source: str
    candidate: int
    has_error: bool
    correct_key_structure: bool
    contains_correct_anomaly: bool
    supervised_voxels: int

    @property
    def layer(self):
        return 0 if self.has_error else 1 if self.correct_key_structure else 2


class CandidateSampler:
    def __init__(self, records, seed=20261111, layer_weights=(.5,.25,.25)):
        if any(w <= 0 for w in layer_weights) or len(layer_weights) != 3:
            raise ValueError('候选层需要三个正权重')
        self.rng = np.random.default_rng(seed)
        self.layer_weights = np.asarray(layer_weights,dtype=float)
        self.records = list(records)
        if not self.records or any(r.supervised_voxels <= 0 for r in self.records):
            raise ValueError('训练候选必须具有可靠监督')
        if len({(r.case,r.candidate) for r in self.records}) != len(self.records):
            raise ValueError('同一候选不能重复进入抽样层')
        self.sources = defaultdict(lambda:defaultdict(list))
        patient_sources = defaultdict(set)
        for r in self.records:
            self.sources[r.source][r.patient].append(r)
            patient_sources[r.patient].add(r.source)
        if any(len(v) != 1 for v in patient_sources.values()):
            raise ValueError('跨来源重复患者需先归并到唯一抽样来源')
        self.source_names = sorted(self.sources)
        weights = np.asarray([math.sqrt(len(self.sources[s])) for s in self.source_names])
        self.source_probabilities = weights/weights.sum()
        self.counts = Counter()
        self.identity = [(r.__dict__) for r in self.records]

    def draw(self):
        source = self.source_names[self.rng.choice(len(self.source_names),p=self.source_probabilities)]
        patients = sorted(self.sources[source])
        patient = patients[self.rng.integers(len(patients))]
        records = self.sources[source][patient]
        layers = [[r for r in records if r.layer == i] for i in range(3)]
        weights = self.layer_weights*np.asarray([bool(v) for v in layers])
        layer = int(self.rng.choice(3,p=weights/weights.sum()))
        pool = layers[layer]
        if layer == 1:
            anomaly = [r for r in pool if r.contains_correct_anomaly]
            ordinary = [r for r in pool if not r.contains_correct_anomaly]
            if anomaly and ordinary:
                pool = anomaly if self.rng.random() < .5 else ordinary
        chosen = pool[self.rng.integers(len(pool))]
        self.counts.update({'draws':1,'source:'+source:1,'layer:'+str(layer):1,
                            'correct_anomaly':int(chosen.contains_correct_anomaly)})
        return chosen

    def state_dict(self):
        return dict(identity=self.identity,layer_weights=self.layer_weights.tolist(),
                    rng=copy.deepcopy(self.rng.bit_generator.state),counts=dict(self.counts))

    def load_state_dict(self, state):
        if state['identity'] != self.identity or state['layer_weights'] != self.layer_weights.tolist():
            raise ValueError('候选抽样检查点来源或权重不一致')
        self.rng.bit_generator.state = copy.deepcopy(state['rng'])
        self.counts = Counter(state['counts'])

    def report(self):
        return dict(source_patients={s:len(self.sources[s]) for s in self.source_names},
                    source_probabilities=dict(zip(self.source_names,self.source_probabilities.tolist())),
                    candidate_layers={str(i):sum(r.layer==i for r in self.records) for i in range(3)},
                    empty_patient_layers={str(i):sum(not any(r.layer==i for r in rows)
                        for patients in self.sources.values() for rows in patients.values()) for i in range(3)},
                    actual_counts=dict(self.counts))

