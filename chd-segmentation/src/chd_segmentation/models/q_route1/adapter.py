from dataclasses import asdict
from contextlib import nullcontext
from pathlib import Path

from monai.inferers import sliding_window_inference
import numpy as np
import torch
import torch.nn.functional as F

from ..stunet.adapter import StunetAdapter
from ...data.dataset import sample_patch
from ...observability.events import event
from ...training.resources import host_capacity
from ...training.checkpoint import tensors_equal
from .events import CLASSES
from .network import LocalNetwork
from .localizer import Localizer
from .losses import segmentation_loss,event_loss
from .targets import build_heatmap_targets,localizer_loss
from .geometry import crop_local_input,sample_context
from .candidates import select_candidates
from .comparisons import infer_local
from .fusion import ProbabilityFusion,core_weight
from .sampling import CandidateRecord,CandidateSampler
from .corruption import controlled_corruption


MAPPING = {'chd1282':(0,1,2,3,4,6,7,5,9,10,8),'imagechd':tuple(range(8))}
SOLVER = dict(steps=3,gamma=1.,lambda_=0.5,eta=1/3)


def allowed(label, missing, dataset, parallel=None):
    mapping = np.asarray(MAPPING[dataset])
    if not ((label>=-1)&(label<len(mapping))).all():
        raise ValueError('Q来源标签超出对应数据集范围')
    mapped = mapping[label.clip(0)]
    if parallel is None:
        support = np.stack([mapped==index for index in range(11)])
    else:
        support = np.empty((11,*label.shape),dtype=bool)
        parallel.map(range(11),int(label.size*3),lambda index:mapped==index,
                     lambda index,value:support.__setitem__(index,value))
    unknown = set(map(int,mapping[list(missing)])) | (set(range(11))-set(map(int,mapping)))
    for index in unknown:
        support[index] |= label==0
    support[:,label==-1] = True
    return support


def export_probabilities(probability,dataset):
    mapping = MAPPING[dataset]
    result = probability[list(mapping)].copy()
    excluded = sorted(set(range(11))-set(mapping))
    if excluded:
        result[0] += probability[excluded].sum(0)
    return result


class QAdapter:
    def __init__(self,config,prepared,stage,initialize=True,parent_states=None):
        if stage not in ('base','localizer','local'):
            raise ValueError('Q阶段必须为base、localizer或local')
        self.config,self.prepared,self.stage = config,prepared,stage
        self.device = torch.device(config['device'])
        self.sampler = None
        self.corruption_rng = np.random.default_rng(config['seed']+1)
        self.current_name,self.current_frozen = None,None
        self.picked = None
        self.parent_models = None
        patch = config['patch']
        spec = asdict(StunetAdapter.spec({'training':{'patch':[patch]*3},'dataset':prepared['dataset']},prepared['plan']))
        spec['num_classes'] = 11
        self.spec = spec
        if stage=='base':
            self.model = StunetAdapter.build(spec)
            if config['pretrained'] and initialize:
                StunetAdapter.load_pretrained(self.model,config['pretrained'])
            elif initialize and not config['technical']:
                raise ValueError('Q正式基座训练需要官方预训练权重')
        elif stage=='localizer':
            self.model = Localizer()
        else:
            from ..workflow import load_state,checked_identity,training_names
            if set(config['parents'])!={'base','localizer'}:
                raise ValueError('Q局部模型需要base与localizer检查点')
            parents = ({key:load_state(value) for key,value in config['parents'].items()}
                if parent_states is None else parent_states)
            if set(parents)!={'base','localizer'}:
                raise ValueError('Q固定上游参数必须同时包含base和localizer')
            for key,state in parents.items():
                if state['stage']!=key or state['prepared']['dataset']!=prepared['dataset']:
                    raise ValueError('Q上游阶段或数据集不匹配')
                if state['scope']!=prepared['scope']:
                    raise ValueError('Q上游范围不匹配')
                parent_config = state['config']
                if parent_config['model']!='q_route1' or state['training_cases']!=training_names(parent_config,prepared,key):
                    raise ValueError('Q上游实际训练名单与配置不同')
                checked_identity(parent_config,prepared,key,state['initialization_sha256'],state['identity'])
                if not config['technical']:
                    import yaml
                    groups = yaml.safe_load(Path(config['groups']).read_text(encoding='utf-8'))
                    if not set(state['training_cases'])<=set(groups['A']):
                        raise ValueError('Q上游使用了A组以外的训练患者')
                    selected = state['history'][-1].get('validation_loss')
                    if selected is None or not np.isfinite(selected) or selected!=state['best']:
                        raise ValueError('Q上游需要由C组选择的最佳完整检查点')
            retained = ('model','spec','stage','prepared','scope','config','training_cases',
                'initialization_sha256','identity','best')
            self.parent_models = {key:{**{name:state[name] for name in retained},
                'history':state['history'][-1:]} for key,state in parents.items()}
            self.base = StunetAdapter.build(parents['base']['spec']).to(self.device).eval().requires_grad_(False)
            self.base.load_state_dict(parents['base']['model'],strict=True)
            self.locator = Localizer().to(self.device).eval().requires_grad_(False)
            self.locator.load_state_dict(parents['localizer']['model'],strict=True)
            self.model = LocalNetwork(method=config['method'])
        self.model.to(self.device)
        from .training_cache import TrainingCache
        import hashlib
        parents_key = None
        if self.parent_models is not None:
            checksum = hashlib.sha256()
            for stage_name,state in sorted(self.parent_models.items()):
                checksum.update(stage_name.encode())
                for name,value in sorted(state['model'].items()):
                    checksum.update(name.encode())
                    checksum.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
            parents_key = checksum.hexdigest()
        self.training_cache = TrainingCache(config,prepared,stage,parents_key)

    def optimizer(self):
        groups = []
        for name,parameter in self.model.named_parameters():
            lr = self.config['learning_rate']
            if self.stage=='base' and not name.startswith('seg_outputs.'):
                lr *= .5
            groups.append(dict(params=[parameter],lr=lr,weight_decay=1e-4 if parameter.ndim>1 and name.endswith('weight') else 0.))
        return torch.optim.AdamW(groups,betas=(.9,.999),eps=1e-8)

    def extra_state(self):
        return dict(sampler=self.sampler.state_dict() if self.sampler is not None else None,
                    corruption_rng=self.corruption_rng.bit_generator.state)

    def restore_extra(self,state):
        self.corruption_rng.bit_generator.state = state['corruption_rng']
        if state['sampler'] is not None:
            if self.sampler is None:
                raise ValueError('Q恢复前需要重建同一候选索引')
            self.sampler.load_state_dict(state['sampler'])

    def schedule(self,optimizer,update):
        warmup = 1000 if self.stage=='base' else 500
        total = {'base':40000,'localizer':5000,'local':20000}[self.stage] if self.config['technical'] else self.config['updates']
        if total<=warmup:
            warmup = max(1,total//10)
        factor = (update+1)/warmup if update<warmup else .01+.99*(1+np.cos(np.pi*(update-warmup)/(total-warmup)))/2
        for group in optimizer.param_groups:
            if 'peak_lr' not in group:
                group['peak_lr'] = group['lr']
            group['lr'] = group['peak_lr']*factor

    def crop(self,entry,candidate,probability,parallel=True):
        size = self.config['local_size']
        start = tuple(int(np.floor(c-(size-1)/2+.5)) for c in candidate.center)
        image = entry['image']
        inputs,editable = crop_local_input(image,probability,start,size)
        grids = np.meshgrid(*(np.arange(s,s+size) for s in start),indexing='ij')
        inside = np.ones((size,)*3,dtype=bool)
        for grid,n in zip(grids,entry['label'].shape):
            inside &= (grid>=0)&(grid<n)
        indices = tuple(grid.clip(0,n-1) for grid,n in zip(grids,entry['label'].shape))
        raw = np.where(inside,entry['label'][indices],-1)
        from .preparation import preparation
        guard = getattr(self,'training_guard',None) if parallel else None
        with preparation(self.config,guard,entry,stage=self.stage) as runtime:
            support = allowed(raw,entry['unknown_classes'],self.prepared['dataset'],runtime)
        return inputs,editable,support

    @torch.no_grad()
    def locate(self,entry):
        from .training_inputs import pooled_image

        settings = self.config['execution']
        if self.device.type == 'cuda':
            count = int(np.prod([(n+31)//32*8 for n in entry['image'].shape[1:]]))
            free,total = torch.cuda.mem_get_info(self.device)
            reserve = max(settings.get('vram_reserve_bytes',4*2**30),
                          total*settings.get('vram_reserve_fraction',.1))
            if count*512+128*2**20 > free-reserve:
                raise MemoryError('定位器粗网格前向超过当前显存余量')
        guard = getattr(self,'training_guard',None)
        count = int(np.prod([(n+31)//32*8 for n in entry['image'].shape[1:]]))
        amount = count*(32 if self.device.type=='cuda' else 512)
        claim = guard.admit(amount,label='q_local_locator_retained') if guard is not None else nullcontext(True)
        with claim as ticket:
            if ticket is None:
                raise InterruptedError('局部定位输入等待已取消')
            from .preparation import preparation
            with preparation(self.config,guard,entry,stage=self.stage) as runtime:
                image,valid = pooled_image(entry,settings.get('localizer_block_bytes',2*2**30),guard=guard,parallel=runtime)
            return self.locator.forward_pooled(image.to(self.device),valid.to(self.device))

    @torch.no_grad()
    def frozen(self,entry):
        if entry['case_id']==self.current_name:
            return self.current_frozen
        self.current_name,self.current_frozen = None,None
        from .candidates import Candidate
        def produce():
            probability = self.base_probability(entry)
            located = self.locate(entry)
            candidates = select_candidates(located['heatmap'][0].float().cpu().numpy(),entry['label'].shape,[4,4,8],.2,.8)
            return dict(probability=probability,features=located['features'].float().cpu(),
                        candidates=[asdict(value) for value in candidates])
        amount = int(entry['image'].size*48)+2**20
        guard = getattr(self,'training_guard',None)
        retained = guard.admit(amount,label='q_frozen_cache_retained') if guard is not None else nullcontext(True)
        with retained as ticket:
            if ticket is None:
                raise InterruptedError('冻结上游缓存内存等待已取消')
            value = self.training_cache.get(entry,produce,amount)
        probability,features = value['probability'],value['features']
        candidates = [Candidate(**value) for value in value['candidates']]
        self.current_name = entry['case_id']
        self.current_frozen = probability,features,candidates
        return self.current_frozen

    def prepare_training(self,dataset,names):
        if self.stage!='local':
            return
        import yaml
        references = yaml.safe_load(Path(self.config['key_references']).read_text(encoding='utf-8')) if self.config['key_references'] else {}
        records = []
        rows = {row['case_id']:row for row in self.prepared['manifest']['cases']}
        for completed,name in enumerate(names,1):
            entry = dataset.get(name)
            probability,features,candidates = self.frozen(entry)
            def inspect(index):
                candidate = candidates[index]
                inputs,editable,support = self.crop(entry,candidate,probability,parallel=False)
                exact = (support.sum(0)==1)&editable[0,0].numpy()
                supervised = (support.sum(0)<11)&editable[0,0].numpy()
                if not supervised.any():
                    return None
                has_error = bool(((inputs[0,1:12].argmax(0).numpy()!=support.argmax(0))&exact).any())
                reference = references.get(name,{}).get(str(index),{})
                if reference and reference['candidate_start']!=list(candidate.start):
                    raise ValueError('Q关键结构参考没有绑定当前自动候选')
                return CandidateRecord(name,rows[name].get('patient') or name,self.prepared['dataset'],index,
                    has_error,bool(reference.get('correct_key_structure',False)),
                    bool(reference.get('contains_correct_anomaly',False)),int(supervised.sum()))
            from .preparation import preparation
            with preparation(self.config,getattr(self,'training_guard',None),entry,stage=self.stage) as runtime:
                if runtime is None:
                    records.extend(value for index in range(len(candidates)) if (value:=inspect(index)) is not None)
                else:
                    def consume(index,value):
                        if value is not None:
                            records.append(value)
                    runtime.map(range(len(candidates)),self.config['local_size']**3*192,inspect,consume)
            if self.config['logging']['case_details']:
                event(Path(self.config['run_dir'])/self.stage,'q_candidates_completed',case_id=name,
                    cases_completed=completed,cases_total=len(names),candidates=len(candidates),records=len(records))
            del probability,features
        self.sampler = CandidateSampler(records,self.config['seed'])

    def sample(self,dataset,names,rng):
        if self.stage!='local':
            patients = {}
            for row in self.prepared['manifest']['cases']:
                if row['case_id'] in names:
                    patients.setdefault(row.get('patient') or row['case_id'],[]).append(row['case_id'])
            group = patients[sorted(patients)[int(rng.integers(len(patients)))]]
            return dataset.get(group[int(rng.integers(len(group)))])
        self.picked = self.sampler.draw()
        return dataset.get(self.picked.case)

    def base_probability(self,entry):
        required = int(entry['image'].size*(11*24+16))
        reserve = self.config['execution'].get('reserve_ram_bytes',4*2**30)
        if host_capacity()['available_bytes'] < reserve+required:
            raise MemoryError('Q整例概率与融合工作区超出主机内存预算')
        model = self.model if self.stage=='base' else self.base
        with torch.no_grad():
            logits = sliding_window_inference(torch.from_numpy(np.asarray(entry['image']))[None],
                roi_size=tuple(self.spec['patch']),sw_batch_size=1,
                predictor=lambda x:model(x)[0],sw_device=self.device,device='cpu',overlap=.5,mode='gaussian')
        return logits.float().softmax(1)

    def loss(self,entry,rng,update):
        size = self.config['patch']
        image,label = sample_patch(entry,[size]*3,rng,.33)
        if self.stage=='base':
            x = torch.from_numpy(image)[None].to(self.device)
            support = allowed(label,entry['unknown_classes'],self.prepared['dataset'])
            valid = torch.from_numpy(label>=0)[None,None].to(self.device)
            target = torch.from_numpy(support)[None].to(self.device)
            outputs = self.model(x)
            weights = np.asarray([.5**i for i in range(len(outputs))]);weights/=weights.sum()
            return sum(float(w)*segmentation_loss(value.float(),F.interpolate(target.float(),size=value.shape[2:],mode='nearest').bool(),
                F.interpolate(valid.float(),size=value.shape[2:],mode='nearest').bool()) for w,value in zip(weights,outputs))
        if self.stage=='localizer':
            from .training_inputs import prepare_inputs,training_plan
            settings = self.config['execution']
            block = settings.get('localizer_block_bytes',2*2**30)
            plan = training_plan(entry['label'].shape,entry['processed_spacing'],settings)
            if self.device.type == 'cuda':
                free,total = torch.cuda.mem_get_info(self.device)
                reserve = max(settings.get('vram_reserve_bytes',4*2**30),
                              total*settings.get('vram_reserve_fraction',.1))
                event(Path(self.config['run_dir'])/self.stage,'q_vram_estimate',case_id=entry['case_id'],
                    requested_bytes=plan['device_workspace_bytes'],free_bytes=free,total_bytes=total,
                    reserved_bytes=torch.cuda.memory_reserved(self.device),
                    allocated_bytes=torch.cuda.memory_allocated(self.device),reserve_bytes=int(reserve),
                    exceeds_estimate=plan['device_workspace_bytes']>free-reserve,advisory_only=True)
            def produce():
                from .preparation import preparation
                guard = getattr(self,'training_guard',None)
                with preparation(self.config,guard,entry,stage=self.stage) as runtime:
                    return prepare_inputs(entry,self.prepared['dataset'],block,
                        guard=guard,log_dir=Path(self.config['run_dir'])/self.stage,
                        heartbeat_seconds=self.config['logging']['heartbeat_seconds'],parallel=runtime)
            x,valid,heat,mask = self.training_cache.get(entry,produce,
                int(np.prod(plan['pooled_shape'])*23)+65536)
            if not mask.any():
                raise ValueError('定位器输入缺少可监督区域')
            if getattr(self,'low_vram',False):
                from .low_vram import OffloadedLocalizer
                guard = getattr(self,'training_guard',None)
                logits = OffloadedLocalizer(self.model,self.low_vram_block_bytes,
                    guard.check_active if guard is not None else None).forward(x,valid)
                return localizer_loss(logits,heat,mask)
            saved = torch.autograd.graph.save_on_cpu(pin_memory=False) if self.device.type=='cuda' else nullcontext()
            with saved:
                logits = self.model.forward_pooled(x.to(self.device),valid.to(self.device))['logits'].float()
                return localizer_loss(logits,heat.to(self.device),mask.to(self.device))
        with torch.no_grad():
            probability,features,candidates = self.frozen(entry)
            if self.picked is None or self.picked.case!=entry['case_id']:
                raise ValueError('Q局部训练缺少分层抽样结果')
            candidate = candidates[self.picked.candidate]
            size = self.config['local_size']
            inputs,editable,support = self.crop(entry,candidate,probability)
            corrupted,self.last_corruption = controlled_corruption(inputs[0,1:12].numpy(),support,
                editable[0,0].numpy(),self.corruption_rng,chance=.2)
            inputs[:,1:12] = torch.from_numpy(corrupted)[None]
            context = sample_context(features,candidate.center,samples=size//8)
            target = torch.from_numpy(support)[None].to(self.device)
        editable = editable.to(self.device)
        out = self.model(inputs.to(self.device),context.to(self.device),editable)
        core = (slice(None),slice(None),*out['core'])
        solver_steps = 3 if self.config['technical'] or update>=self.config['warmup_updates'] else 0
        final,_ = infer_local(out,editable,self.config['method'],SOLVER,checkpointing=True,steps=solver_steps)
        loss = segmentation_loss(final,target[core],editable[core])+.5*segmentation_loss(out['z0'][core],target[core],editable[core])
        return loss+event_loss(out['q_logits'],target[core],editable[core])

    def predict(self,entry,prompts=None,internal=False):
        if self.stage!='local':
            raise ValueError('Q完整预测需要local阶段检查点')
        image = entry['image']
        probability = self.base_probability(entry)
        located = self.locate(entry)
        candidates = select_candidates(located['heatmap'][0].float().cpu().numpy(),image.shape[1:],[4,4,8],.2,.8)
        candidates = candidates[:self.config['max_candidates']]
        fusion = ProbabilityFusion(probability)
        diagnostics = []
        size = self.config['local_size']; margin=size//8; core_size=size-2*margin
        for candidate in candidates:
            start = tuple(int(np.floor(c-(size-1)/2+.5)) for c in candidate.center)
            inputs,editable = crop_local_input(image,probability,start,size)
            context = sample_context(located['features'].float().cpu(),candidate.center,samples=size//8)
            out = self.model(inputs.to(self.device),context.to(self.device),editable.to(self.device))
            final,trace = infer_local(out,editable.to(self.device),self.config['method'],SOLVER)
            core = (slice(None),slice(None),*out['core'])
            fusion.add(final.softmax(1).cpu(),tuple(s+margin for s in start),
                       core_weight(core_size,width=min(8,core_size//3))*editable[core][0,0])
            diagnostics.append(dict(energy=trace.energy_F.cpu().tolist() if trace is not None else [],start=list(start)))
        result = fusion.finish()[0].numpy()
        return (result if internal else export_probabilities(result,self.prepared['dataset'])),dict(candidates=len(candidates),solver=SOLVER,rois=diagnostics)

    def validate(self,dataset,names):
        from .validation import validate
        return validate(self,dataset,names)
