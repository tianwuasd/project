"""Read verified Blosc2 patches, preserving roles and case supervision metadata.

No complete volume is decoded during batch generation. Each worker opens a
bounded case set and reads only the native crop needed for one augmented patch.
Official nnU-Net bbox/augmentation rules are reused; background ambiguity stays
in a separate per-case flag and is never converted into a fake voxel label.
"""
import copy
from contextlib import nullcontext
import json
from pathlib import Path
import pickle
import time
from types import SimpleNamespace
import numpy as np
import torch
from ..data.analysis import sha256_file
from ..preprocessing.annotations import make_annotation,validate_case_roles
from ..preprocessing.cohort import TARGETS,frozen_roles
from .contracts import digest

# Exact previously audited producers with the three-column coordinate bug.
# New/unknown producers cannot opt into the compatibility path by version text.
LEGACY_COORDINATE_PRODUCERS={
    '7e85e37bd3abbd81f9a9b3b00bdfb080fa398dab5444b5d109c301b45d81b718',
    '9456c36e31b7eb37f003e26b3aa951139d71ed26d3bf53c3dc1967e1004d09bb',
}


def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def cached_smoke_ids(run,fold,include_auxiliary=True):
    """Select cached role-correct examples; never borrow an outer test volume."""
    run=Path(run);manifest=read_json(run/'manifests/imagechd_input.json')
    rows=[r for r in manifest['cases'] if (run/f'fold_{fold}/completion'/f"{r['case_id']}.json").is_file()]
    main=[r['case_id'] for r in rows if r['roles'][str(fold)]=='train']
    aux=[r['case_id'] for r in rows if r['roles'][str(fold)]=='aux_train'] if include_auxiliary else []
    val=[r['case_id'] for r in rows if r['roles'][str(fold)]=='val']
    if not main or not val:raise ValueError('Smoke requires cached main train and main val cases; prepare these original roles first')
    has_aux=any(r['roles'][str(fold)]=='aux_train' for r in manifest['cases'])
    if include_auxiliary and has_aux and not aux:raise ValueError('Partial smoke requires one cached auxiliary case')
    return main[:1]+aux[:1],val[:1]


def normalize_locations(properties, shape, allow_legacy):
    """Return a copy with audited (channel,z,y,x) coordinates, without writes."""
    result=copy.copy(properties)
    locations={};adapted=False
    for key,values in properties['class_locations'].items():
        values=np.asarray(values)
        if not values.size:
            locations[key]=[];continue
        if values.ndim!=2 or not np.issubdtype(values.dtype,np.integer):
            raise ValueError('Malformed foreground coordinates')
        if values.shape[1]==3 and allow_legacy:
            values=np.column_stack((np.zeros(len(values),dtype=values.dtype),values));adapted=True
        if values.shape[1]!=4 or (values[:,0]!=0).any():
            raise ValueError('Expected channel_zyx coordinates; unrecognized legacy producer')
        if (values[:,1:]<0).any() or (values[:,1:]>=np.asarray(shape[1:])).any():
            raise ValueError('Foreground coordinate outside stored array')
        locations[key]=values.astype(np.int32,copy=False)
    result['class_locations']=locations
    result['class_location_coordinate_format']='channel_zyx'
    return result,adapted


def verify_coordinate_labels(seg,properties):
    """Check every saved point, decoding each touched 3D chunk at most once.

    Plane-at-a-time reads repeatedly decompress large 3D chunks. Group all
    classes' coordinates by the array's actual chunk grid instead; this keeps
    full verification while bounding memory to coordinates plus one chunk.
    """
    coordinate_groups=[];expected_groups=[]
    annotated_key=tuple([-1]+list(range(8)))
    for key,coords in properties['class_locations'].items():
        if not len(coords):continue
        if isinstance(key,int) and 1<=key<=7:expected=key
        elif key==annotated_key:expected=-2
        else:raise ValueError('Unexpected foreground sampling key')
        coordinate_groups.append(coords)
        expected_groups.append(np.full(len(coords),expected,dtype=np.int8))
    if not coordinate_groups:raise ValueError('No annotated sampling coordinates')
    coords=np.concatenate(coordinate_groups);expected=np.concatenate(expected_groups)
    shape=np.asarray(seg.shape[1:]);chunks=np.asarray(seg.chunks[1:])
    grid=(shape+chunks-1)//chunks
    chunk_ids=((coords[:,1]//chunks[0])*grid[1]+coords[:,2]//chunks[1])*grid[2]+coords[:,3]//chunks[2]
    order=np.argsort(chunk_ids,kind='stable');ids=chunk_ids[order]
    boundaries=np.flatnonzero(np.diff(ids))+1
    for indices in np.split(order,boundaries):
        origin=coords[indices[0],1:]//chunks*chunks;end=np.minimum(origin+chunks,shape)
        tile=seg[0,*[slice(int(a),int(b)) for a,b in zip(origin,end)]]
        local=coords[indices,1:]-origin
        actual=tile[local[:,0],local[:,1],local[:,2]];wanted=expected[indices]
        valid=np.where(wanted==-2,(actual>=-1)&(actual<=7),actual==wanted)
        if not bool(valid.all()):raise ValueError('Class-location/segmentation mismatch')


def load_prepared(run, fold, settings, *, train_cases=(), val_cases=(), group_audit=False):
    """Require role-correct complete caches; subset selection is smoke-only."""
    from ..observability.input_check import InputCheckProgress
    # The launcher first verifies the whole input. Each newly launched group
    # independently repeats that check, distributing the expensive volume reads.
    # Membership comes only from the live process group, never from a CLI flag.
    rank, world = 0, 1
    if group_audit:
        import torch.distributed as dist
        rank, world = dist.get_rank(), dist.get_world_size()
    label = f'fold{fold}' if world == 1 else f'fold{fold} rank{rank}/{world}'
    with InputCheckProgress(label) as progress:
        try:
            result = _load_prepared(run,fold,settings,train_cases=train_cases,val_cases=val_cases,
                                    progress=progress,audit_rank=rank,audit_world=world)
        except Exception as error:
            if world == 1:
                raise
            outcome = (None, f'rank {rank}: {type(error).__name__}: {error}')
        else:
            if world == 1:
                return result
            outcome = (digest(result['identity']), None)
        # Report an owner's error to the whole group before any rank uses the
        # input. Paths are fixed by the launcher and cases by the shared digest.
        progress.update('全组',1,world,'本路已返回，等待全组核验结果')
        outcomes = [None] * world
        dist.all_gather_object(outcomes, outcome)
        for identity, failure in outcomes:
            if failure:
                raise ValueError('Distributed input audit failed: '+failure)
        if any(identity != outcome[0] for identity, _ in outcomes):
            raise ValueError('Distributed input audit disagrees on the prepared dataset')
    return result


def _load_prepared(run,fold,settings,*,train_cases,val_cases,progress,audit_rank=0,audit_world=1):
    """The existing full verifier; progress never alters identities or files."""
    import blosc2
    run=Path(run).resolve(strict=True);foldroot=run/f'fold_{fold}'
    manifest=read_json(run/'manifests/imagechd_input.json')
    signature=read_json(run/'signature.json')
    plan=read_json(foldroot/'preprocessing_plan.json')
    dataset=read_json(run/'dataset.json')
    if dataset['labels']!=TARGETS or manifest['source_pairs']!=110:
        raise ValueError('Not the audited ImageCHD eight-output/ignore8 input')
    protocol=manifest['protocol']
    supported={f'imagechd-stunet-s-{body}-ct-v{v}'
               for body in ('clean96','main96-aux13') for v in (1,2)}
    if protocol not in supported:raise ValueError('Unknown preprocessing protocol')
    if signature['method']['protocol']!=protocol:raise ValueError('Manifest/producer protocol differs')
    legacy_hash=signature['producer_sha256']['src/chd_stunet/preprocessing/stunet_s.py']
    legacy_allowed=legacy_hash in LEGACY_COORDINATE_PRODUCERS
    cases={r['case_id']:r for r in manifest['cases']}
    all_rows=manifest['cases']+manifest['excluded']
    if len(all_rows)!=110 or len({r['case_id'] for r in all_rows})!=110:
        raise ValueError('Incomplete/duplicate case manifest')
    complete=[r['case_id'] for r in all_rows if not r.get('missing_label_ids')]
    if len(complete)!=97:raise ValueError('Historical complete cohort differs')
    historical=frozen_roles(complete,signature['method']['split_seed'])
    main=[r for r in manifest['cases'] if not r.get('missing_label_ids')]
    auxiliary=[r for r in manifest['cases'] if r.get('missing_label_ids')]
    if len(main)!=96 or len(auxiliary)!=(13 if 'main96-aux13' in protocol else 0):
        raise ValueError('Main/auxiliary cohort counts differ')
    if any(r['roles']!=historical[r['case_id']] for r in main):
        raise ValueError('Frozen main roles were changed')
    for row in auxiliary:
        if row['missing_label_ids']!=[5] or row['roles']!={str(f):'aux_train' for f in range(3)}:
            raise ValueError('Invalid auxiliary supervision/roles')
    expected_train=[cid for cid,r in cases.items() if r['roles'][str(fold)]=='train'
                    or (settings['include_auxiliary'] and r['roles'][str(fold)]=='aux_train')]
    expected_val=[cid for cid,r in cases.items() if r['roles'][str(fold)]=='val']
    if (train_cases or val_cases) and settings['scope']!='smoke':
        raise ValueError('Formal training must not silently shorten the cohort')
    chosen_train=list(train_cases or expected_train);chosen_val=list(val_cases or expected_val)
    if not chosen_train or not chosen_val or len(set(chosen_train))!=len(chosen_train) or len(set(chosen_val))!=len(chosen_val):
        raise ValueError('Nonempty unique training/validation cases are required')
    if set(chosen_train)-set(expected_train) or set(chosen_val)-set(expected_val):
        raise ValueError('Requested cache has wrong role (test/validation cannot enter training)')
    if set(chosen_train)&set(chosen_val):raise ValueError('Training/validation overlap')
    expected_fit=sorted(cid for cid,r in cases.items() if r['roles'][str(fold)]=='train')
    if sorted(plan['fit_ids'])!=expected_fit:
        raise ValueError('Preprocessing plan fitted outside the fixed main training set')
    rows={};adapters=[];identity=[]
    selected=chosen_train+chosen_val
    for index,cid in enumerate(selected):
        progress.update(cid,index,len(selected),'完成标记/源合同')
        source=cases[cid];completion_path=foldroot/'completion'/f'{cid}.json'
        if not completion_path.is_file():raise ValueError('Missing complete preprocessing cache: '+cid)
        completion=read_json(completion_path)
        if not completion.get('success') or completion['source_hashes']!=source['source_hashes']:
            raise ValueError('Incomplete or changed source cache: '+cid)
        expected_contract=digest(dict(run=signature['identity'],plan=plan))
        if completion['case_id']!=cid or completion['fold']!=fold or completion['contract']!=expected_contract:
            raise ValueError('Wrong fold or preprocessing plan cache: '+cid)
        required={cid+'.b2nd',cid+'_seg.b2nd',cid+'.pkl'}
        if set(completion['file_sha256'])!=required:raise ValueError('Unexpected cache files: '+cid)
        audit_volume = index % audit_world == audit_rank
        for name,sha in completion['file_sha256'].items():
            # Every reader verifies the properties it will unpickle, including
            # ranks that delegate the volume checks for this case to a peer.
            if not audit_volume and name != cid+'.pkl':
                continue
            progress.update(cid,index,len(selected),'完整文件SHA256 '+name)
            if sha256_file(foldroot/'data'/name)!=sha:raise ValueError('Cache SHA256 differs: '+name)
        with (foldroot/'data'/f'{cid}.pkl').open('rb') as stream:properties=pickle.load(stream)
        if properties.get('method_protocol')!=protocol or properties.get('source_hashes')!=source['source_hashes']:
            raise ValueError('Stored property provenance differs: '+cid)
        annotation=source.get('annotation')
        if annotation is None:
            if not legacy_allowed or 'clean96' not in protocol:
                raise ValueError('Missing annotation metadata for unrecognized cache')
            annotation=make_annotation([])
        if annotation!=make_annotation(source.get('missing_label_ids',[])):
            raise ValueError('Annotation availability differs from native audit: '+cid)
        group=source.get('cohort_group','main')
        validate_case_roles(annotation,group,source['roles'])
        for record in (completion,properties):
            for key,wanted in (('annotation',annotation),('cohort_group',group),('roles',source['roles'])):
                if key in record and record[key]!=wanted:raise ValueError('Case supervision differs: '+cid+' '+key)
                if key not in record and not legacy_allowed:raise ValueError('Case supervision missing: '+cid+' '+key)
        image=blosc2.open(str(foldroot/'data'/f'{cid}.b2nd'),mode='r',dparams={'nthreads':1})
        seg=blosc2.open(str(foldroot/'data'/f'{cid}_seg.b2nd'),mode='r',dparams={'nthreads':1})
        if image.shape!=seg.shape or image.shape[0]!=1 or image.dtype!=np.float32 or seg.dtype!=np.int8:
            raise ValueError('Unexpected prepared array format: '+cid)
        properties,adapted=normalize_locations(properties,image.shape,legacy_allowed)
        if audit_volume:
            progress.update(cid,index,len(selected),'按chunk核验全部前景坐标')
            verify_coordinate_labels(seg,properties)
        if adapted:adapters.append(dict(case_id=cid,adapter='audited_zyx_to_channel_zyx',producer=legacy_hash))
        rows[cid]=dict(case_id=cid,base=str(foldroot/'data'/cid),shape=list(image.shape),
                       properties=properties,annotation=annotation,cohort_group=group,roles=source['roles'])
        identity.append(dict(case_id=cid,source_hashes=completion['source_hashes'],
                             file_sha256=completion['file_sha256'],annotation=annotation,roles=source['roles'],
                             preprocessing_contract=completion['contract']))
        del image,seg
        progress.update(cid,index+1,len(selected),'病例核验完成')
    data_identity=dict(preprocessing_identity=signature['identity'],protocol=protocol,fold=fold,
        plan=plan,train_ids=chosen_train,val_ids=chosen_val,cases=identity,
        patient_identity_verified=manifest.get('patient_identity_verified',False),adapters=adapters)
    return dict(rows=rows,train_ids=chosen_train,val_ids=chosen_val,plan=plan,identity=data_identity)


def disable_wall_clock_benchmark(transform):
    """Wall-clock blur autotuning can break replay across newly spawned workers."""
    if type(transform).__name__=='GaussianBlurTransform':transform.benchmark=False
    for child in getattr(transform,'transforms',[]):disable_wall_clock_benchmark(child)
    child=getattr(transform,'transform',None)
    if child is not None:disable_wall_clock_benchmark(child)


class PatchSampler:
    """Picklable CPU-only loader; a supplied seed determines the entire batch."""
    def __init__(self, rows, ids, settings, topology, training=True, slots=None):
        self.rows={cid:rows[cid] for cid in ids};self.ids=list(ids)
        self.settings=settings;self.topology=topology;self.training=training
        self._transform=None
        self.read_gate=None
        self.slots=slots

    def __getstate__(self):
        state=dict(self.__dict__);state['_transform']=None
        return state

    def _configure(self):
        from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
        configuration=SimpleNamespace(patch_size=np.array(self.settings['patch_size']))
        fake=SimpleNamespace(configuration_manager=configuration,print_to_log_file=lambda *a:None)
        rotation,dummy,initial,mirror=nnUNetTrainer.configure_rotation_dummyDA_mirroring_and_inital_patch_size(fake)
        scales=self.topology['deep_supervision_scales']
        if self.training:
            self._transform=nnUNetTrainer.get_training_transforms(self.settings['patch_size'],rotation,scales,
                mirror,dummy,use_mask_for_norm=[False],ignore_label=8)
            disable_wall_clock_benchmark(self._transform)
            self._initial=np.asarray(initial)
        else:
            self._transform=nnUNetTrainer.get_validation_transforms(scales,ignore_label=8)
            self._initial=np.asarray(self.settings['patch_size'])

    def __call__(self, seed):
        from ..acceleration.training_cpu import deterministic_cpu_rng,sequence_seed
        if self.slots is not None:
            # Global slot seeds survive rank/worker changes. Each rank reads
            # only its own crops; no duplicate full-global-batch augmentation.
            parts=[]
            for slot in self.slots:
                with deterministic_cpu_rng(sequence_seed(seed,'global-slot',0,slot)):
                    parts.append(self._sample([slot]))
            if len(parts) == 1:
                # The sample already includes its batch dimension. With one
                # slot per GPU, concatenation would copy every patch and target.
                return parts[0]
            return dict(data=np.concatenate([p['data'] for p in parts]),
                target=[np.concatenate([p['target'][i] for p in parts]) for i in range(len(parts[0]['target']))],
                available=np.concatenate([p['available'] for p in parts]),
                cases=[case for p in parts for case in p['cases']],
                supply_stages={name:sum(p['supply_stages'][name] for p in parts)
                               for name in parts[0]['supply_stages']})
        with deterministic_cpu_rng(seed):return self._sample()

    def _sample(self,slots=None):
        import blosc2
        from ..observability.locking import read_slot
        from acvl_utils.cropping_and_padding.bounding_boxes import crop_and_pad_nd
        from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
        if self._transform is None:self._configure()
        batch=[];targets=[];metadata=[];available=[]
        stages = dict(read_gate=0., read_decode_crop=0., augmentation=0.)
        batch_size=self.settings['batch_size']
        for slot in range(batch_size) if slots is None else slots:
            cid=self.ids[np.random.randint(len(self.ids))];row=self.rows[cid]
            force_fg=slot>=round(batch_size*(1-self.settings['oversample_foreground']))
            bbox_state=SimpleNamespace(patch_size=self._initial,
                need_to_pad=self._initial-np.asarray(self.settings['patch_size']),
                has_ignore=True,annotated_classes_key=tuple([-1]+list(range(8))))
            lo,hi=nnUNetDataLoader.get_bbox(bbox_state,row['shape'][1:],force_fg,row['properties']['class_locations'])
            bbox=[[int(lower),int(upper)] for lower,upper in zip(lo,hi)]
            queued = time.monotonic()
            with read_slot(self.read_gate) if self.read_gate else nullcontext():
                reading = time.monotonic()
                stages['read_gate'] += reading - queued
                image=blosc2.open(row['base']+'.b2nd',mode='r',dparams={'nthreads':1})
                seg=blosc2.open(row['base']+'_seg.b2nd',mode='r',dparams={'nthreads':1})
                data=torch.from_numpy(crop_and_pad_nd(image,bbox,0)).float()
                labels=torch.from_numpy(crop_and_pad_nd(seg,bbox,-1,cast_cropped_to=np.int16)).to(torch.int16)
                del image,seg
                stages['read_decode_crop'] += time.monotonic() - reading
            transforming = time.monotonic()
            transformed=self._transform(image=data,segmentation=labels)
            stages['augmentation'] += time.monotonic() - transforming
            if not bool(torch.isfinite(transformed['image']).all()):
                raise ValueError('Non-finite augmented image patch: '+cid)
            batch.append(transformed['image']);targets.append(transformed['segmentation'])
            available.append([k in row['annotation']['annotated_class_ids'] for k in range(1,8)])
            metadata.append(dict(case_id=cid,cohort_group=row['cohort_group'],force_foreground=force_fg,
                                 bbox=[[int(v) for v in lo],[int(v) for v in hi]]))
        return dict(data=torch.stack(batch).numpy(),
            target=[torch.stack([target[level] for target in targets]).numpy() for level in range(len(targets[0]))],
            available=np.asarray(available,dtype=bool),cases=metadata,supply_stages=stages)
