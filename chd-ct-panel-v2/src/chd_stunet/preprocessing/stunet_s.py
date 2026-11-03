"""STU-Net-S preprocessing leaves: audit, native nnU-Net transforms, write QC.

Every write targets a derived run. Raw images, labels and headers are read-only.
The executor calls these same leaves for real short tests and full processing.
"""
import json
import os
from pathlib import Path
import pickle
import time
import numpy as np
import nibabel as nib
from ..data.analysis import sha256_file
from ..observability.locking import read_slot
from ..observability.runlog import write_json
from .cohort import map_labels
from .annotations import make_annotation, validate_resampled_labels, validate_case_roles
from .fingerprint import sample_foreground


def hashes(pair):
    return {k:sha256_file(pair[k]) for k in ('image','label')}


def load_pair(job):
    """Hold a read permit through hash + decompression; release before compute."""
    pair = job['pair']
    with read_slot(job['gate']):
        before = hashes(pair)
        for kind,expected in job.get('expected_hashes',{}).items():
            if expected and before[kind] != expected:
                raise ValueError('Source SHA256 changed: '+pair['case_id']+' '+kind)
        image, label = nib.load(pair['image']), nib.load(pair['label'])
        if image.ndim != 3 or image.shape != label.shape:
            raise ValueError('Image/label shape mismatch or non-3D data')
        if not np.allclose(image.affine,label.affine,rtol=0,atol=1e-5):
            raise ValueError('Image/label affine mismatch; no guessed repair permitted')
        data = image.get_fdata(dtype=np.float32)
        segmentation = np.asanyarray(label.dataobj)
        affine = image.affine.copy()
        header_spacing = list(map(float,image.header.get_zooms()[:3]))
        header_units = image.header.get_xyzt_units()[0]
    if not np.isfinite(data).all():
        raise ValueError('CT has NaN/Inf; this method has no automatic intensity repair')
    mapped = map_labels(segmentation,job['method']['allowed_extra_labels'])
    return data,mapped,segmentation,affine,header_spacing,header_units,before


def check_source_unchanged(job, before):
    with read_slot(job['gate']):
        after = hashes(job['pair'])
    if after != before:
        raise ValueError('Source changed while processing; result cannot be committed')


def audit_case(job):
    """Inspect every source pair before freezing cohorts and main-train fit IDs."""
    from nnunetv2.preprocessing.cropping.cropping import crop_to_nonzero
    started = time.monotonic()
    data,seg,original,affine,header_spacing,units,before = load_pair(job)
    values, counts = np.unique(original,return_counts=True)
    missing = sorted(set(range(1,8))-set(map(int,values)))
    # Nibabel xyz -> nnU-Net's zyx, with no canonical orientation assumption.
    image_zyx, seg_zyx = data.transpose(2,1,0)[None],seg.transpose(2,1,0)[None]
    cropped, cropped_seg, bbox = crop_to_nonzero(image_zyx,seg_zyx)
    original_target_count = int(np.count_nonzero((seg>=1)&(seg<=7)))
    cropped_target_count = int(np.count_nonzero((cropped_seg>=1)&(cropped_seg<=7)))
    if original_target_count != cropped_target_count:
        raise ValueError('Image-only nonzero crop would remove true labels; requires review')
    samples = sample_foreground(data,seg,job['method']['foreground_samples_per_case'],job['method']['sample_seed'])
    sample_path = Path(job['sample_path'])
    sample_path.parent.mkdir(parents=True,exist_ok=True)
    temp = sample_path.with_suffix('.partial')
    with temp.open('wb') as f:
        np.save(f,samples,allow_pickle=False)
    os.replace(temp,sample_path)
    check_source_unchanged(job,before)
    row = dict(success=True,case_id=job['pair']['case_id'],pair=job['pair'],
               source_hashes=before,shape_xyz=list(data.shape),header_spacing_xyz=header_spacing,
               header_spatial_unit=units,source_affine=affine.tolist(),
               spacing_xyz=job['spacing_xyz'],spacing_zyx=job['spacing_xyz'][::-1],
               spacing_evidence=job['spacing_evidence'],
               cropped_shape_zyx=list(cropped.shape[1:]),bbox_zyx=bbox,
               label_counts={str(int(v)):int(n) for v,n in zip(values,counts)},
               missing_label_ids=missing,extra_label_ids=sorted(set(map(int,values))-set(range(8))),
               sample_path=str(sample_path),sample_sha256=sha256_file(sample_path),
               original_target_voxels=original_target_count,cropped_target_voxels=cropped_target_count,
               elapsed_seconds=time.monotonic()-started,worker_pid=os.getpid())
    return row


def bounded_locations(seg, classes_or_regions, seed=1234, verbose=False,
                      min_num_samples=10000,min_percent_coverage=.01):
    """Uniform location sampling without sorting full-volume background coords.

    nnU-Net 2.8.1 collects all background coordinates when ignore is active.
    This adapter counts labels in 1M-voxel chunks, samples offsets uniformly,
    then resolves only selected offsets. It keeps the annotated-area key and
    target sampling fractions; RNG draws differ from upstream, versioned here.
    """
    flat = seg[0].ravel()
    block = 1_000_000
    rng = np.random.default_rng(seed)
    result = {}
    for c in classes_or_regions:
        labs = tuple(c) if isinstance(c,(tuple,list)) else (int(c),)
        key = labs if isinstance(c,(tuple,list)) else int(c)
        counts = [int(np.isin(flat[s:s+block],labs).sum()) for s in range(0,len(flat),block)]
        total = sum(counts)
        if not total:
            result[key] = []
            continue
        k = min(total,max(min(min_num_samples,total),int(np.ceil(total*min_percent_coverage))))
        offsets = np.sort(rng.choice(total,k,replace=False,shuffle=False))
        indices = np.empty(k,dtype=np.int64)
        seen = 0
        for block_index,count in enumerate(counts):
            left,right = np.searchsorted(offsets,[seen,seen+count])
            if right>left:
                start = block_index*block
                positions = np.flatnonzero(np.isin(flat[start:start+block],labs))
                indices[left:right] = start+positions[offsets[left:right]-seen]
            seen += count
        # nnU-Net 2.8.1 get_bbox indexes selected_voxel[axis + 1]. Its
        # contract is (channel,z,y,x), not just (z,y,x), even for one channel.
        coordinates = np.column_stack(np.unravel_index(indices,seg.shape[1:]))
        result[key] = np.column_stack((np.zeros(k,dtype=np.int32),coordinates)).astype(np.int32)
    return result


def corrected_affine(affine, spacing):
    """Derived author grid: scale columns, preserve directions and origin.

    Excel J,K,M map to original array x,y,z exactly as in the previous audited
    adapter. This establishes a reported grid, not independent DICOM geometry.
    """
    result = np.asarray(affine,dtype=float).copy()
    lengths = np.linalg.norm(result[:3,:3],axis=0)
    if not np.isfinite(result).all() or (lengths<=0).any():
        raise ValueError('Invalid source affine')
    result[:3,:3] *= np.asarray(spacing)/lengths
    return result


def preprocess_case(job):
    from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
    from nnunetv2.training.dataloading.nnunet_dataset import comp_blosc2_params
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    import blosc2
    import torch
    started = time.monotonic()
    torch.set_num_threads(1)
    # Each owned worker calls this once. The guard also permits repeated
    # synthetic roundtrips in one process without resetting PyTorch's runtime.
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    blosc2.set_nthreads(1)
    data,seg,original,affine,_,_,before = load_pair(job)
    native_counts = {str(k):int((seg==k).sum()) for k in range(9)}
    native_ids = {k for k in range(9) if native_counts[str(k)]}
    observed_annotation = make_annotation(set(range(1,8)) - native_ids)
    annotation = job.get('annotation', observed_annotation)
    if annotation != observed_annotation:
        raise ValueError('Native labels differ from audited annotation contract')
    if annotation['requires_partial_loss']:
        if (job['method'].get('partial_annotation_policy') != 'missing_myo_only'
                or annotation['unannotated_class_ids'] != [5] or 'annotation' not in job):
            raise ValueError('Partial source requires the audited missing-MYO preprocessing protocol')
    group = job.get('cohort_group', 'auxiliary' if annotation['requires_partial_loss'] else 'main')
    validate_case_roles(annotation,group,job.get('roles',{}))
    derived_affine = corrected_affine(affine,job['spacing_xyz'])
    properties = dict(nibabel_stuff=dict(original_affine=derived_affine),
                      spacing=job['spacing_xyz'][::-1],source_affine=affine,
                      source_shape_xyz=list(data.shape),source_hashes=before,
                      geometry_evidence=job['spacing_evidence'],method_protocol=job['method']['protocol'],
                      annotation=annotation,cohort_group=group,roles=job.get('roles',{}),
                      foreground_sampling_class_ids=annotation['annotated_class_ids'],
                      class_location_coordinate_format='channel_zyx',
                      annotated_area_semantics=('nonignore_region_with_ambiguous_background'
                          if annotation['requires_partial_loss'] else 'nonignore_region'))
    plans = PlansManager(job['plan'])
    config = plans.get_configuration('3d_fullres')

    class ProjectBoundedPreprocessor(DefaultPreprocessor):
        _sample_foreground_locations = staticmethod(bounded_locations)

    processor = ProjectBoundedPreprocessor(verbose=True)
    print('阶段：影像非零裁剪 → 训练集统计归一化 → 官方重采样 → 有界前景位置采样',flush=True)
    image, labels, properties = processor.run_case_npy(data.transpose(2,1,0)[None],
        seg.transpose(2,1,0)[None],properties,plans,config,job['dataset_json'])
    if image.shape != labels.shape or not np.isfinite(image).all():
        raise ValueError('Non-finite or shape-inconsistent preprocessed arrays')
    label_ids = set(map(int,np.unique(labels)))
    validate_resampled_labels(native_ids,label_ids,annotation)
    image = image.astype(np.float32,copy=False)
    labels = labels.astype(np.int8,copy=False)
    cid = job['pair']['case_id']
    foldroot = Path(job['foldroot'])
    staging = foldroot/'staging'/f'{cid}-{os.getpid()}'
    staging.mkdir(parents=True,exist_ok=True)
    base = staging/cid
    blocks, chunks = comp_blosc2_params(image.shape,tuple(job['method']['compression_patch_hint']),image.itemsize)
    seg_blocks, seg_chunks = comp_blosc2_params(labels.shape,tuple(job['method']['compression_patch_hint']),labels.itemsize)
    # Upstream save_case hardcodes four compression threads. Use the SAME b2nd
    # file format with explicit one-thread LZ4: a lossless storage choice only.
    cparams = dict(codec=blosc2.Codec.LZ4,clevel=5,nthreads=1,filters=[blosc2.Filter.BITSHUFFLE])
    print('阶段：写入并重新打开Blosc2数据，核验shape/dtype与全数组字节',flush=True)
    for array,suffix,b,c in ((image,'.b2nd',blocks,chunks),(labels,'_seg.b2nd',seg_blocks,seg_chunks)):
        path = str(base)+suffix
        if Path(path).exists():
            Path(path).unlink()
        blosc2.asarray(np.ascontiguousarray(array),urlpath=path,blocks=b,chunks=c,cparams=cparams)
        stored = blosc2.open(path,mode='r',dparams={'nthreads':1})
        if stored.shape != array.shape or stored.dtype != array.dtype:
            raise ValueError('Stored shape/dtype differs from memory')
        # Chunkwise read-back compares every value without a second full volume.
        for z in range(0,array.shape[1],16):
            if not np.array_equal(stored[:,z:z+16],array[:,z:z+16]):
                raise ValueError('Stored array failed full read-back verification')
        del stored
    with Path(str(base)+'.pkl').open('wb') as f:
        pickle.dump(properties,f,pickle.HIGHEST_PROTOCOL)
    check_source_unchanged(job,before)
    target = foldroot/'data'
    target.mkdir(parents=True,exist_ok=True)
    files = {}
    for suffix in ('.b2nd','_seg.b2nd','.pkl'):
        name = cid+suffix
        os.replace(staging/name,target/name)
        files[name] = sha256_file(target/name)
    staging.rmdir()
    row = dict(success=True,case_id=cid,source_hashes=before,
        annotation=annotation,cohort_group=group,roles=job.get('roles',{}),
        shape=list(image.shape),data_dtype=str(image.dtype),seg_dtype=str(labels.dtype),
        output_label_ids=sorted(label_ids),native_label_counts=native_counts,
        resampled_label_counts={str(k):int((labels==k).sum()) for k in sorted(label_ids)},
        bbox_used_for_cropping=properties['bbox_used_for_cropping'],
        shape_before_cropping=list(properties['shape_before_cropping']),
        target_spacing=config.spacing,transpose_forward=plans.transpose_forward,
        derived_affine=derived_affine.tolist(),source_affine=affine.tolist(),
        source_spacing_xyz=job['spacing_xyz'],geometry_evidence=job['spacing_evidence'],
        class_location_counts={str(k):len(v) for k,v in properties['class_locations'].items()},
        file_sha256=files,contract=job['contract'],fold=job['plan']['fold'],
        elapsed_seconds=time.monotonic()-started,worker_pid=os.getpid(),
        threads=dict(torch=torch.get_num_threads(),compression=1,scipy_resampling=1),
        warnings=['ignore_region_disappeared_during_resampling'] if native_counts['8'] and 8 not in label_ids else [])
    # Completion lives OUTSIDE data/ so nnU-Net will never scan it as a case.
    write_json(foldroot/'completion'/f'{cid}.json',row)
    return row
