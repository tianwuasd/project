"""Pinned nnU-Net rank target: one shared experiment, finite trials or full epochs."""
import copy
import os
from pathlib import Path
import random
import time


def run_rank(rank,world_size,args):
    # Environment and selected devices are established by ddp_runtime before import.
    os.environ['WHOLE_HEART_WORKSPACE']=args['workspace']
    os.environ['VISTA_RANK_GPU_UUID']=args['gpu_uuids'][rank]
    from prepare_nnunet import configure_paths
    configure_paths()
    import numpy as np
    import torch
    import torch.distributed as dist
    from data_pipeline import load_json
    from model_catalog import trainer_class
    from nnunetv2.training.logging.nnunet_logger import MetaLogger
    from nnunet_batches import close_loaders
    from server_data import write_json
    from smoke_train import state_hash
    from task_resources import apply_training_cpu
    from training_timing import measure_step,summarize_steps
    resource=args['resource']
    from rank_affinity import bind
    affinity=bind(rank,args['gpu_uuids'],resource)
    limiter=apply_training_cpu(resource,torch)
    random.seed(0);np.random.seed(0);torch.random.default_generator.manual_seed(0)
    if torch.cuda.is_available():torch.cuda.manual_seed(0)
    torch.backends.cudnn.benchmark=False
    work=Path(args['workspace']);fold=args['fold'];method=args['method']
    split=load_json(work/'04_data/manifests/splits_v1.json')['folds'][fold]
    prep=work/f"04_data/derived/nnUNet_preprocessed/Dataset{split['dataset_id']:03d}_{fold}"
    plans=copy.deepcopy(load_json(prep/'WholeheartBaselinePlans.json'))
    plans['configurations']['3d_fullres']['batch_size']=args['global_batch_size']
    output=Path(args['output']);output.mkdir(parents=True,exist_ok=True)
    device=torch.device('cuda',rank) if dist.get_backend()=='nccl' else torch.device('cpu')
    trainer=trainer_class(method)(plans,'3d_fullres',0,load_json(prep/'dataset.json'),device=device)
    trainer._vista_resource_config=dict(resource)
    trainer._vista_run_kind=args['run_kind']
    trainer._vista_world_size=world_size
    trainer._vista_global_batch=args['global_batch_size']
    trainer.output_folder_base=str(output)
    trainer.output_folder=str(output/'fold_0')
    Path(trainer.output_folder).mkdir(exist_ok=True)
    trainer.log_file=str(output/('training.log' if rank==0 else f'training.rank{rank}.log'))
    trainer.logger=MetaLogger(trainer.output_folder,resume=bool(args.get('checkpoint')))
    trainer.save_every=1
    trainer.initialize()
    initial_hash=state_hash(trainer.network.module if trainer.is_ddp else trainer.network)
    if args.get('checkpoint'):trainer.load_checkpoint(args['checkpoint'])
    if trainer.num_epochs!=250 or trainer.num_iterations_per_epoch!=250 or trainer.num_val_iterations_per_epoch!=50:
        raise ValueError('正式CT预算与获准250轮×250训练/50验证不符')
    trial=args.get('operation')=='trial'
    started=time.monotonic()
    result=dict(status='running',world_size=world_size,global_batch_size=args['global_batch_size'],
        local_batch_size=trainer.batch_size,resource=resource,run_kind=args['run_kind'],
        patch_size=list(map(int,trainer.configuration_manager.patch_size)),cuda=device.type=='cuda',
        initial_weight_sha256=initial_hash,affinity=affinity,
        resumed_initial_checkpoint=args.get('checkpoint'),first_epoch=trainer.current_epoch)
    sync=(lambda:torch.cuda.synchronize(device)) if device.type=='cuda' else (lambda:None)
    try:
        trainer.on_train_start()
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
        for epoch in range(trainer.current_epoch,trainer.num_epochs):
            trainer.on_epoch_start();trainer.on_train_epoch_start()
            outputs=[];measurements=[]
            train_steps=12 if trial else 250
            for batch_index in range(train_steps):
                if trial:
                    _,value,timing=measure_step(trainer.dataloader_train,trainer.train_step,sync)
                    measurements.append(timing)
                else:value=trainer.train_step(next(trainer.dataloader_train))
                outputs.append(value)
                if rank==0 and (batch_index==0 or (batch_index+1)%5==0 or batch_index+1==train_steps):
                    write_json(output/'progress.rank0.json',dict(status='running',phase='train',
                        epoch=trainer.current_epoch,planned_epochs=250,batch=batch_index+1,total_batches=train_steps,
                        world_size=world_size,global_batch_size=args['global_batch_size'],updated=time.time()))
            trainer.on_train_epoch_end(outputs)
            with torch.no_grad():
                trainer.on_validation_epoch_start();outputs=[];validation=[]
                validation_steps=2 if trial else 50
                for batch_index in range(validation_steps):
                    if trial:
                        _,value,timing=measure_step(trainer.dataloader_val,trainer.validation_step,sync)
                        validation.append(timing)
                    else:value=trainer.validation_step(next(trainer.dataloader_val))
                    outputs.append(value)
                    if rank==0 and (batch_index==0 or (batch_index+1)%5==0 or batch_index+1==validation_steps):
                        write_json(output/'progress.rank0.json',dict(status='running',phase='validation',
                            epoch=trainer.current_epoch,planned_epochs=250,batch=batch_index+1,total_batches=validation_steps,
                            world_size=world_size,global_batch_size=args['global_batch_size'],updated=time.time()))
                trainer.on_validation_epoch_end(outputs)
            tick=time.monotonic();trainer.on_epoch_end();checkpoint_seconds=time.monotonic()-tick
            if trial:
                checkpoint=output/'fold_0/checkpoint_latest.pth'
                expected=state_hash(trainer.network)
                close_loaders(trainer)
                # Prove loading modifies a changed model, not merely that a file exists.
                with torch.no_grad():next(trainer.network.parameters()).add_(1)
                trainer.load_checkpoint(str(checkpoint))
                if state_hash(trainer.network)!=expected or not trainer.optimizer.state:
                    raise RuntimeError('实际权重或优化器检查点恢复失败')
                trainer.on_train_start();trainer.on_epoch_start();trainer.on_train_epoch_start()
                value=trainer.train_step(next(trainer.dataloader_train))
                if not np.isfinite(value['loss']).all():raise FloatingPointError('恢复后训练loss无效')
                stats=summarize_steps(measurements,warmup=2)
                result.update(status='passed',measured_training_steps=10,
                    median_step_seconds=stats['total_seconds']['median'],
                    validation_step_seconds=sum(v['total_seconds'] for v in validation)/2,
                    checkpoint_seconds=checkpoint_seconds,phase_timing=dict(train=stats,validation=summarize_steps(validation)),
                    checkpoint_weights_equal=True,optimizer_state_restored=True,
                    rng_state_restored=trainer.rng_state_restored,one_resumed_training_step_passed=True,
                    actual_optimizer_steps=13,checkpoint=str(checkpoint))
                break
            # Every rank observes one rank0 decision at the same completed epoch.
            request=[False]
            if rank==0 and args.get('regroup_request'):
                path=Path(args['regroup_request'])
                if path.exists():request[0]=load_json(path).get('gpu_uuids')!=args['gpu_uuids']
            dist.broadcast_object_list(request,src=0)
            if request[0] and trainer.current_epoch<trainer.num_epochs:
                result.update(status='regroup',next_epoch=trainer.current_epoch,
                    checkpoint=str(output/'fold_0/checkpoint_latest.pth'))
                break
        else:
            trainer.on_train_end()
            result.update(status='completed',next_epoch=trainer.current_epoch,
                checkpoint=str(output/'fold_0/checkpoint_final.pth'))
        if device.type=='cuda':
            result.update(peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,
                          peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30,
                          device=torch.cuda.get_device_name(device),total_gib=torch.cuda.get_device_properties(device).total_memory/2**30)
        result['seconds']=time.monotonic()-started
        all_results=[None]*world_size;dist.all_gather_object(all_results,result)
        if trial:
            result['median_step_seconds']=max(r['median_step_seconds'] for r in all_results)
            result['validation_step_seconds']=max(r['validation_step_seconds'] for r in all_results)
            if device.type=='cuda':
                result['peak_allocated_gib']=max(r['peak_allocated_gib'] for r in all_results)
                result['peak_reserved_gib']=max(r['peak_reserved_gib'] for r in all_results)
            result['rank_results']=all_results if rank==0 else []
        if rank==0:write_json(output/'result.json',result)
        return result
    finally:
        close_loaders(trainer)
        limiter.restore_original_limits()
