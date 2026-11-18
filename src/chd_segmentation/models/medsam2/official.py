from pathlib import Path

from filelock import FileLock

from ...artifacts.store import read_json,write_json
from ...preprocessing.prepare import load_prepared
from .adapter import activate


OFFICIAL_CONFIG = 'sam2/configs/sam2.1_hiera_tiny512_FLARE_RECIST.yaml'


def effective_config(config,prepared,prepared_path,root,names):
    official = activate(config)
    from omegaconf import OmegaConf
    from training.utils.train_utils import register_omegaconf_resolvers
    if not OmegaConf.has_resolver('times'):
        register_omegaconf_resolvers()
    recipe = OmegaConf.load(official/OFFICIAL_CONFIG)
    recipe.launcher.experiment_log_dir = str(root)
    recipe.scratch.train_video_batch_size = config['batch']
    recipe.scratch.num_train_workers = config['workers']
    recipe.scratch.base_lr = config['learning_rate']
    recipe.scratch.vision_lr = config['learning_rate']*.6
    recipe.trainer.seed_value = config['seed']
    recipe.trainer.accelerator = 'cpu' if config['device']=='cpu' else 'cuda'
    recipe.trainer.distributed.backend = 'gloo' if config['device']=='cpu' else 'nccl'
    recipe.trainer.optim.amp.enabled = config['device']!='cpu'
    recipe.trainer.optim.amp.amp_dtype = 'bfloat16'
    if config['precision']=='fp32':
        recipe.trainer.optim.amp.enabled = False
    recipe.trainer.checkpoint.save_freq = 0
    recipe.trainer.checkpoint.model_weight_initializer.state_dict.checkpoint_path = config['pretrained']
    recipe.trainer.logging.tensorboard_writer.should_log = False
    if config['epochs'] is not None:
        recipe.scratch.num_epochs = config['epochs']
    from .native import resolve_window
    window = resolve_window(config,prepared)
    raw = dict(_target_='chd_segmentation.models.medsam2.data.ManifestRawDataset',
               prepared_path=str(Path(prepared_path).resolve()),names=names,window=window,
               log_dir=str(root),logging=config['logging'], input_source=config.get('input_source', {}))
    leaf = recipe.trainer.data.train.datasets[0].dataset.datasets[0]
    leaf._target_ = 'chd_segmentation.models.medsam2.data.HeartVOSDataset'
    leaf.video_dataset = OmegaConf.create(raw)
    resolved = OmegaConf.to_container(recipe,resolve=True)
    # 数据、资源和保存位置是本工程的接口；模型、增强、采样和优化配置保留官方定义。
    return resolved


def train_official(config,prepared_path,stage,resume=False,stop_after=None):
    if stage!='sequence':
        raise ValueError('MedSAM2阶段必须为sequence')
    prepared = load_prepared(prepared_path)
    from ..workflow import stage_case_names,selected_prepared
    consumed = stage_case_names(config,prepared,stage)
    if config.get('input_source',{}).get('path'):
        from ...data.input_binding import bind_prepared_source
        prepared = bind_prepared_source(prepared,config['input_source']['path'],case_ids=consumed)
    from ..offline import model_storage_lease
    root = (Path(config['run_dir'])/stage).resolve()
    protected = [Path(prepared['manifest']['data_root']).resolve(),
                 Path(config['official_root']).resolve(),Path(config['dependency_root']).resolve(),
                 Path(config['pretrained']).resolve().parent]
    if any(root==path or root.is_relative_to(path) or path.is_relative_to(root) for path in protected):
        raise ValueError('MedSAM2训练目录与原始数据或复用依赖重叠')
    root.parent.mkdir(parents=True,exist_ok=True)
    with FileLock(root.parent/(stage+'.lock'),timeout=0), model_storage_lease(config,selected_prepared(prepared,consumed),stage):
        return _train_locked(config,prepared,prepared_path,root,stage,resume,stop_after)


def _train_locked(config,prepared,prepared_path,root,stage,resume,stop_after):
    if config['frames']!=8 or config['resolution']!=512 or config['accumulation']!=1 or config['freeze_image_encoder']:
        raise ValueError('MedSAM2官方配方需要8帧、512分辨率、无梯度累积且训练全部模型')
    from .weights import configured_weight
    receipt = configured_weight(config['pretrained'])
    from ..workflow import training_names,checked_identity
    from ..execution import admitted,run_workers
    names = training_names(config,prepared,stage)
    if config['technical'] and config['updates']>2:
        raise ValueError('MedSAM2技术验证最多两次更新')
    if stop_after is not None and (type(stop_after) is not int or stop_after<1 or
                                  config['technical'] and stop_after>config['updates']):
        raise ValueError('MedSAM2停止更新数超出预算')
    if not resume:
        from ..startup import check_start
        check_start(root,config['model'],prepared['identity'],stage)
    if resume and not (root/'checkpoints/checkpoint.pt').is_file():
        raise FileNotFoundError('MedSAM2恢复需要完整官方检查点')
    source_changed = False
    if resume:
        from ...training.checkpoint import require_resume_device
        previous_input = read_json(root/'worker-input.json')
        from ..contracts import require_resume_contract
        source_changed = require_resume_contract(previous_input['config'],previous_input['prepared'],config,prepared,stage)
        previous_names = previous_input['recipe']['trainer']['data']['train']['datasets'][0]['dataset']['datasets'][0]['video_dataset']['names']
        if (previous_input['stage']!=stage or previous_input['config']['model']!=config['model'] or
            previous_input['prepared']['dataset']!=prepared['dataset'] or
            previous_input['prepared']['dataset_json']!=prepared['dataset_json'] or
            previous_input['prepared']['scope']!=prepared['scope'] or previous_names!=names):
            raise ValueError('MedSAM2恢复的模型阶段、数据集类别或当前训练名单不同')
        require_resume_device(previous_input['config']['device'],config['device'])
    recipe = effective_config(config,prepared,prepared_path,root,names)
    previous = read_json(root/'identity.json')['identity'] if resume else None
    identity = checked_identity(config,prepared,stage,previous=previous)
    payload = dict(config=config,prepared=prepared,recipe=recipe,identity=identity,weight=receipt,stage=stage,
                   prepared_path=str(Path(prepared_path).resolve()),resume=resume,stop_after=stop_after,
                   selection_source_changed=source_changed)
    topology = read_json(root/'execution-topology.json') if resume else None
    with admitted(config,prepared,stage,topology) as (common,allocation):
        return run_workers(payload,common,allocation)
