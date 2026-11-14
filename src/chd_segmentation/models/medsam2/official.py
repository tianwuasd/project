from pathlib import Path

from filelock import FileLock

from ...artifacts.store import read_json,write_json
from ...preprocessing.prepare import load_prepared
from .adapter import activate
from .weights import verify_official_weight


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
    stats = prepared['plan']['foreground_intensity_properties_per_channel']['0']
    window = config['gray_window'] or [stats['percentile_00_5'],stats['percentile_99_5']]
    raw = dict(_target_='chd_segmentation.models.medsam2.data.ManifestRawDataset',
               prepared_path=str(Path(prepared_path).resolve()),names=names,window=window,
               log_dir=str(root),logging=config['logging'])
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
    root = (Path(config['run_dir'])/stage).resolve()
    protected = [Path(prepared['manifest']['data_root']).resolve(),
                 Path(config['official_root']).resolve(),Path(config['dependency_root']).resolve(),
                 Path(config['pretrained']).resolve().parent]
    if any(root==path or root.is_relative_to(path) or path.is_relative_to(root) for path in protected):
        raise ValueError('MedSAM2训练目录与原始数据或复用依赖重叠')
    root.parent.mkdir(parents=True,exist_ok=True)
    with FileLock(root.parent/(stage+'.lock'),timeout=0):
        return _train_locked(config,prepared,prepared_path,root,stage,resume,stop_after)


def _train_locked(config,prepared,prepared_path,root,stage,resume,stop_after):
    if config['frames']!=8 or config['resolution']!=512 or config['accumulation']!=1 or config['freeze_image_encoder']:
        raise ValueError('MedSAM2官方配方需要8帧、512分辨率、无梯度累积且训练全部模型')
    receipt = verify_official_weight(config['pretrained'])
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
    if resume:
        from ...training.checkpoint import require_resume_device
        require_resume_device(read_json(root/'worker-input.json')['config']['device'],config['device'])
    recipe = effective_config(config,prepared,prepared_path,root,names)
    previous = read_json(root/'identity.json')['identity'] if resume else None
    identity = checked_identity(config,prepared,stage,previous=previous)
    payload = dict(config=config,prepared=prepared,recipe=recipe,identity=identity,weight=receipt,stage=stage,
                   prepared_path=str(Path(prepared_path).resolve()),resume=resume,stop_after=stop_after)
    topology = read_json(root/'execution-topology.json') if resume else None
    with admitted(config,prepared,stage,topology) as (common,allocation):
        return run_workers(payload,common,allocation)
