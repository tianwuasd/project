from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root/'src'))
    from chd_segmentation.models.medsam3.finetune import settings, check
    config = settings(root/'configs/models/medsam3_finetune.yaml')
    run = Path(config['training']['run_dir'])
    existed = run.exists()
    result = check(config)
    assert result['model_loaded'] is False and isinstance(result['ready'], bool)
    assert run.exists() == existed
    assert 'torch' not in sys.modules
    for command in ('train', 'validate', 'check-training', 'predict', 'evaluate', 'inspect'):
        output = subprocess.check_output([sys.executable, '-B', str(root/'run.py'), 'models', 'medsam3', command, '--help'])
        assert b'--config' in output
        if command == 'train':
            assert b'--resume' in output and b'--stop-after' in output and b'--prepared' in output
    print('MedSAM3真实CLI及无模型、无目录写入的预检查通过')


if __name__ == '__main__':
    main()
