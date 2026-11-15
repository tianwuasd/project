from copy import deepcopy
from pathlib import Path

from chd_segmentation.artifacts.store import read_json, write_json
from chd_segmentation.config import load_config
from chd_segmentation.models.offline import preprocess_model
from chd_segmentation.preprocessing.prepare import prepare


def offline_model(model_path, source_config, source_path, output):
    output = Path(output)
    root = Path(load_config(source_config)['work_root']).parent/'data'
    return preprocess_model(model_path, source_config, source_path, root, output)


def unlabeled_model(model_path, config_path, trained, directory):
    directory = Path(directory)
    config = deepcopy(read_json(config_path))
    manifest = deepcopy(read_json(config['manifest']))
    for row in manifest['cases']:
        row.update(roles=['predict'], label=None)
        row.get('sha256', {}).pop('label', None)
        row.get('source_stamp', {}).pop('label', None)
    write_json(directory/'prediction-manifest.json', manifest)
    write_json(directory/'prediction-plan.json', trained['plan'])
    config.update(manifest=str(directory/'prediction-manifest.json'), plan=str(directory/'prediction-plan.json'))
    config['planning']['mode'] = 'existing'
    source_config = directory/'prediction-config.json'
    write_json(source_config, config)
    source = directory/'course/prediction-source.json'
    prepare(load_config(source_config), source)
    output = directory/'course/prediction-input.json'
    offline_model(model_path, source_config, source, output)
    return output
