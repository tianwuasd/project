from dataclasses import dataclass


@dataclass
class InputBinding:
    dataset: str
    roles: dict
    source: object
    intensity: dict


def resolve_input_source(source_path, experiment):
    from ..preprocessing.inputs import resolve_preprocessed
    roles = experiment['roles']
    ids = list(dict.fromkeys(name for names in roles.values() for name in names))
    source = resolve_preprocessed(source_path, dataset=experiment['dataset'], case_ids=ids)
    return InputBinding(experiment['dataset'], roles, source, experiment.get('input_source', {}))


def load_case(binding, case_id, with_label=True):
    if not any(case_id in names for names in binding.roles.values()):
        raise ValueError('病例不在当前实验名单中：'+case_id)
    return binding.source.open(case_id, with_label=with_label)


def bind_prepared_source(prepared, path):
    from copy import deepcopy
    from ..artifacts.store import digest
    from ..preprocessing.inputs import resolve_preprocessed
    names = [row['case_id'] for row in prepared['manifest']['cases']]
    source = resolve_preprocessed(path,dataset=prepared['dataset'],case_ids=names)
    catalog = source.catalog
    if catalog['recipe']['scope'] != prepared['scope']:
        raise ValueError('当前任务与指定预处理的正式或技术范围不同')
    result = deepcopy(prepared)
    result['storage'] = dict(format=catalog['format'],root=str(source.root),complete=True,
        recipe_identity=catalog['recipe_identity'],cases=source.cases)
    result['plan'] = deepcopy(catalog['recipe'].get('plan',prepared['plan']))
    result['identity'] = digest(dict(dataset=result['dataset'],cases=names,storage=result['storage']))
    return result


def medsam_case(entry, input_source):
    from ..models.medsam2.intensity import convert_image, resolve_intensity
    source = entry.get('intensity', {})
    if source.get('kind') == 'ct_normalized':
        stats = (source.get('statistics') or {}).get('0', {})
        source = dict(source, semantics='zscore_clipped_hu', mean=stats.get('mean'), std=stats.get('std'),
            gray_window=[stats['percentile_00_5'], stats['percentile_99_5']]
                if 'percentile_00_5' in stats and 'percentile_99_5' in stats else None)
    if source.get('kind') == 'native_window_uint8':
        source = dict(source, semantics='windowed_uint8')
    if entry.get('medsam_windowed'):
        source = dict(source, semantics='windowed_uint8', gray_window=entry.get('gray_window'))
    policy = resolve_intensity(input_source, source)
    image, conversion = convert_image(entry['image'], policy)
    return dict(entry, image=image, medsam_windowed=True, intensity_conversion=conversion,
                gray_window=policy.get('gray_window'))
