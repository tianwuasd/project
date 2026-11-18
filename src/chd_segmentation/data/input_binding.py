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


def medsam_case(entry, input_source):
    from ..models.medsam2.intensity import convert_image, resolve_intensity
    source = entry.get('intensity', {})
    if entry.get('medsam_windowed'):
        source = dict(source, semantics='windowed_uint8', gray_window=entry.get('gray_window'))
    policy = resolve_intensity(input_source, source)
    image, conversion = convert_image(entry['image'], policy)
    return dict(entry, image=image, medsam_windowed=True, intensity_conversion=conversion,
                gray_window=policy.get('gray_window'))
