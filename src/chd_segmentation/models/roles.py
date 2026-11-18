def technical_names(roles, stage, role, available):
    if stage not in roles or set(roles[stage]) != {'train', 'validation', 'test'}:
        raise ValueError('技术阶段缺少明确train/validation/test角色名单：'+stage)
    assigned = roles[stage]
    seen = set()
    for kind in ('train', 'validation', 'test'):
        names = assigned[kind]
        if not isinstance(names, list) or any(not isinstance(name, str) for name in names):
            raise ValueError('技术角色名单必须为病例名称列表')
        if len(set(names)) != len(names) or seen.intersection(names) or not set(names) <= set(available):
            raise ValueError('技术阶段角色重叠、重复或病例未登记：'+stage)
        seen.update(names)
    if not assigned['train']:
        raise ValueError('技术阶段训练角色为空：'+stage)
    return list(assigned[role])
