from .contracts import git


def full_commit(value):
    if not isinstance(value, str) or len(value) != 40 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('版本记录需要完整提交SHA')
    return value


def model_versions(engine, model, item, final_commit):
    final_commit = full_commit(final_commit)
    sources = item['training_commits']
    if not isinstance(sources, list) or not sources or len(set(sources)) != len(sources):
        raise ValueError('每模型须列出不重复的实际训练提交')
    result = []
    for source in sources:
        source = full_commit(source)
        if git(engine, 'rev-parse', '--verify', source+'^{commit}') != source:
            raise ValueError('本地不能解析实际训练提交')
        changed = sorted(p for p in git(engine, 'diff', '--no-renames', '--name-only', '-z',
                                       source, final_commit, '--').split('\0') if p)
        reference = None
        if source != final_commit:
            review = item.get('version_reviews', {}).get(source)
            if not review or review.get('production_commit') != final_commit:
                raise ValueError(model+'训练/预测版本差异尚无对应最终版本的秘书审核')
            reference = review.get('secretary_review_reference')
            if not isinstance(reference, str) or not reference.strip():
                raise ValueError('版本差异需要秘书明确审核记录')
            approved = review['approved_changed_files']
            if (not isinstance(approved, list) or any(not isinstance(p,str) for p in approved) or
                    len(set(approved)) != len(approved) or sorted(approved) != changed):
                raise ValueError(model+'实际Git差异与批准文件列表不完全一致')
        result.append(dict(model=model, training_commit=source, prediction_commit=final_commit,
                           changed_files=changed, secretary_review_reference=reference,
                           evidence_scope='训练证据保留原SHA；仅批准在所列差异后的版本执行预测'))
    return result
