from math import isfinite


def selection_options(values, model):
    defaults = dict(enabled=False, monitor='validation_sequence_loss' if model == 'medsam2' else 'validation_loss',
                    mode='min', patience=10 if model == 'q_route1' else 20, min_delta=.0001,
                    unit='completed_validation_event' if model == 'q_route1' else 'completed_validation_epoch')
    if not isinstance(values, dict) or set(values) - defaults.keys():
        raise ValueError('提前停止配置字段无效')
    result = {**defaults, **values}
    if type(result['enabled']) is not bool or type(result['patience']) is not int or result['patience'] < 1:
        raise ValueError('提前停止开关或耐心次数无效')
    if result['mode'] != 'min' or result['monitor'] != defaults['monitor'] or result['unit'] != defaults['unit']:
        raise ValueError('提前停止指标、方向或计数单位与模型不符')
    if type(result['min_delta']) not in (float, int) or not isfinite(result['min_delta']) or result['min_delta'] < 0:
        raise ValueError('提前停止改善阈值必须为有限非负数')
    return result


def update_selection(state, metric, position, options):
    if not isfinite(metric):
        raise FloatingPointError('验证指标不是有限数值')
    current = dict(state or {})
    improved = current.get('best') is None or metric < current['best']
    if improved:
        current.update(best=float(metric), best_position=position)
    reference = current.get('early_stop_reference')
    if reference is None or metric < reference - options['min_delta']:
        current.update(early_stop_reference=float(metric), bad_validations=0)
    else:
        current['bad_validations'] += 1
    current.update(improved=improved, last_metric=float(metric), last_position=position,
                   stop=options['enabled'] and current['bad_validations'] >= options['patience'],
                   monitor=options['monitor'], unit=options['unit'])
    return current


def resolve_budget(config, model, stage):
    if model == 'q_route1':
        defaults = dict(base=40000, localizer=1250, local=5000)
        updates = config.get('updates')
        epochs = config.get('epochs')
        per_epoch = config.get('updates_per_epoch')
        if per_epoch is None:
            per_epoch = 250
        if type(per_epoch) is not int or per_epoch<1:
            raise ValueError('Q的每轮更新数量必须为正整数')
        if updates is None and epochs is None and stage!='base':
            epochs = 5 if stage=='localizer' else 20
        if epochs is not None:
            if type(epochs) is not int or epochs < 1:
                raise ValueError('Q的epoch数量必须为正整数')
            if type(per_epoch) is not int or per_epoch < 1:
                raise ValueError('Q的epoch预算需要正整数updates_per_epoch')
            converted = epochs * per_epoch
            if updates is not None and updates != converted:
                raise ValueError('Q的updates与epoch换算预算冲突')
            updates = converted
        if updates is None:
            updates = defaults[stage]
        if type(updates) is not int or updates < 1:
            raise ValueError('Q更新预算必须为正整数')
        return dict(updates=updates, epochs=epochs, updates_per_epoch=per_epoch, unit='updates')
    epochs = config.get('epochs')
    if model == 'stunet' and epochs is None and not config.get('use_model_default', False):
        attempts = config.get('attempts', 1000)
        if type(attempts) is not int or attempts < 1:
            raise ValueError('STU训练尝试预算必须为正整数')
        return dict(attempts=attempts, epochs=None, unit='attempts')
    if epochs is None:
        epochs = 75 if model == 'medsam2' else 200
    if type(epochs) is not int or epochs < 1:
        raise ValueError('训练epoch预算必须为正整数')
    return dict(epochs=epochs, unit='epochs')
