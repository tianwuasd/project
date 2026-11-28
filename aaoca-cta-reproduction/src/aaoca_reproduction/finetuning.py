import torch


STRATEGY = "author_detector_finetune"


def settings_for_finetuning(settings):
    if settings.get("strategy") != STRATEGY:
        return None
    value = settings["fine_tuning"]
    if (type(value["warmup_epochs"]) is not int or not 0 <= value["warmup_epochs"] < settings["epochs"]
            or value["freeze_batchnorm"] is not True or value["selection_metric"] != "average_precision"
            or not 0 < value["backbone_lr"] <= value["head_lr"] <= 0.001
            or type(value["patience"]) is not int or value["patience"] < 1):
        raise ValueError("检测微调须显式声明有效暖机轮数、两组学习率、固定BatchNorm、AP选优和早停")
    if settings.get("augmentation", "author") not in {"author", "source_robust_v1"}:
        raise ValueError("未知检测微调增强配方")
    return value


def make_optimizer(model, fine, learning_rate):
    if fine is None:
        return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    # 两阶段使用同一组优化器参数，接续时不会改变参数组或丢失动量。
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in (model.layer4, model.last_linear):
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    return torch.optim.AdamW([
        {"params": model.last_linear.parameters(), "lr": fine["head_lr"]},
        {"params": model.layer4.parameters(), "lr": fine["backbone_lr"]}], weight_decay=0.01)


def set_training_phase(model, fine, epoch):
    model.train()
    if fine is None:
        return "full_network"
    core = model.module if isinstance(model, torch.nn.DataParallel) else model
    unfrozen = epoch >= fine["warmup_epochs"]
    for parameter in core.layer4.parameters():
        parameter.requires_grad_(unfrozen)
    # 小样本训练保留作者的归一化统计；分类层Dropout继续执行。
    for module in core.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
    return "last_stage_and_head" if unfrozen else "head_warmup"
