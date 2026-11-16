from pathlib import Path
import math

from ..models.stunet.adapter import StunetAdapter


def stage_memory(config, prepared, prefetch=None):
    prefetch = min(2, config["resources"]["prefetch"]) if prefetch is None else prefetch
    if type(prefetch) is not int or prefetch not in (1, 2):
        raise ValueError("训练主机缓冲预算只支持1或2批")
    spec = StunetAdapter.spec(config, prepared["plan"])
    shape = list(spec.patch)
    sizes = [math.prod(shape)]
    for stride in spec.strides:
        shape = [(size + factor - 1) // factor for size, factor in zip(shape, stride)]
        sizes.append(math.prod(shape))
    channels = (16, 32, 64, 128, 256, 256)
    encoder = sum(size * width for size, width in zip(sizes, channels))
    decoder = sum(size * width for size, width in zip(sizes[:-1], channels[:-1]))
    batch = config["training"]["batch"]
    # 同时覆盖卷积/归一化保存量、跳跃连接、反向临时量和深监督损失；按FP32保守核算。
    cpu_compute = batch * (encoder * 64 + decoder * 96 + sum(sizes[:-1]) * spec.num_classes * 32)
    # 覆盖当前CPU批、pinned源和下一批原始供数结果，额度不足时供数规划退回单批。
    transfers = batch * sizes[0] * (24 + 16 * (prefetch - 1))
    weights = Path(config["pretrained"]).stat().st_size
    return {"protocol": "stunet-stage-ram-v2", "batch_bytes": int(transfers +
                (cpu_compute if config["resources"]["device"] == "cpu" else 0)),
            "prefetch": prefetch,
            "model_load_per_rank_bytes": 3 * weights + 8 * 1024 ** 2,
            "checkpoint_reserve_bytes": 4 * weights + 8 * 1024 ** 2,
            "basis": "STU-Net-S逐层尺寸与全局batch的保守预算，实际值由MemoryGuard持续核对"}
