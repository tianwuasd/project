from contextlib import contextmanager
import os
import threading
import math

import psutil
import SimpleITK as sitk
import torch

from .records import admit_stage, event


def configure_cpu(config):
    count = int(config.get("cpu_threads", 2))
    if count < 1 or count > (os.cpu_count() or 1):
        raise ValueError("CPU线程数必须在实际主机允许范围")
    torch.set_num_threads(count)
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(count)
    return {"threads": count, "workers": 0, "prefetch": 0, "device": "cpu"}


def classification_device(config, estimated_gpu_bytes=0):
    name = config["classification"]["device"]
    device = torch.device(name)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("分类设备须显式选择CPU或获准的CUDA设备")
    if device.type == "cuda":
        if device.index is None or not torch.cuda.is_available() or device.index >= torch.cuda.device_count():
            raise ValueError("CUDA设备不可用或未明确指定索引")
        if not isinstance(estimated_gpu_bytes, int) or estimated_gpu_bytes <= 0:
            raise ValueError("CUDA执行前须提供获准且有依据的显存工作集估计")
        free, total = torch.cuda.mem_get_info(device)
        usable = free - max(4 * 2**30, int(total * 0.1))
        if estimated_gpu_bytes > usable:
            raise MemoryError("所选设备空闲显存不足，停止等待资源安排")
    return device


def model_resource_plan(config, training):
    from .classification import build_model

    settings = config["classification"]
    batch = int(settings["batch_size"]) if training else 1
    if batch < 1:
        raise ValueError("batch必须为正整数")
    # 使用真实MONAI在meta设备登记参数形状，不构造虚假病例或执行forward。
    with torch.device("meta"):
        model = build_model(settings["task"])
    parameters = sum(p.numel() for p in model.parameters())
    state_bytes = sum(t.numel() * t.element_size() for t in model.state_dict().values())
    shape = (85, 215, 215)
    activation_sizes = []

    def convolution(layer, spatial):
        output = tuple((n + 2 * p - d * (k - 1) - 1) // s + 1
                       for n, p, d, k, s in zip(spatial, layer.padding, layer.dilation, layer.kernel_size, layer.stride))
        activation_sizes.append(batch * layer.out_channels * math.prod(output) * 4)
        return output

    for layer in model.layer0:
        if isinstance(layer, torch.nn.Conv3d):
            shape = convolution(layer, shape)
        elif isinstance(layer, torch.nn.MaxPool3d):
            kernel = layer.kernel_size if isinstance(layer.kernel_size, tuple) else (layer.kernel_size,) * 3
            stride = layer.stride if isinstance(layer.stride, tuple) else (layer.stride,) * 3
            padding = layer.padding if isinstance(layer.padding, tuple) else (layer.padding,) * 3
            dilation = layer.dilation if isinstance(layer.dilation, tuple) else (layer.dilation,) * 3
            output = []
            for n, k, s, p, d in zip(shape, kernel, stride, padding, dilation):
                raw = (n + 2 * p - d * (k - 1) - 1) / s + 1
                size = math.ceil(raw) if layer.ceil_mode else math.floor(raw)
                output.append(size - 1 if layer.ceil_mode and (size - 1) * s >= n + p else size)
            shape = tuple(output)
    for stage in (model.layer1, model.layer2, model.layer3, model.layer4):
        for block in stage:
            incoming = shape
            for component in (block.conv1, block.conv2, block.conv3):
                shape = convolution(component.conv, shape)
            if hasattr(block.project, "conv"):
                convolution(block.project.conv, incoming)
    input_bytes = batch * 85 * 215 * 215 * 4
    activation_estimate = sum(activation_sizes) * 6 + input_bytes * 4 if training else max(activation_sizes) * 3 + input_bytes * 2
    device_working = state_bytes * (4 if training else 1) + activation_estimate
    cpu = torch.device(settings["device"]).type == "cpu"
    host_working = device_working if cpu else state_bytes * 3 + input_bytes * 4
    plan = {"parameter_count": parameters, "state_tensor_bytes": state_bytes,
            "input_bytes": input_bytes, "activation_working_estimate_bytes": activation_estimate,
            "estimated_host_working_bytes": host_working, "estimated_gpu_working_bytes": device_working,
            "checkpoint_capacity_estimate_bytes": (parameters * 16 + 8 * 2**20) * 4,
            "capacity_measured": False, "requires_actual_capacity_measurement": True,
            "estimate_basis": "actual MONAI meta parameter shapes; analytic convolution/pool sizes; conservative activation multipliers"}
    del model
    return plan


def admit_model(config, training=True):
    plan = model_resource_plan(config, training)
    allocation = config["classification"].get("estimated_host_working_bytes", plan["estimated_host_working_bytes"])
    if not isinstance(allocation, int) or allocation <= 0:
        raise ValueError("提供的主存估计需为正整数；省略时自动从真实网络结构估计")
    result = admit_stage(config, "classification_model_working_set", {"model_optimizer_activation_estimate": allocation})
    return {**result, "plan": plan}


def memory_state(config):
    memory = psutil.virtual_memory()
    rss = psutil.Process().memory_info().rss
    limit = min(int(config["max_memory_bytes"]), 100_000_000_000)
    fraction = min(float(config["host_memory_fraction"]), 0.8)
    return {"rss_bytes": rss, "limit_bytes": limit, "host_total_bytes": memory.total,
            "host_available_bytes": memory.available, "host_fraction": fraction,
            "exceeded": rss > limit or memory.available < memory.total * (1 - fraction)}


@contextmanager
def memory_guard(config):
    stop = threading.Event()
    initial = memory_state(config)
    if initial["exceeded"]:
        raise MemoryError("实际主存已超任务或整机80%预算")
    result = {"peak_rss_bytes": initial["rss_bytes"], "sample_interval_seconds": 0.05,
              "protection": "sampled_process_termination", "hard_os_memory_limit": False}

    def watch():
        while not stop.wait(0.05):
            state = memory_state(config)
            result["peak_rss_bytes"] = max(result["peak_rss_bytes"], state["rss_bytes"])
            if state["exceeded"]:
                event("memory_budget_exceeded", **state)
                # 预处理没有可提交的训练状态，立即终止本进程，父监督器收尾。
                os._exit(75)

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    try:
        yield result
    finally:
        stop.set()
        watcher.join()
