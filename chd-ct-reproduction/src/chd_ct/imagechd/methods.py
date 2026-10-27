"""Scientific processing identities, independent of model architecture and hardware."""

LABEL_RESIZE = "nearest-exact-pixel-centers-v1"
PROCESSING = {
    "label_resize": LABEL_RESIZE,
    "blood_refinement": "exclude-boundary-grow-interior-v1",
}


def require_current_processing(record):
    if record.get("processing_method") != PROCESSING:
        raise ValueError(
            "模型的标签缩放/血池细化协议不同或未记录；请在独立新任务重新训练。"
            "历史结果仍可查看；旧实验使用原冻结代码，不混用新方法。"
        )


def require_aligned_cache(data):
    """Read legacy manifests freely, but never train on silently shifted labels."""
    if data.get("for_prediction"):
        return  # Image-only resizing did not use the old categorical sampler.
    resized = data.get("resolution") == "resized" or bool(data.get("size"))
    if resized and data.get("label_resize") != LABEL_RESIZE:
        raise ValueError("旧尺寸缩放缓存的标签采样未对齐；请从原始数据在新目录重新预处理")
    for row in data.get("cases", []):
        original = row.get("resampling", {}).get("resampled_shape", row.get("canonical_shape"))
        changed = original is not None and row.get("cache_shape") != original
        if (resized or changed) and row.get("label_resize") != LABEL_RESIZE:
            raise ValueError("病例缺少当前标签缩放协议；请在新目录重新预处理")
