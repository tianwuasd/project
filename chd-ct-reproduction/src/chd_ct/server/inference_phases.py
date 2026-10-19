"""Internal GPU/CPU phases without adding user-visible workflow steps."""

from pathlib import Path

from ..recovery import file_hash, read_json


def gpu_ready(step):
    """Read-only resume check; CPU-only completion must not query CUDA."""
    args = step["arguments"]
    output = Path(step["output"]) / ("test/predictions" if args["task"] == "test" else "predict")
    try:
        receipt = read_json(output / "gpu-phase-ready.json")
        contract = read_json(output / "prediction-contract.json")
        prepared = Path(args["prepared"])
        models = Path(args["models"])
        models = models if models.is_file() else models / "models.json"
        if (
            receipt["contract"] != contract
            or contract["prepared_sha256"] != file_hash(prepared / "dataset.json")
            or contract["models_sha256"] != file_hash(models)
            or receipt["request"]
            != dict(
                split=args.get("split") or ("auto" if args["task"] == "test" else None),
                case_id=args.get("case_id"),
            )
        ):
            return False
        manifest = read_json(prepared / "dataset.json")
        caches = {r["case_id"]: r["cache"] for r in manifest["cases"]}
        for case, digest in contract["cache_hashes"].items():
            if file_hash(prepared / caches[case]) != digest:
                return False
        if set(receipt["cases"]) != set(contract["case_ids"]):
            return False
        for files in receipt["cases"].values():
            for name, digest in files.items():
                path = (output / name).resolve()
                if not path.is_relative_to(output.resolve()) or file_hash(path) != digest:
                    return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def operations(plan):
    """Keep CPU prerequisites in front; defer independent CPU tails until GPU end."""
    front, gpu, tail = [], [], []
    seen_gpu = False
    for index, step in enumerate(plan["steps"]):
        args = step["arguments"]
        uses_gpu = args.get("device") == "cuda" and not step.get("reuse")
        if uses_gpu and args["task"] in {"test", "predict"}:
            seen_gpu = True
            if not (args.get("resume") and gpu_ready(step)):
                gpu.append((index, "gpu"))
            tail.append((index, "cpu"))
        elif uses_gpu:
            seen_gpu = True
            gpu.append((index, "all"))
        else:
            (tail if seen_gpu else front).append((index, "all"))
    return front + gpu + tail


def normalize_caps(plan):
    """Fill legacy inference caps before creating the workflow holder registry."""
    cap = max((s["arguments"].get("gpu_count") or 1 for s in plan["steps"]), default=1)
    for step in plan["steps"]:
        args = step["arguments"]
        if args.get("task") in {"test", "predict"} and args.get("device") == "cuda":
            if args.get("gpu_count") is None:
                args["gpu_count"] = cap
