import argparse
import inspect
import json
from pathlib import Path
import sys
from time import perf_counter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--official-root', required=True)
    args = parser.parse_args()
    started = perf_counter()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
    from chd_segmentation.models.medsam3.environment import inspect_environment, verify_worker_environment
    from chd_segmentation.models.medsam3.weights import import_official
    environment = inspect_environment(sys.executable)
    verify_worker_environment(environment)
    official, lora, builder = import_official({'official_root': args.official_root})
    import torch
    import torchvision
    import triton
    import decord
    from sam3.train.matcher import BinaryHungarianMatcherV2, BinaryOneToManyMatcher
    from sam3.train.loss.loss_fns import Boxes, IABCEMdetr, Masks
    from sam3.train.loss.sam3_loss import Sam3LossWrapper
    from sam3.train.data.sam3_image_dataset import Datapoint, FindQueryLoaded, Object
    required = {'device', 'checkpoint_path', 'load_from_HF', 'bpe_path', 'eval_mode', 'compile'}
    assert required <= inspect.signature(builder.build_sam3_image_model).parameters.keys()
    assert all(callable(value) for value in (official.collate_fn_api, official.copy_data_to_device,
        lora.apply_lora_to_model, BinaryHungarianMatcherV2, BinaryOneToManyMatcher,
        Boxes, IABCEMdetr, Masks, Sam3LossWrapper, Datapoint, FindQueryLoaded, Object))
    assert torch.cuda.is_available()
    result = dict(environment=environment, torch=torch.__version__, torchvision=torchvision.__version__,
        triton=triton.__version__, decord=decord.__version__, cuda_runtime=torch.version.cuda,
        cuda_available=True, official_root=str(Path(args.official_root).resolve()),
        native_imports=True, native_interfaces=True, model_loaded=False, cases_read=0,
        cuda_kernel_executed=False, seconds=perf_counter()-started)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
