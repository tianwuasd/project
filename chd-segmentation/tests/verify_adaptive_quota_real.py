import argparse
from copy import deepcopy
from pathlib import Path
import sys
from time import perf_counter

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from smoke import SmokeWorkspace
from chd_segmentation.artifacts.store import read_json, write_json
from chd_segmentation.config import DEFAULTS
from chd_segmentation.models.stunet.streaming import BlockWorkspace, resize_image
from chd_segmentation.training.memory_guard import MemoryGuard


def run(args):
    with SmokeWorkspace(args.work_root, args.report) as test:
        row = next(r for r in read_json(args.manifest)["cases"] if r["case_id"] == "ct_1003")
        image = nib.load(Path(args.data_root) / row["image"])
        source = np.asarray(image.dataobj[240:304, 240:304, 87:151], dtype=np.float32)
        config = deepcopy(DEFAULTS)
        config.update(scope="technical", work_root=str(test.root / "course"), run_dir=str(test.root / "course" / "quota"))
        config["resources"].update(ram_fraction=1., reserve_ram_bytes=256 * 1024 ** 2)
        config["preprocessing"].update(block_bytes=256 * 1024, max_block_bytes=8 * 1024 ** 2,
                                       max_case_threads=2, growth_seconds=.01)
        result = {"case_id": row["case_id"], "roi_xyz": [[240, 304], [240, 304], [87, 151]],
                  "quota_reservation_bytes": 240 * 1024 ** 2, "optimizer_updates": 0,
                  "scope": "真实共享登记中的预留额度收紧及释放；不申请对应的大数组", "runs": []}
        test.results.append(result)
        with MemoryGuard(config, "adaptive_quota") as guard:
            baseline = guard.boundary()
            config["resources"]["memory_limit_bytes"] = baseline["project"]["accounted_bytes"] + 256 * 1024 ** 2
            outputs = []
            for name, reserved in (("available", 0), ("reserved", result["quota_reservation_bytes"]), ("released", 0)):
                with guard.admit(reserved, label="next_stage_workspace", reserve=True) as ticket:
                    if ticket is None:
                        raise MemoryError("真实额度收紧检查无法获得所需预留")
                    workspace = BlockWorkspace(test.root / name, config, guard, row["case_id"])
                    began = perf_counter()
                    try:
                        output = workspace.create("result", (96, 96, 96), np.float32)
                        resize_image(source, output, 1, workspace, "coeff")
                        outputs.append(np.array(output, copy=True))
                        result["runs"].append({"name": name, "seconds": perf_counter() - began,
                                               **workspace.execution.metrics()})
                    finally:
                        workspace.close()
            np.testing.assert_array_equal(outputs[0], outputs[1])
            np.testing.assert_array_equal(outputs[0], outputs[2])
            peaks = [r["peak_job_bytes"] for r in result["runs"]]
            assert peaks[1] < peaks[0] and peaks[2] > peaks[1], peaks
            result.update(state="passed", outputs_bitwise_equal=True, peak_job_bytes=peaks)
        write_json(test.report / "quota.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("manifest", "data-root", "work-root", "report"):
        parser.add_argument("--" + name, required=True)
    run(parser.parse_args())
