from collections import Counter
from copy import deepcopy
from pathlib import Path
from time import perf_counter

from ..artifacts.store import digest, operation_lock, read_json, require_output, write_json
from ..datasets.splits import select_test
from ..observability.events import event, logged
from ..preprocessing.prepare import validate_prepared_config
from .infer import predict
from .metrics import evaluate


@logged("testing")
def run_test(config, test_set, group=None, case_ids=None):
    started = perf_counter()
    training = read_json(Path(config["run_dir"]) / "run.json")
    prepared = training["prepared"]
    validate_prepared_config(prepared, config)
    manifest = select_test(prepared["manifest"], test_set, group, case_ids)
    counts = Counter(row.get("test_group", "source_test") for row in manifest["cases"])
    event(config["run_dir"], "test_selection", test_set=test_set, groups=dict(counts), cases=len(manifest["cases"]))
    result_root = config["result_root"] or config["work_root"]
    base = Path(result_root) / config["dataset"] / config["model"] / f"fold_{config['fold']}" / Path(config["run_dir"]).name / "tests" / test_set
    owner = require_output(Path(config["work_root"]) / "tests" / digest(str(base))[:16], config["work_root"],
                           [prepared["manifest"]["data_root"]])
    with operation_lock(owner):
        prediction = predict(config, base / "predictions", manifest=manifest)
        receipts = read_json(base / "predictions" / "predictions.json")["cases"]
        checkpoints = {receipts[row["case_id"]]["checkpoint_sha256"] for row in manifest["cases"]}
        if len(checkpoints) != 1:
            raise ValueError("同次测试的预测使用了不同检查点")
        reports = {}
        for name in sorted(counts):
            event(config["run_dir"], "test_group_started", test_set=test_set, group=name, cases=counts[name])
            rows = [row for row in manifest["cases"] if row.get("test_group", "source_test") == name]
            all_rows = select_test(prepared["manifest"], test_set, name if test_set == "external" else None)["cases"]
            complete_group = {r["case_id"] for r in rows} == {r["case_id"] for r in all_rows}
            destination = base / name if test_set == "external" else base
            filename = "metrics.json" if complete_group else "selection-" + digest(sorted(r["case_id"] for r in rows))[:16] + ".json"
            selected = deepcopy(prepared)
            selected["manifest"] = {**manifest, "cases": rows}
            metrics = evaluate(base / "predictions", selected, destination / filename, config)
            reports[name] = {"cases": len(rows), "complete_group": complete_group, "report": str(destination / filename),
                             "aggregate": metrics["aggregate"], "reused": metrics["reused"]}
            event(config["run_dir"], "test_group_completed", test_set=test_set, group=name,
                  cases=len(rows), reused=metrics["reused"], report=str(destination / filename))
        summary = {"state": "completed", "test_set": test_set, "scope": config["scope"],
                   "cases": len(manifest["cases"]), "groups": reports, "checkpoint_sha256": next(iter(checkpoints)),
                   "new_predictions": prediction["new"], "reused_predictions": prediction["reused"],
                   "seconds": perf_counter() - started, "results": str(base)}
        write_json(owner / "latest.json", summary)
        event(config["run_dir"], "test", test_set=test_set, cases=summary["cases"], groups=dict(counts),
              checkpoint_sha256=summary["checkpoint_sha256"], seconds=summary["seconds"])
        return summary
