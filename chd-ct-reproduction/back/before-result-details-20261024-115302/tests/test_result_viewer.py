"""Historical reports must remain viewable without ML packages, writes or a GPU."""

import json
import shlex
import subprocess
import sys

import pytest

from chd_ct.server import inspection, jobs, launcher, panel, results


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def snapshot(root):
    return {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()}


@pytest.fixture
def completed(tmp_path):
    runtime, root = tmp_path / "runtime", tmp_path / "results"
    job = root / "20261010_old run's_workflow"
    test = job / "steps/06_test"
    save(job / "plan.json", dict(runtime=str(runtime), steps=[dict(name="test", output=str(test))]))
    save(
        job / "job_state.json",
        dict(status="passed", steps=[dict(name="test", status="passed", output=str(test))]),
    )
    save(
        test / "test/evaluation/evaluation.json",
        dict(
            status="passed",
            split="test",
            cases_evaluated=2,
            mean_case_foreground_dice=0.75,
            cases=[dict(dice=dict(LV=0.8, MYO=None)), dict(dice=dict(LV=0.6, MYO=0.4))],
        ),
    )
    save(
        test / "test/predictions/prediction-report.json", dict(status="passed", model_mode="train", cases=[])
    )
    save(runtime / "last-job.json", dict(directory=str(job)))
    return runtime, root, job, test


def test_old_single_split_results_cli_without_site_packages_is_readonly(completed):
    _, root, job, _ = completed
    before = snapshot(root.parent)
    command = [
        sys.executable,
        "-S",
        str(launcher.ROOT / "start_server.py"),
        "--query",
        "results",
        "--job-dir",
        str(job),
    ]
    response = subprocess.run(command, capture_output=True, encoding="utf-8", timeout=15)
    assert response.returncode == 0, response.stderr
    assert "0.7500（75.00%）" in response.stdout
    assert "左心室 (LV)： 0.7000" in response.stdout
    assert "心肌 (MYO)： 0.4000" in response.stdout
    assert "旧版清单" in response.stdout and "结果概览" in response.stdout
    assert before == snapshot(root.parent)
    # The printed Bash command preserves apostrophes and spaces in the selected path.
    query_line = next(
        line.strip() for line in response.stdout.splitlines() if line.strip().startswith("python3 ")
    )
    assert shlex.split(query_line)[-2:] == ["--job-dir", str(job)]


def test_threefold_null_metrics_and_smoke_are_labeled(completed, capsys):
    _, _, job, test = completed
    save(
        test / "test/evaluation/evaluation.json",
        dict(
            status="passed",
            split="cv",
            cases_evaluated=6,
            mean_case_foreground_dice=0.9,
            dice_by_class=dict(LV=0.95, MYO=None),
            protocols=["out_of_fold"],
            space="prepared_grid",
            folds={"fold_0": dict(cases_evaluated=2, mean_case_foreground_dice=0.8)},
        ),
    )
    save(
        test / "test/predictions/prediction-report.json", dict(status="passed", model_mode="smoke", cases=[])
    )
    results.show_results(job)
    output = capsys.readouterr().out
    assert "三折独立测试（OOF）" in output and "fold_0：2 例" in output
    assert "心肌 (MYO)： —" in output and "仅用于软件验证" in output
    assert "不是诊断准确率" in output and "预处理网格" in output


def test_failed_missing_and_bad_json_do_not_claim_final_scores(completed, capsys):
    _, _, job, test = completed
    (test / "test/evaluation/evaluation.json").write_text("{broken", encoding="utf-8")
    save(test / "test/predictions/prediction-report.json", dict(status="pending_cpu", cases=[]))
    results.show_results(job)
    output = capsys.readouterr().out
    assert "无法读取" in output and "等待 CPU 后处理" in output
    assert "0.0000" not in output and "平均前景 Dice" not in output
    (test / "test/evaluation/evaluation.json").unlink()
    results.show_results(job)
    assert "尚无可读取的评估文件" in capsys.readouterr().out


def test_result_browser_selects_old_task_even_after_runtime_change(completed, monkeypatch, capsys):
    runtime, root, job, _ = completed
    latest = root / "20261019_latest_workflow"
    save(latest / "plan.json", dict(runtime=str(runtime), steps=[]))
    save(latest / "job_state.json", dict(name="最新任务", status="running", steps=[]))
    save(runtime / "last-job.json", dict(directory=str(latest)))
    plan = results.read(job / "plan.json")
    plan["runtime"] = str(runtime.parent / "old-runtime")
    save(job / "plan.json", plan)
    before = snapshot(root.parent)
    assert results.result_history(runtime, root) == [latest, job]
    assert job not in inspection.history(runtime, root)  # Resume rules stay strict.
    monkeypatch.setattr(jobs, "acquire_lock", lambda *a: pytest.fail("acquired training lock"))
    monkeypatch.setattr(launcher, "query_gpus", lambda: pytest.fail("queried GPU"))
    monkeypatch.setattr("builtins.input", lambda _: "2")
    panel.results_menu(runtime, root)
    assert "0.7500" in capsys.readouterr().out
    assert before == snapshot(root.parent)


def test_panel_result_action_remains_available_while_running(completed, monkeypatch):
    runtime, root, job, _ = completed
    args = launcher.make_parser().parse_args(["--runtime", str(runtime)])
    monkeypatch.setattr(inspection, "job_active", lambda *a: True)
    answers = iter(["o", "", "0"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    seen = []
    monkeypatch.setattr(results, "show_results", lambda path: seen.append(path))
    assert panel.run(args, dict(runtime=str(runtime), results=str(root))) is False
    assert seen == [job]


def test_manual_directory_old_plan_and_moved_output(completed, monkeypatch, capsys):
    runtime, root, job, test = completed
    (job / "job_state.json").unlink()
    plan = results.read(job / "plan.json")
    plan["steps"][0]["output"] = "/old/location/steps/06_test"
    save(job / "plan.json", plan)
    answers = iter(["d", str(job)])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    panel.results_menu(runtime, root)
    assert "0.7500" in capsys.readouterr().out
    assert results.steps_for(job)[0]["output"] == test


def test_filter_and_unknown_fields_do_not_mask_known_reports(completed, capsys):
    _, _, job, test = completed
    save(
        test / "test/evaluation/evaluation.json",
        dict(
            status="failed",
            error="incomplete",
            cases=[],
            mean_case_foreground_dice=float("nan"),
            dice_by_class={"LV": float("inf"), "RV": "bad"},
        ),
    )
    results.show_results(job, "test")
    output = capsys.readouterr().out
    assert "尚未完整通过" in output and "incomplete" in output
    assert "nan" not in output and "inf" not in output and "bad" not in output
    with pytest.raises(ValueError, match="没有此步骤"):
        results.show_results(job, "not-selected")


def test_empty_query_and_follow_dont_create_runtime(completed, tmp_path, monkeypatch, capsys):
    _, _, job, _ = completed
    missing = tmp_path / "absent"
    profile = dict(runtime=str(missing), results=str(tmp_path / "missing-results"))
    args = launcher.make_parser().parse_args(["--query", "results"])
    assert inspection.query(args, profile) == 0
    assert not missing.exists()
    assert "没有任务记录" in capsys.readouterr().out
    args.job_dir, args.follow = str(job), True
    monkeypatch.setattr(inspection.time, "sleep", lambda _: (_ for _ in ()).throw(KeyboardInterrupt()))
    assert inspection.query(args, profile) == 0
    assert "退出结果查看" in capsys.readouterr().out


def test_missing_prediction_and_diagnosis_denominators(completed, capsys):
    _, _, job, test = completed
    save(
        test / "test/predictions/prediction-report.json",
        dict(status="passed", cases=[dict(prediction="gone.nii.gz")]),
    )
    results.show_results(job)
    assert "1 例的最终预测文件目前不可访问" in capsys.readouterr().out
    path = job / "diagnosis-metrics.json"
    save(
        path,
        dict(
            status="passed",
            cases_evaluated=2,
            split="test",
            classifier_micro=dict(accuracy=0.8, coverage=0.5),
        ),
    )
    results.diagnosis(path)
    output = capsys.readouterr().out
    assert "80.00%" in output and "50.00%" in output and "患者-疾病对" in output


def test_standalone_evaluate_shows_prediction_model_provenance(completed, capsys):
    _, _, job, test = completed
    output = job / "steps/07_evaluate"
    source = test / "test/predictions"
    save(source / "prediction-report.json", dict(status="passed", model_mode="smoke", cases=[]))
    save(
        output / "evaluate/evaluation.json",
        dict(status="passed", cases_evaluated=2, mean_case_foreground_dice=0.9),
    )
    save(
        job / "plan.json",
        dict(steps=[dict(name="evaluate", output=str(output), arguments=dict(predictions=str(source)))]),
    )
    save(
        job / "job_state.json",
        dict(status="passed", steps=[dict(name="evaluate", output=str(output), status="passed")]),
    )
    results.show_results(job, "evaluate")
    assert "不能当作正式实验结果" in capsys.readouterr().out
    # Old records with no provenance must not look like proven formal scores.
    (source / "prediction-report.json").unlink()
    results.show_results(job, "evaluate")
    assert "无法确认是否为短测结果" in capsys.readouterr().out


def test_plan_only_commands_use_supported_readonly_entries(completed, capsys):
    _, _, job, _ = completed
    (job / "job_state.json").unlink()
    (job / "launcher.log").write_text("old log", encoding="utf-8")
    results.show_results(job)
    output = capsys.readouterr().out
    assert "--query models" not in output and "--query logs" not in output
    assert "tail -n 60" in output and "--query results" in output


def test_legacy_threefold_training_and_step_filter(completed, capsys):
    _, _, job, _ = completed
    train = job / "steps/05_train"
    save(
        train / "train/report.json",
        dict(
            task="chd_threefold",
            mode="train",
            folds={
                f"fold_{i}": dict(status="complete", train_cases=10, val_cases=2, test_cases=5)
                for i in range(3)
            },
        ),
    )
    results.show_results(job, "train")
    output = capsys.readouterr().out
    assert "3/3 折" in output and "10/2/5" in output
    assert "--query models" not in output  # The synthesized step isn't in old job_state.
    assert "--query results" in output


def test_bad_neighbor_and_history_index_dont_hide_good_results(completed):
    runtime, root, job, _ = completed
    broken = root / "20261011_bad_workflow"
    save(broken / "plan.json", [])
    save(runtime / "job-history.json", {"unexpected": "not a list"})
    assert job in results.result_history(runtime, root)
    save(runtime / "job-history.json", [None, 42, {}, str(job)])
    assert job in results.result_history(runtime, root)
