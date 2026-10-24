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


def detailed_case(case, fold, mean, **dice):
    return dict(
        case_id=case,
        inference_protocol="out_of_fold",
        fold_indices=[fold],
        mean_foreground_dice=mean,
        dice=dice,
    )


def test_fold_details_use_saved_case_metrics_and_valid_denominators(completed, capsys):
    _, root, job, step = completed
    save(
        step / "test/evaluation/evaluation.json",
        dict(
            status="passed",
            split="cv",
            cases_evaluated=3,
            protocols=["out_of_fold"],
            cases=[
                detailed_case("case_a", 0, 0.8, LV=0.8, MYO=None),
                detailed_case("case_b", 0, 0.5, LV=0.6, MYO=0.4),
                detailed_case("case_c", 1, 0.2, LV=0.2, MYO=0.2),
            ],
            folds={
                "fold_0": dict(
                    cases_evaluated=2, mean_case_foreground_dice=0.65, dice_by_class=dict(LV=0.7, MYO=0.4)
                )
            },
        ),
    )
    before = snapshot(root.parent)
    results.show_results(job)
    out = capsys.readouterr().out
    first = out.split("fold_0：2 例", 1)[1].split("fold_1：", 1)[0]
    assert "左心室 (LV)： 0.7000" in first and "有效病例 2/2" in first
    assert "心肌 (MYO)： 0.4000" in first and "有效病例 1/2" in first
    assert "case_a" in first and "case_b" in first and "case_c" not in first
    assert "fold_2：未记录" in out
    assert "0.5000（50.00%）" in out  # Per-case average; not equal weighting of fold averages.
    assert before == snapshot(root.parent)


def test_ensemble_is_not_reported_as_three_heldout_folds(completed, capsys):
    _, _, job, step = completed
    save(
        step / "test/evaluation/evaluation.json",
        dict(
            status="passed",
            split="test",
            cases_evaluated=1,
            protocols=["threefold_ensemble"],
            cases=[
                dict(
                    case_id="new_case",
                    inference_protocol="threefold_ensemble",
                    fold_indices=[0, 1, 2],
                    mean_foreground_dice=0.8,
                    dice=dict(LV=0.8),
                )
            ],
        ),
    )
    results.show_results(job)
    out = capsys.readouterr().out
    assert "集成模型的结果，不是三个独立测试折的分数" in out
    assert "fold_0：" not in out and "new_case" in out


def test_missing_fold_detail_never_invents_cases_or_denominators(completed, capsys):
    _, _, job, step = completed
    save(
        step / "test/evaluation/evaluation.json",
        dict(
            status="passed",
            split="cv",
            protocols=["out_of_fold"],
            folds={
                "fold_0": dict(cases_evaluated=2, mean_case_foreground_dice=0.8, dice_by_class=dict(LV=0.9))
            },
        ),
    )
    results.show_results(job)
    out = capsys.readouterr().out
    assert "病例名单：未记录" in out and "有效病例 未记录" in out
    assert "97" not in out and "33/" not in out
    assert "fold_1：未记录" in out and "fold_2：未记录" in out


def test_result_identity_and_prediction_only_scope(completed, capsys):
    _, _, job, _ = completed
    output = job / "steps/02_predict"
    source = job / "prepared"
    save(
        job / "plan.json",
        dict(
            steps=[
                dict(
                    name="predict",
                    output=str(output),
                    arguments=dict(prepared=str(source), models="/models/original/models.json"),
                )
            ]
        ),
    )
    save(
        output / "predict/prediction-report.json",
        dict(
            status="passed",
            model_mode="train",
            split=None,
            cases=[dict(case_id="case_x", inference_protocol="out_of_fold", fold_indices=[1])],
        ),
    )
    results.show_results(job, "predict")
    out = capsys.readouterr().out
    assert "菜单 6" in out and "独立预测" in out and "不计算 Dice" in out
    assert str(source) in out and "/models/original/models.json" in out
    assert "全部病例" in out and "case_x" in out


def test_synthetic_demo_cannot_be_presented_as_real_cohort(completed, capsys):
    _, _, job, _ = completed
    output = job / "steps/08_demo"
    save(
        output / "demo/evaluation/evaluation.json",
        dict(status="passed", split="test", cases_evaluated=1, mean_case_foreground_dice=0.9),
    )
    save(output / "demo/predictions/prediction-report.json", dict(model_mode="smoke", cases=[]))
    results.show_results(job, "demo")
    out = capsys.readouterr().out
    assert "菜单 13" in out and "合成数据" in out
    assert "0.9000" in out and "不代表真实 CT 分割精度" in out


def test_corrupt_fold_metadata_remains_readable(completed, capsys):
    _, _, job, step = completed
    save(
        step / "test/evaluation/evaluation.json",
        dict(
            status="failed",
            split=["cv"],
            protocols=[{}, "out_of_fold"],
            cases=[
                dict(
                    case_id="bad\x1b[31m",
                    inference_protocol=["out_of_fold"],
                    fold_indices={},
                    mean_foreground_dice=True,
                    dice=dict(LV=True),
                ),
                detailed_case("case_ok", 0, 0.6, LV=0.6),
            ],
            folds={"fold_0": [], "fold_1": dict(cases_evaluated=2, dice_by_class=dict(LV=float("nan")))},
        ),
    )
    results.show_results(job)
    out = capsys.readouterr().out
    assert "尚未完整通过" in out and "case_ok" in out
    assert "nan" not in out and "\x1b" not in out


def test_decisions_display_actual_paths_missing_evidence_and_leaf(completed, capsys):
    _, root, job, _ = completed
    output = job / "steps/11_diagnose"
    save(
        output / "diagnose/diagnosis.json",
        dict(
            status="passed",
            split="test",
            method="both",
            segmentation_mode="train",
            classifier_blank_policy="negative",
            cases=[
                dict(
                    case_id="case_a",
                    patient_id="patient_a",
                    split="test",
                    rule_evidence_source=None,
                    classifier={
                        "VSD": dict(
                            status="positive",
                            probability=0.75,
                            path=[
                                dict(
                                    feature="fraction_LV",
                                    observed=0.41,
                                    used_value=0.41000001,
                                    operator=">",
                                    threshold=0.2,
                                )
                            ],
                            leaf=dict(id=3, samples=8, positive=6),
                        ),
                        "ASD": dict(
                            status="indeterminate",
                            reason="missing_required_feature",
                            path=[dict(feature="contact_LA_RA", observed=None, threshold=0.5)],
                        ),
                    },
                    rules={
                        "VSD": dict(status="indeterminate", reason="missing_evidence", missing=["comm_LV_RV"])
                    },
                )
            ],
        ),
    )
    before = snapshot(root.parent)
    results.show_results(job, "diagnose")
    out = capsys.readouterr().out
    assert "菜单 11" in out and "case_a" in out and "VSD：阳性候选" in out
    assert "fraction_LV" in out and "0.4100" in out and "> 0.2000" in out
    assert "6/8" in out and "75.00%" in out and "未经临床风险校准" in out
    assert "ASD：证据不足" in out and "comm_LV_RV" in out
    assert "空白按阴性" in out and before == snapshot(root.parent)


def test_diagnosis_metrics_include_per_disease_denominators(completed, capsys):
    _, _, job, _ = completed
    output = job / "steps/12_diagnosis-evaluate"
    metrics = dict(
        known=5,
        unknown_truth=1,
        decided=3,
        abstained=2,
        tp=1,
        tn=1,
        fp=1,
        fn=0,
        accuracy=2 / 3,
        coverage=0.6,
        sensitivity=1.0,
        specificity=0.5,
        precision=0.5,
        f1=2 / 3,
        auroc=None,
        correct_fraction_all_known=0.4,
    )
    save(
        output / "diagnosis-evaluate/diagnosis-metrics.json",
        dict(
            status="passed",
            split="test",
            cases_evaluated=6,
            classifier_micro=metrics,
            classifier={"VSD": metrics},
            rules={},
            blank_policy="negative",
            assumed_negative_cells=4,
            segmentation_mode="smoke",
            software_only=True,
        ),
    )
    results.show_results(job, "diagnosis-evaluate")
    out = capsys.readouterr().out
    assert "VSD" in out and "已知真值 5" in out and "已判定 3" in out and "弃判 2" in out
    assert "TP/TN/FP/FN：1/1/1/0" in out and "全部已知真值中的正确比例" in out
    assert "60.00%" in out and "仅用于软件验证" in out
    assert "空白按阴性" in out and "4" in out


def test_decision_training_displays_untrainable_diseases(completed, capsys):
    _, _, job, _ = completed
    output = job / "steps/10_diagnosis-train"
    save(
        output / "diagnosis-train/trainability.json",
        dict(
            status="passed",
            trained_diseases=1,
            training_cases=8,
            validation_cases=2,
            test_cases_used=0,
            blank_policy="negative",
            diseases={
                "VSD": dict(status="trained", positive=4, negative=4, unknown=0, max_depth=2),
                "ASD": dict(
                    status="not_trainable",
                    positive=1,
                    negative=7,
                    unknown=0,
                    reason="训练集至少需要 2 个阳性和 2 个阴性",
                ),
            },
        ),
    )
    results.show_results(job, "diagnosis-train")
    out = capsys.readouterr().out
    assert "VSD" in out and "深度 2" in out and "ASD" in out and "不能训练" in out
    assert "至少需要 2 个阳性" in out and "训练/验证病例：8/2" in out


def test_external_prediction_is_not_replaced_by_same_named_local_step(completed, capsys):
    _, _, job, _ = completed
    external = job.parent / "external_workflow/steps/06_predict/predict"
    local = job / "steps/06_predict/predict"
    save(external / "prediction-report.json", dict(model_mode="smoke"))
    save(local / "prediction-report.json", dict(model_mode="train"))
    output = job / "steps/07_evaluate"
    save(output / "evaluate/evaluation.json", dict(status="passed", cases_evaluated=1))
    save(
        job / "plan.json",
        dict(
            steps=[
                dict(name="predict", output=str(local.parent)),
                dict(name="evaluate", output=str(output), arguments=dict(predictions=str(external))),
            ]
        ),
    )
    results.show_results(job, "evaluate")
    out = capsys.readouterr().out
    assert "不能当作正式实验结果" in out and str(external) in out
    assert "模型来源模式：正式训练" not in out


def test_boolean_rule_evidence_and_diagnosis_cli_need_no_ml(completed):
    _, root, job, _ = completed
    output = job / "steps/11_diagnose"
    save(
        output / "diagnose/diagnosis.json",
        dict(
            status="passed",
            method="rules",
            cases=[
                dict(
                    case_id="case_bool",
                    rules={
                        "VSD": dict(
                            status="positive",
                            evidence=[
                                dict(
                                    feature="conn_LV_RV",
                                    observed=True,
                                    operator="eq",
                                    threshold=True,
                                    satisfied=True,
                                )
                            ],
                        )
                    },
                )
            ],
        ),
    )
    before = snapshot(root.parent)
    response = subprocess.run(
        [
            sys.executable,
            "-S",
            str(launcher.ROOT / "start_server.py"),
            "--query",
            "results",
            "--job-dir",
            str(job),
            "--step",
            "diagnose",
        ],
        capture_output=True,
        encoding="utf-8",
        timeout=15,
    )
    assert response.returncode == 0, response.stderr
    assert "观测 1（是）" in response.stdout and "= 1（是）" in response.stdout
    assert before == snapshot(root.parent)


def test_partial_case_details_do_not_claim_complete_denominator(completed, capsys):
    _, _, job, step = completed
    save(
        step / "test/evaluation/evaluation.json",
        dict(
            status="passed",
            cases_evaluated=3,
            protocols=["out_of_fold"],
            cases=[detailed_case("one_saved", 0, 0.6, LV=0.6)],
            folds={
                "fold_0": dict(cases_evaluated=3, mean_case_foreground_dice=0.8, dice_by_class=dict(LV=0.8))
            },
        ),
    )
    results.show_results(job)
    out = capsys.readouterr().out
    assert "已保存明细 1 例，与登记总数不一致" in out
    assert "有效病例 未记录" in out and "有效病例 1/3" not in out


@pytest.mark.parametrize(
    "cases",
    [
        [dict(mean_foreground_dice=1.0, dice=dict(LV=1.0))],
        [dict(mean_foreground_dice=1.0, dice=dict(LV=1.0)), dict(dice={})],
        [
            dict(mean_foreground_dice=1.0, dice=dict(LV=1.0)),
            dict(mean_foreground_dice=True, dice=dict(LV=True)),
        ],
    ],
)
def test_partial_or_invalid_saved_values_cannot_fill_whole_summary(cases):
    from chd_ct.server.result_metrics import summary

    value, _ = summary(dict(cases_evaluated=2), cases, results.LABELS)
    assert "mean_case_foreground_dice" not in value and "LV" not in value["dice_by_class"]


def test_partial_cache_metadata_does_not_claim_zero_cases_or_available_truth(tmp_path, capsys):
    save(tmp_path / "dataset.json", dict(discovered=8))
    results.cache_details(str(tmp_path))
    out = capsys.readouterr().out
    assert "保留 未记录" in out and "排除记录 未记录" in out
    assert "无法确认真值可用性" in out
