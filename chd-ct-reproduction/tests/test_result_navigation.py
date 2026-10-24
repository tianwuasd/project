"""The result center keeps detail available without flooding the first screen."""

import json
import subprocess
import sys

import pytest

from chd_ct.server import launcher, result_menu, result_views
from chd_ct.server.prompts import UserExit


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def snapshot(root):
    return {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()}


@pytest.fixture
def job(tmp_path):
    root = tmp_path / "sample_workflow"
    steps = [
        dict(name=n, output=str(root / "steps" / n), status="passed")
        for n in ("test", "diagnose", "diagnosis-evaluate", "train")
    ]
    save(root / "plan.json", dict(name="示例任务", steps=steps))
    save(root / "job_state.json", dict(name="示例任务", status="passed", steps=steps))
    cases = [
        dict(
            case_id=f"PRIVATE_CASE_{i}",
            inference_protocol="out_of_fold",
            fold_indices=[i % 3],
            mean_foreground_dice=0.8,
            dice=dict(LV=0.9),
        )
        for i in range(30)
    ]
    save(
        root / "steps/test/test/evaluation/evaluation.json",
        dict(
            status="passed",
            split="cv",
            cases_evaluated=30,
            protocols=["out_of_fold"],
            mean_case_foreground_dice=0.8,
            cases=cases,
        ),
    )
    save(root / "steps/test/test/predictions/prediction-report.json", dict(model_mode="smoke", cases=cases))
    save(
        root / "steps/diagnose/diagnose/diagnosis.json",
        dict(
            status="passed",
            split="test",
            method="tree",
            classifier_blank_policy="unknown",
            cases=[
                dict(
                    case_id="DECISION_CASE",
                    classifier={
                        "VSD": dict(
                            status="indeterminate",
                            reason="missing_required_feature",
                            path=[dict(feature="contact_LV_RV", observed=None)],
                        )
                    },
                )
            ],
        ),
    )
    save(
        root / "steps/diagnosis-evaluate/diagnosis-evaluate/diagnosis-metrics.json",
        dict(
            status="passed",
            split="test",
            cases_evaluated=1,
            blank_policy="unknown",
            classifier_micro=dict(accuracy=0.8, coverage=0.5, known=10, decided=5, abstained=5),
        ),
    )
    return root


def test_center_opens_only_status_and_sections(job, monkeypatch, capsys):
    before = snapshot(job)
    monkeypatch.setattr("builtins.input", lambda _: "b")
    result_menu.browse_results(job)
    out = capsys.readouterr().out
    assert "结果中心" in out and "分割结果" in out and "文件与来源" in out
    assert "PRIVATE_CASE" not in out and "0.8000" not in out and "contact_LV_RV" not in out
    assert before == snapshot(job)


def test_simple_segmentation_preserves_scope_and_software_warning(job):
    out = result_views.render(job, "segmentation", "summary")
    assert "fold_0" in out and "0.8000" in out and "30" in out and "cv" in out
    assert "软件验证" in out and "OOF" in out
    assert "PRIVATE_CASE" not in out and "叶节点" not in out and "可复制" not in out
    detail = result_views.render(job, "segmentation", "detailed")
    assert "PRIVATE_CASE_29" in detail and "左心室" in detail and "有效病例" in detail


def test_decisions_and_evaluation_are_separate_and_show_denominators(job):
    out = result_views.render(job, "decisions", "summary")
    assert "VSD" in out and "证据不足" in out and "不是准确率" in out
    assert "DECISION_CASE" not in out and "contact_LV_RV" not in out
    metrics = result_views.render(job, "diagnosis-evaluation", "summary")
    assert "覆盖率" in metrics and "已知 10" in metrics and "弃判 5" in metrics
    assert "未知" in metrics and "PRIVATE_CASE" not in metrics
    assert "contact_LV_RV" in result_views.render(job, "decisions", "detailed")


def test_navigation_toggle_paging_and_back_keep_original_bytes(job, monkeypatch, capsys):
    before = snapshot(job)
    answers = iter(["1", "d", "n", "p", "s", "b", "b"])
    screens = []

    def respond(_):
        screens.append(capsys.readouterr().out)
        return next(answers)

    monkeypatch.setattr("builtins.input", respond)
    result_menu.browse_results(job)
    assert "简洁" in screens[1] and "PRIVATE_CASE" not in screens[1]
    assert "详细" in screens[2] and "第 2/" in screens[3] and "第 1/" in screens[4]
    assert "简洁" in screens[5] and "结果中心" in screens[6]
    assert before == snapshot(job)


def test_quit_and_interrupt_do_not_stop_jobs(job, monkeypatch):
    before = snapshot(job)
    monkeypatch.setattr("builtins.input", lambda _: "q")
    with pytest.raises(UserExit):
        result_menu.browse_results(job)
    answers = iter(["1", KeyboardInterrupt(), "b"])

    def respond(_):
        value = next(answers)
        if isinstance(value, BaseException):
            raise value
        return value

    monkeypatch.setattr("builtins.input", respond)
    result_menu.browse_results(job)
    assert before == snapshot(job)


def test_long_lines_are_wrapped_and_pages_bounded():
    pages = result_menu.paginate("中文" * 200 + "\n" + "\n".join(str(i) for i in range(50)), width=60)
    assert len(pages) > 2
    assert all(len(page) <= result_menu.PAGE_LINES for page in pages)
    assert all(result_menu.display_width(line) <= 60 for page in pages for line in page)


def test_cli_simple_default_and_explicit_detail_without_ml(job):
    before = snapshot(job)
    command = [
        sys.executable,
        "-S",
        str(launcher.ROOT / "start_server.py"),
        "--query",
        "results",
        "--job-dir",
        str(job),
        "--result-section",
        "segmentation",
    ]
    simple = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", timeout=15)
    assert simple.returncode == 0, simple.stderr
    assert "0.8000" in simple.stdout and "PRIVATE_CASE" not in simple.stdout
    detail = subprocess.run(
        command + ["--result-detail", "detailed"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=15,
    )
    assert detail.returncode == 0, detail.stderr
    assert "PRIVATE_CASE_29" in detail.stdout and before == snapshot(job)


def test_missing_reports_and_mismatched_filters_are_explicit(job):
    (job / "steps/test/test/evaluation/evaluation.json").write_text("{")
    out = result_views.render(job, "segmentation", "summary")
    assert "无法读取" in out and "0.0000" not in out
    with pytest.raises(ValueError, match="没有此步骤"):
        result_views.render(job, step="absent")
    out = result_views.render(job, "decisions", "summary", step="test")
    assert "没有" in out and "VSD" not in out


def test_files_page_keeps_paths_out_of_other_simple_sections(job):
    out = result_views.render(job, "files", "detailed")
    assert "evaluation.json" in out and "--query results" in out
    assert "PRIVATE_CASE" not in out and "平均前景 Dice" not in out
    assert str(job) not in result_views.render(job, "decisions", "summary")


@pytest.mark.parametrize(
    "extra", [["--result-detail", "detailed"], ["--query", "status", "--result-section", "files"]]
)
def test_result_controls_cannot_trigger_work_without_result_query(extra, capsys):
    assert launcher.main(extra) == 1
    assert "需要配合 --query results" in capsys.readouterr().err


def test_diagnosis_training_keeps_classifier_software_metadata(job):
    output = job / "steps/10_diagnosis-train/diagnosis-train"
    save(
        output / "trainability.json",
        dict(status="passed", trained_diseases=1, training_cases=4, validation_cases=1),
    )
    save(
        output / "classifier.json",
        dict(segmentation_mode="smoke", software_only=True, blank_policy="negative"),
    )
    out = result_views.render(job, "training", "summary")
    assert "仅用于软件验证" in out and "空白按阴性" in out


def test_synthetic_training_detail_remains_reachable_in_its_section(job):
    output = job / "steps/14_diagnosis-demo/diagnosis-demo/training"
    save(
        output / "trainability.json",
        dict(status="passed", diseases={"VSD": dict(status="not_trainable", reason="no positives")}),
    )
    out = result_views.render(job, "training", "detailed")
    assert "合成数据" in out and "VSD" in out and "不能训练" in out


def test_prior_only_decision_is_not_hidden_by_summary(job):
    save(
        job / "steps/diagnose/diagnose/diagnosis.json",
        dict(
            status="passed",
            cases=[dict(classifier={"VSD": dict(status="indeterminate", reason="no_split_training_prior")})],
        ),
    )
    out = result_views.render(job, "decisions", "summary")
    assert "1 项仅为训练先验" in out and "没有个体判断分支" in out


def test_training_summary_hides_paths_and_preserves_short_test_notice(job):
    output = job / "steps/train/train"
    save(
        output / "report.json",
        dict(
            task="chd_threefold",
            mode="smoke",
            folds={"fold_0": dict(status="complete", train_cases=4, val_cases=1, test_cases=2)},
        ),
    )
    out = result_views.render(job, "training", "summary")
    assert "软件验证" in out and "4/1/2" in out
    assert "fold_1：未记录" in out and str(job) not in out


def test_simple_prediction_pending_cpu_and_ensemble_are_not_fold_scores(job):
    path = job / "steps/test/test/evaluation/evaluation.json"
    save(
        path,
        dict(
            status="passed",
            split="all",
            protocols=["threefold_ensemble"],
            cases_evaluated=1,
            mean_case_foreground_dice=0.4,
        ),
    )
    out = result_views.render(job, "segmentation", "summary")
    assert "集成" in out and "fold_0：" not in out
    output = job / "steps/06_predict"
    save(output / "predict/prediction-report.json", dict(status="pending_cpu", split=None, cases=[]))
    out = result_views.render(job, "segmentation", "summary", step="predict")
    assert "等待 CPU 后处理" in out and "只预测，不计算 Dice" in out
