import pytest
from pydantic import ValidationError

from academic_agent.config import create_project, load_config
from academic_agent.models import ProjectConfig, Work


@pytest.mark.parametrize("value", [{"max_candidates": 0}, {"model_workers": 4}, {"backend": "unknown"}])
def test_config_rejects_invalid_runtime_limits(value):
    with pytest.raises(ValidationError):
        ProjectConfig(title="test", **value)


def test_project_roundtrip_and_no_overwrite(tmp_path):
    create_project(tmp_path, ProjectConfig(title="原始项目"))
    with pytest.raises(ValueError):
        create_project(tmp_path, ProjectConfig(title="覆盖"))
    assert load_config(tmp_path).title == "原始项目"


def test_unknown_publication_status_is_not_claimed_checked():
    work = Work(title="No status available")
    assert work.publication_checks == []
    assert not work.retracted
