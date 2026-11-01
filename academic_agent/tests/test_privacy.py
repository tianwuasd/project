import json
import os
from contextlib import ExitStack
from unittest.mock import MagicMock

import pytest
from langchain_core.tracers import langchain
from langchain_core.tracers.context import tracing_v2_callback_var, tracing_v2_enabled
from langsmith import Client, get_tracing_context, run_trees, tracing_context

from academic_agent.backends import DemoBackend
from academic_agent.config import atomic_json, load_config
from academic_agent.demo import seed_demo
from academic_agent.models import ModelResult
from academic_agent.workflow import Runner, WorkflowPaused


@pytest.mark.parametrize("flag", ["LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2"])
@pytest.mark.parametrize("fail_review", [False, True])
@pytest.mark.parametrize("tracing_parent", ["environment", "langsmith", "langchain"])
def test_workflow_never_inherits_external_tracing(tmp_path, monkeypatch, flag, fail_review, tracing_parent):
    # Synthetic-only data and fake clients; never construct an authenticated client.
    client = MagicMock(spec=Client)
    monkeypatch.setattr(langchain, "get_client", lambda: client)
    monkeypatch.setattr(run_trees, "get_cached_client", lambda **kwargs: client)
    network = MagicMock(side_effect=AssertionError("Unexpected network access"))
    monkeypatch.setattr("requests.sessions.Session.request", network)
    monkeypatch.setattr("httpx.Client.send", network)
    for name in ("LANGSMITH_API_KEY", "LANGCHAIN_API_KEY"):
        monkeypatch.setenv(name, "synthetic-test-key")
    for name in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(flag, "true")
    sentinel = "PRIVATE_RESEARCH_MATERIAL_SENTINEL"
    class PartialReview(DemoBackend):
        reviews = 0
        def run(self, job):
            if job.kind == "review":
                self.reviews += 1
                if fail_review and self.reviews == 2:
                    return ModelResult(status="failed", backend=self.name, error="Synthetic review failure")
                result = super().run(job)
                for verdict in result.data["verdicts"]:
                    verdict["rationale"] = sentinel + " The original synthetic dataset contained 40 simulated jobs."
                return result
            return super().run(job)
    seed_demo(tmp_path)
    config = load_config(tmp_path)
    config.model_workers = 1
    atomic_json(tmp_path / "project.json", config.model_dump())
    runner = Runner(tmp_path, offline=True, backend=PartialReview())
    parent = run_trees.RunTree(name="Synthetic parent", run_type="chain", client=client)
    with ExitStack() as stack:
        if tracing_parent == "langsmith":
            stack.enter_context(tracing_context(enabled=True, parent=parent, client=client))
        if tracing_parent == "langchain":
            callback = stack.enter_context(tracing_v2_enabled(client=client))
        if fail_review:
            with pytest.raises(WorkflowPaused) as error:
                runner.run()
            message = str(error.value)
            assert sentinel not in message and "40 simulated" not in message and "verdicts" not in message
            failures = json.loads(message)["results"]
            assert len(failures) == 1 and failures[0]["status"] == "failed"
            assert set(failures[0]) == {"work_id", "status", "error"}
            assert sentinel in json.dumps(runner.store.logs("review"))  # Successful sibling existed.
        else:
            assert runner.run()["complete"]
            assert runner.run(resume=True)["already_complete"]
        if tracing_parent == "langsmith":
            assert get_tracing_context()["enabled"] is True
            assert get_tracing_context()["parent"] is parent
        if tracing_parent == "langchain":
            assert tracing_v2_callback_var.get() is callback
    assert os.environ[flag] == "true"  # Suppression is scoped, not a global setting edit.
    client.create_run.assert_not_called()
    client.update_run.assert_not_called()
    network.assert_not_called()
