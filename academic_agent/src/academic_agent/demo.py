"""Original synthetic fixtures bundled with the wheel; no third-party papers."""

from pathlib import Path

from .config import create_project
from .documents import parse_document
from .filesystem import io_path
from .models import ProjectConfig, PublicationCheck, Work
from .store import Store

FIXTURES = [
    ("Synthetic scheduling study A", """ORIGINAL_SYNTHETIC_FIXTURE
problem: This synthetic note compares waiting time in a small scheduling simulation.
method: A fixed-priority scheduler was compared with a round-robin baseline.
data: The original synthetic dataset contained 40 simulated jobs and no real participants.
result: Mean waiting time fell from 10 to 8 simulation units, a 20 percent decrease.
limitation: Only one simulated workload was tested; the result does not establish real-world effectiveness.
contribution: The note demonstrates a reproducible comparison procedure, without claiming field-wide novelty.
"""),
    ("Synthetic scheduling study B", """ORIGINAL_SYNTHETIC_FIXTURE
problem: This synthetic note tests whether priority scheduling improves all simulated workloads.
method: The same scheduling rule was tested under a bursty arrival pattern.
data: The original synthetic dataset contained 60 simulated jobs and no real participants.
result: Mean waiting time increased from 10 to 11 simulation units; performance did not improve.
limitation: The two synthetic workloads cannot establish a general trend across real systems.
contribution: The note provides a counterexample to an unconditional improvement claim.
"""),
]


def seed_demo(root: Path, *, backend: str = "demo") -> None:
    create_project(root, ProjectConfig(title="合成调度研究示例", question="不同负载下，调度方法的效果和局限是什么？",
                                      sources=[], backend=backend, max_model_calls=12))
    store = Store(root)
    inputs = root / "inputs"
    io_path(inputs).mkdir()
    for index, (title, text) in enumerate(FIXTURES, 1):
        work = store.upsert_work(Work(title=title, authors=["Academic Agent original fixture"], year=2026,
                                     source_ids={"synthetic": f"study-{index}"}, screening="include",
                                     screening_reason="Explicitly included by the synthetic demo command",
                                     publication_checks=[PublicationCheck(source="synthetic", detail="Not a real publication")]))
        path = inputs / f"study-{index}.txt"
        io_path(path).write_text(text, encoding="utf-8")
        store.add_document(parse_document(path, work.id))
