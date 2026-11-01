"""Measured progress, explicitly approximate ETA, with no training changes."""

import time
from pathlib import Path

from .recovery import write_json


class Progress:
    def __init__(self, output, phase, total):
        self.path = Path(output) / "progress.json"
        self.phase, self.total = phase, total
        self.started = self.last = time.monotonic()
        self.computing = 0.0
        self.was_waiting = False

    def update(self, completed, status="running"):
        moment = time.monotonic()
        if not self.was_waiting:
            self.computing += moment - self.last
        self.last, self.was_waiting = moment, status == "waiting_gpu"
        eta = self.computing / completed * (self.total - completed) if completed else None
        if status in {"waiting_gpu", "failed", "interrupted"}:
            eta = None
        write_json(
            self.path,
            dict(
                phase=self.phase,
                completed=completed,
                total=self.total,
                elapsed_seconds=moment - self.started,
                eta_seconds=eta,
                status=status,
                sampled_at=time.time(),
                note="Estimate from completed work; excludes future GPU waits.",
            ),
        )
