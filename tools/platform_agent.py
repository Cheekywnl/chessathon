"""Local match controller for the live contract's one-core, suspended-opponent behavior.

This is test infrastructure outside harness/. psutil is a controller dependency;
agent subprocesses can use the unchanged competition-pinned CPU interpreter.
The existing runner, wire protocol, clocks and referee remain unchanged.
"""

from __future__ import annotations

import os
import sys
from contextlib import suppress
from pathlib import Path
from typing import Any

from harness.sandbox import RUNNER, Agent


class SuspendedAgent(Agent):
    def __init__(self, command: list[str], cpu: int) -> None:
        super().__init__(command)
        self.cpu = cpu
        self.control: Any = None
        self.controls: list[Any] = []

    def start(self, init_budget_s: float) -> None:
        import psutil  # type: ignore[import-untyped]

        super().start(init_budget_s)
        assert self._process is not None
        root = psutil.Process(self._process.pid)
        # A Windows venv python.exe can be a launcher with a child interpreter.
        # Suspending only Popen.pid leaves that interpreter running unrestricted.
        self.controls = [root, *root.children(recursive=True)]
        self.control = self.controls[-1]
        for process in self.controls:
            if process.cpu_affinity() != [self.cpu]:
                raise RuntimeError("agent did not inherit the controller's one-core affinity")
        for process in reversed(self.controls):
            process.suspend()

    def move(self, fen: str, time_left_ms: int) -> str:
        if self.control is None:
            raise RuntimeError("agent not started")
        for process in self.controls:
            process.resume()
        try:
            return super().move(fen, time_left_ms)
        finally:
            if self._process is not None and self._process.poll() is None:
                for process in reversed(self.controls):
                    if process.is_running():
                        process.suspend()

    def stop(self) -> None:
        import psutil

        for process in reversed(self.controls[1:]):
            with suppress(psutil.NoSuchProcess):
                process.kill()
        super().stop()
        self.controls = []


def local(directory: Path, cpu: int = 2, python: str | None = None) -> Agent:
    import psutil

    # Children inherit affinity, including during import/Numba compilation.
    psutil.Process().cpu_affinity([cpu])
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    return SuspendedAgent([python or sys.executable, str(RUNNER), str(directory.resolve())], cpu)
