"""Exercise actual subprocess suspension/affinity and reject void games before scoring."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import chess

from harness.referee import Outcome
from tools import sprt_arena, version_arena
from tools.platform_agent import SuspendedAgent, local

BUSY_AGENT = """
import threading
from pathlib import Path
ticks=Path(__file__).parent/'ticks.txt'
ticks.write_text('0')
def spin():
    count=0
    while True:
        count+=1
        if count%10000==0:
            ticks.write_text(str(count))
threading.Thread(target=spin, daemon=True).start()
def get_move(fen, time_left_ms):
    return 'e2e4'
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-python", required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="chess-platform-check-") as temp:
        root = Path(temp)
        (root / "agent.py").write_text(BUSY_AGENT, encoding="utf8")
        agent = local(root, cpu=2, python=args.engine_python)
        assert isinstance(agent, SuspendedAgent)
        try:
            agent.start(10)
            first_ticks = (root / "ticks.txt").read_text()
            before = agent.control.cpu_times().user
            time.sleep(0.25)
            frozen = agent.control.cpu_times().user - before
            assert frozen <= 0.016, frozen
            assert (root / "ticks.txt").read_text() == first_ticks
            agent.control.resume()
            before = agent.control.cpu_times().user
            time.sleep(1.0)
            running = agent.control.cpu_times().user - before
            agent.control.suspend()
            current_ticks = (root / "ticks.txt").read_text()
            assert current_ticks and int(current_ticks) > int(first_ticks or 0), current_ticks
            assert agent.move(chess.STARTING_FEN, 1000) == "e2e4"
            before = agent.control.cpu_times().user
            final_ticks = (root / "ticks.txt").read_text()
            time.sleep(0.25)
            after_move = agent.control.cpu_times().user - before
            assert after_move <= 0.016, after_move
            assert (root / "ticks.txt").read_text() == final_ticks
            affinity = agent.control.cpu_affinity()
            assert affinity == [2]
        finally:
            agent.stop()
    for module in (version_arena, sprt_arena):
        argv = ["arena", "--opponent", "."]
        if module is version_arena:
            argv += ["--openings", "startpos"]
        with patch.object(sys, "argv", argv), patch.object(
            module, "play_match", return_value=Outcome("void", "both_failed", "*")
        ):
            try:
                module.main()
            except SystemExit as exc:
                assert "Invalid" in str(exc)
            else:
                raise AssertionError("void game was scored as a draw")
    print(json.dumps({"affinity": affinity, "suspended_cpu_seconds": frozen,
                      "running_cpu_seconds": running, "post_move_cpu_seconds": after_move,
                      "counter_advanced_only_while_resumed": True,
                      "void_rejected_by_version_arena_and_sprt": True}, indent=2))


if __name__ == "__main__":
    main()
