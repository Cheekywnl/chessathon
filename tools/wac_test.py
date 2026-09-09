"""Runs the engine against Win At Chess (Reinfeld, 1958; 300 tactical positions, EPD format
with each position's known best move(s)) directly through chess_search.Search, in-process --
same rationale as fast_arena.py: no subprocess, one JIT warm-up, fast iteration.

This is a correctness/tactics smoke test, not a strength measurement: it asks "does the search
still find known tactics at a short, fixed time budget", the same question `make gate` asks with
its two clean games, just far more targeted. It does not replace fast_arena.py's A/B testing for
"is this change actually stronger" -- WAC passing is necessary, not sufficient, for that.

Usage:
    uv run python -m tools.wac_test                       # full 300, 1s/position
    uv run python -m tools.wac_test --count 20 --time-ms 500   # quick smoke
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import chess

import chess_eval as ce
import chess_movegen as mg
import chess_search as cs
import chess_state as cst

WAC_PATH = Path(__file__).resolve().parent / "wac.epd"
TT_SIZE_POWER = 18
MAX_SEARCH_DEPTH = 64
EPD_BM_RE = re.compile(r'^(?P<fen_fields>.+?)\s+bm\s+(?P<moves>.+?);.*?id\s+"(?P<id>[^"]+)"')


def load_positions(path: Path) -> list[tuple[str, str, list[str]]]:
    positions = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        match = EPD_BM_RE.match(line)
        if match is None:
            raise ValueError(f"unparsable EPD line: {line!r}")
        fen = match["fen_fields"] + " 0 1"
        moves = match["moves"].strip().split()
        positions.append((match["id"], fen, moves))
    return positions


def best_move_at(fen: str, time_ms: float) -> chess.Move:
    board = chess.Board(fen)
    tt = cs.TranspositionTable(TT_SIZE_POWER)
    search = cs.Search(tt, {})
    deadline = time.monotonic() + time_ms / 1000.0
    best_move = next(iter(board.legal_moves))
    depth, completed, last_score = 1, 0, 0
    while depth <= MAX_SEARCH_DEPTH:
        prev_score = last_score if completed > 0 else None
        try:
            move, score, _ = search.search_root(board, depth, deadline, prev_score=prev_score)
        except cs.TimeUp:
            break
        best_move, completed, last_score = move, depth, score
        if abs(score) >= cs.MATE_THRESHOLD or time.monotonic() >= deadline:
            break
        depth += 1
    return best_move


def run(count: int, time_ms: float) -> None:
    ce.warm_up()
    mg.warm_up()
    cst.warm_up()

    positions = load_positions(WAC_PATH)[:count]
    passed, failed = 0, []
    for position_id, fen, san_moves in positions:
        board = chess.Board(fen)
        accepted = {board.parse_san(san) for san in san_moves}
        move = best_move_at(fen, time_ms)
        if move in accepted:
            passed += 1
        else:
            failed.append((position_id, fen, san_moves, board.san(move)))
        print(f"{position_id}: {'pass' if move in accepted else 'FAIL'}", flush=True)

    total = len(positions)
    print(f"\n{passed}/{total} ({passed / total:.1%})")
    if failed:
        print("\nfailed:")
        for position_id, fen, wanted, got in failed:
            print(f"  {position_id}: wanted {'/'.join(wanted)}, got {got}  ({fen})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the engine against Win At Chess.")
    parser.add_argument("--count", type=int, default=300, help="Positions to test, from WAC.001.")
    parser.add_argument("--time-ms", type=float, default=1000, help="Search budget per position.")
    arguments = parser.parse_args()
    run(arguments.count, arguments.time_ms)


if __name__ == "__main__":
    main()
