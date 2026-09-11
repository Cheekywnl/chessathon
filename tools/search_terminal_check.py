"""Check terminal leaf scores against the actual referee's python-chess outcomes."""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
import time
from pathlib import Path

import chess

import agent  # noqa: F401  # Warm the actual runtime before terminal probes.
import chess_search as cs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference-search", type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("terminal_search_before", args.reference_search)
    assert spec is not None and spec.loader is not None
    before = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = before
    spec.loader.exec_module(before)
    fens = [
        "7k/6Q1/5K2/8/8/8/8/8 b - - 100 90",  # mate before counter
        "7k/5K2/6Q1/8/8/8/8/8 b - - 0 90",  # stalemate at qsearch leaf
        "6k1/8/8/8/8/8/3q4/6K1 w - - 99 50",  # intended fifty-move claim
        "8/8/8/8/5k2/8/6q1/7K w - - 99 50",  # only legal move is a capture
    ]
    boards = [chess.Board(fen) for fen in fens]
    rng = random.Random(20260911)
    board = chess.Board()
    for _ in range(120):
        if board.is_game_over():
            board = chess.Board()
        board.push(rng.choice(list(board.legal_moves)))
        boards.append(board.copy(stack=False))
    checked = 0
    terminal = 0
    for original in boards:
        for clock in (0, 98, 99, 100):
            board = original.copy(stack=False)
            board.halfmove_clock = clock
            assert board.is_valid()
            outcome = board.outcome(claim_draw=True)
            state = cs.state_from_board(board)
            key = cs.hash_of(state)
            for ply in (0, 1):
                search = cs.Search(cs.TranspositionTable(12), {key: 1})
                search.deadline = time.monotonic() + 60
                if outcome is None:
                    assert not search._is_draw(state, key), board.fen()
                else:
                    expected = (-(cs.MATE - ply) if board.is_checkmate()
                                else search._draw_score(ply))
                    assert search.quiescence(state, -cs.MATE, cs.MATE, ply) == expected
                    for depth in (0, 1, 3):
                        assert search.negamax(state, depth, -cs.MATE, cs.MATE, ply) == expected
                    terminal += 1
                checked += 1
    reproduced = []
    for fen in fens[:3]:
        board = chess.Board(fen)
        state = before.state_from_board(board)
        search = before.Search(before.TranspositionTable(12), {})
        search.deadline = time.monotonic() + 60
        score = search.quiescence(state, -cs.MATE, cs.MATE, 1)
        expected = -(cs.MATE - 1) if board.is_checkmate() else cs.CONTEMPT
        assert score != expected, (fen, score)
        reproduced.append({"fen": fen, "before_score": score, "correct_score": expected})
    result = {"referee_state_checks": checked, "terminal_leaf_checks": terminal,
              "old_failures_reproduced": reproduced, "passed": True}
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
