"""Prove extracted mop-up evaluation and fixed-depth search equal the rollback build."""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import subprocess
import sys
from pathlib import Path

import chess

import chess_eval as candidate
from tools.nnue_halfkp_bench import _random_positions
from tools.version_arena import OPENING_LINES, _fen_for

SEARCH_SCRIPT = """
import json, sys, time
sys.path.insert(0, sys.argv[1])
import chess, chess_eval as ce, chess_movegen as mg, chess_state as cst, chess_search as cs
ce.warm_up(); mg.warm_up(); cst.warm_up()
results=[]
for fen in json.loads(sys.argv[2]):
    search=cs.Search(cs.TranspositionTable(16), {})
    move, score, ranked=search.search_root(chess.Board(fen), 5, time.monotonic()+600)
    results.append([move.uci(), score, search.nodes])
print(json.dumps(results))
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("rollback_eval", args.baseline / "chess_eval.py")
    assert spec is not None and spec.loader is not None
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    baseline.warm_up()
    candidate.warm_up()
    boards = _random_positions(1000, seed=715)
    rng = random.Random(778)
    for _ in range(3000):
        board = chess.Board(None)
        squares = rng.sample(range(64), 8)
        board.set_piece_at(squares[0], chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(squares[1], chess.Piece(chess.KING, chess.BLACK))
        color = rng.choice([chess.WHITE, chess.BLACK])
        pieces = rng.choice([[chess.ROOK], [chess.QUEEN], [chess.BISHOP, chess.KNIGHT],
                             [chess.PAWN], [chess.ROOK, chess.QUEEN]])
        for square, piece in zip(squares[2:], pieces, strict=False):
            board.set_piece_at(square, chess.Piece(piece, color))
        board.turn = rng.choice([chess.WHITE, chess.BLACK])
        boards.append(board)
    for board in boards:
        for mobility in (0, 20, 37):
            expected = baseline.evaluate_board(board, mobility)
            assert candidate.evaluate_board(board, mobility) == expected
    positions = [_fen_for(line) for line in OPENING_LINES.values()]
    results = []
    for root in (args.baseline.resolve(), Path.cwd()):
        process = subprocess.run(
            [sys.executable, "-c", SEARCH_SCRIPT, str(root), json.dumps(positions)],
            capture_output=True, text=True, check=True,
        )
        results.append(json.loads(process.stdout.strip().splitlines()[-1]))
    assert results[0] == results[1], results
    result = {"evaluation_comparisons": len(boards) * 3, "evaluation_mismatches": 0,
              "search_positions": len(positions), "depth": 5,
              "move_score_node_mismatches": 0, "search_results": results[1]}
    args.out.write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
