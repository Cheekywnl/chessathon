"""Check hybrid selection, preserved endgames, book boundary and one-core search throughput."""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import chess

import agent
import chess_halfkp_int as qi
import chess_search as cs
from tools.nnue_halfkp_bench import _random_positions
from tools.version_arena import OPENING_LINES, _fen_for


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--cpu", type=int, default=4)
    parser.add_argument("--compare-search", type=Path,
                        help="Prior search module for exact move/score/node equivalence")
    args = parser.parse_args()
    import psutil  # type: ignore[import-untyped]

    psutil.Process().cpu_affinity([args.cpu])
    weights = qi.load_weights(args.weights)
    qi.warm_up(weights)
    search = cs.Search(cs.TranspositionTable(16), {})
    endgames = [
        "1K1k4/1P6/8/8/8/8/r7/2R5 w - - 0 1",
        "1K1k4/1P6/8/8/8/8/r7/2R5 b - - 0 1",
        "4k3/8/8/8/8/8/8/2BNK3 w - - 0 1",
        "4k3/8/8/8/8/8/8/3QK3 b - - 0 1",
    ]
    for fen in endgames:
        state = cs.state_from_board(chess.Board(fen))
        cs.HALFKP_WEIGHTS = None
        expected = search.evaluate(state, 20)
        cs.HALFKP_WEIGHTS = weights
        for blend in (0, 50, 75, 100):
            cs.HALFKP_BLEND = blend
            assert search.evaluate(state, 20) == expected
    boards = _random_positions(100, seed=25)
    for board in boards:
        state = cs.state_from_board(board)
        cs.HALFKP_WEIGHTS = None
        expected = search.evaluate(state, 20)
        cs.HALFKP_WEIGHTS = weights
        cs.HALFKP_BLEND = 0
        assert search.evaluate(state, 20) == expected
    book_entry = SimpleNamespace(move=chess.Move.from_uci("e2e4"), weight=10)
    book = SimpleNamespace(find_all=lambda board: iter([book_entry]))
    board = chess.Board()
    with patch.object(agent, "_BOOK", book):
        board.fullmove_number = 20
        assert agent._book_move(board) == book_entry.move
        board.fullmove_number = 21
        assert agent._book_move(board) is None
    results = []
    before: Any = None
    if args.compare_search:
        spec = importlib.util.spec_from_file_location("search_before_skip", args.compare_search)
        assert spec is not None and spec.loader is not None
        before = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(before)
    positions = [_fen_for(moves) for moves in OPENING_LINES.values()]
    for blend in (0, 75, 100):
        cs.HALFKP_WEIGHTS = weights
        cs.HALFKP_BLEND = blend
        elapsed = 0.0
        nodes = 0
        for fen in positions:
            search = cs.Search(cs.TranspositionTable(16), {})
            start = time.perf_counter()
            move, score, _ = search.search_root(
                chess.Board(fen), args.depth, time.monotonic() + 600,
            )
            elapsed += time.perf_counter() - start
            nodes += search.nodes
            if before is not None:
                before.HALFKP_WEIGHTS = weights
                before.HALFKP_BLEND = blend
                reference = before.Search(before.TranspositionTable(16), {})
                expected, expected_score, _ = reference.search_root(
                    chess.Board(fen), args.depth, time.monotonic() + 600,
                )
                assert (move, score, search.nodes) == (expected, expected_score, reference.nodes)
        results.append({"blend": blend, "positions": len(positions), "nodes": nodes,
                        "seconds": elapsed, "nodes_per_second": nodes / elapsed})
    output = {"endgame_fallback_comparisons": len(endgames) * 4,
              "zero_blend_matches_classical": len(boards), "book_move_20_21_boundary": True,
              "cpu_affinity": [args.cpu], "depth": args.depth, "search": results,
              "prior_move_score_node_comparisons": 30 if before is not None else 0,
              "weights": str(args.weights), "playing_strength_validated": False}
    args.out.write_text(json.dumps(output, indent=2), encoding="utf8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
