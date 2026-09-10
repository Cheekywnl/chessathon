"""Cross-check fused transitions and unchanged fixed-depth search with real history."""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import chess
import chess.pgn
import numpy as np

import agent  # noqa: F401  # Warm all runtime JIT signatures before measurements.
import chess_draw as cd
import chess_search as cs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-search", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cpu", type=int, default=4)
    args = parser.parse_args()
    import psutil  # type: ignore[import-untyped]

    psutil.Process().cpu_affinity([args.cpu])
    positions = [chess.Board(fen) for fen in (
        chess.STARTING_FEN,
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
        "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
        "1r2k3/P7/8/8/8/8/7p/4K3 w - - 0 1",
        "4k3/8/8/8/8/8/p6P/1R2K3 b - - 0 1",
        "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
        "4k3/8/8/8/3pP3/8/8/4K3 b - e3 0 1",
        "4r1k1/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
    )]
    rng = random.Random(20260911)
    board = chess.Board()
    chain = cs.state_from_board(board)
    chain_key = cs.hash_of(chain)
    chained = 0
    for _ in range(2400):
        if board.is_game_over():
            board = chess.Board()
            chain = cs.state_from_board(board)
            chain_key = cs.hash_of(chain)
        move = rng.choice(list(board.legal_moves))
        chain, chain_key, check = cs.apply_move_info(
            chain, chain_key, move.from_square, move.to_square, move.promotion or 0,
        )
        assert chain_key == cs.hash_of(chain) and check == cs.is_in_check(chain)
        board.push(move)
        assert check == board.is_check()
        positions.append(board.copy(stack=False))
        chained += 1
    transitions: Counter[str] = Counter()
    for board in positions:
        state = cs.state_from_board(board)
        key = cs.hash_of(state)
        for move in board.legal_moves:
            child, child_key, check = cs.apply_move_info(
                state, key, move.from_square, move.to_square, move.promotion or 0,
            )
            reference = cs.apply_move(state, move.from_square, move.to_square, move.promotion or 0)
            assert child == reference and child_key == cs.hash_of(reference)
            assert check == cs.is_in_check(reference)
            assert all(isinstance(child[i], np.uint64) for i in (*range(8), 9))
            after = board.copy(stack=False)
            after.push(move)
            expected = cs.state_from_board(after)
            assert child[:10] == expected[:10] and child[11] == expected[11]
            assert child[10] == (after.ep_square if after.has_pseudo_legal_en_passant() else -1)
            assert check == after.is_check()
            kind = ("castling" if board.is_castling(move) else
                    "en_passant" if board.is_en_passant(move) else
                    "capture_promotion" if move.promotion and board.is_capture(move) else
                    "promotion" if move.promotion else
                    "capture" if board.is_capture(move) else "quiet")
            transitions[kind] += 1
    assert all(transitions[kind] for kind in (
        "castling", "en_passant", "capture_promotion", "promotion", "capture", "quiet",
    ))

    spec = importlib.util.spec_from_file_location("search_before_fusion", args.reference_search)
    assert spec is not None and spec.loader is not None
    before: Any = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(before)
    probes = json.loads(Path("docs/validation/draw-choice-probe.json").read_text())["records"]
    source = json.loads(Path("docs/validation/draw-analysis.json").read_text())["draws"]
    comparisons = []
    for index, probe in enumerate(probes):
        row = next(row for row in source if row["pair_index"] == probe["pair"]
                   and row["candidate_white"] == probe["white"])
        game = chess.pgn.read_game(io.StringIO(row["pgn"]))
        assert game is not None
        board = game.board()
        history = Counter({cs.hash_of_board(board): 1})
        for move in list(game.mainline_moves())[:probe["ply"]]:
            board.push(move)
            history[cs.hash_of_board(board)] += 1
        assert board.fen() == probe["fen"]
        claims, complete = cd.root_claims(
            board, dict(history), list(board.legal_moves), time.monotonic() + 60,
        )
        assert complete
        repeated = cd.repeated_moves(board, dict(history), list(board.legal_moves))
        results: dict[str, Any] = {}
        modules = (("before", before), ("fused", cs))
        for label, module in modules if index % 2 == 0 else modules[::-1]:
            search = module.Search(module.TranspositionTable(16), dict(history))
            search.root_draw_claims = dict(claims)
            search.root_repeated_moves = set(repeated)
            started = time.perf_counter()
            choice, score, ranked = search.search_root(board, 6, time.monotonic() + 120)
            elapsed = time.perf_counter() - started
            results[label] = {"move": choice.uci(), "score": score, "nodes": search.nodes,
                              "ranked": ranked, "seconds": elapsed, "seen": search.seen,
                              "killers": search.killers, "history": search.history,
                              "tt": search.tt.table}
        for field in ("move", "score", "nodes", "ranked", "seen", "killers", "history", "tt"):
            assert results["before"][field] == results["fused"][field], (index, field)
        comparisons.append({"pair": probe["pair"], "white": probe["white"],
                            "move": results["fused"]["move"], "score": results["fused"]["score"],
                            "nodes": results["fused"]["nodes"],
                            "before_seconds": results["before"]["seconds"],
                            "fused_seconds": results["fused"]["seconds"]})
        print(f"history case {index+1}/{len(probes)}: exact", flush=True)
    old_seconds = sum(row["before_seconds"] for row in comparisons)
    new_seconds = sum(row["fused_seconds"] for row in comparisons)
    result = {"legal_transitions": sum(transitions.values()), "move_types": dict(transitions),
              "chained_hash_checks": chained, "history_search_comparisons": len(comparisons),
              "depth": 6, "cpu": args.cpu, "before_seconds": old_seconds,
              "fused_seconds": new_seconds, "speed_ratio": old_seconds / new_seconds,
              "comparisons": comparisons, "playing_strength_validated": False}
    args.out.write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps({k: v for k, v in result.items() if k != "comparisons"}, indent=2))


if __name__ == "__main__":
    main()
