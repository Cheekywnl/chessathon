"""Exact old/new kernel, integer-reference and fixed-depth search comparisons."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import chess
import numpy as np

import chess_halfkp_int as qi
import chess_search as cs
from tools.halfkp_data import RECORD, encode_row
from tools.halfkp_quant_check import board_args, integer_reference, weight_args


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--heldout", required=True, type=Path)
    parser.add_argument("--openings", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--cpu", type=int, default=14)
    parser.add_argument("--depth", type=int, default=6)
    args = parser.parse_args()
    import psutil  # type: ignore[import-untyped]

    psutil.Process().cpu_affinity([args.cpu])
    import agent

    agent._stop_pondering()  # Its import warms the complete search, not just the NN kernel.
    spec = importlib.util.spec_from_file_location("halfkp_kernel_reference", args.reference)
    assert spec is not None and spec.loader is not None
    reference = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = reference
    spec.loader.exec_module(reference)
    old = reference.load_weights(args.weights)
    new = qi.load_weights(args.weights)
    reference.warm_up(old)
    qi.warm_up(new)
    old_args, new_args = weight_args(old), weight_args(new)
    row = np.zeros(1, dtype=RECORD)
    checked = 0
    with args.heldout.open() as stream:
        for entry in csv.DictReader(stream):
            if checked >= 10000:
                break
            encode_row(entry["fen"], 0, True, row[0])
            common = (row[0]["white"], row[0]["black"], int(row[0]["count"]),
                      bool(row[0]["stm"]))
            expected = reference.forward(*common, *old_args)
            actual = qi.forward(*common, *new_args)
            assert expected == actual == integer_reference(row[0], new)
            ba = board_args(chess.Board(entry["fen"]))
            assert reference.evaluate(*ba, *old_args) == qi.evaluate(*ba, *new_args)
            checked += 1

    boards = [chess.Board(item["fen"]) for item in
              json.loads(args.openings.read_text())["positions"][:10]]
    # Compilation/warm-up is outside timing. Preserve the search's move/score/node
    # behavior while alternating comparison order to reduce clock/order bias.
    original_kernel, original_weights = cs.__dict__["halfkp"], cs.HALFKP_WEIGHTS
    totals = {"before": 0.0, "after": 0.0}
    comparisons = []
    try:
        for index, board in enumerate(boards):
            results: dict[str, Any] = {}
            versions = [("before", reference, old), ("after", qi, new)]
            if index % 2:
                versions.reverse()
            for name, kernel, weights in versions:
                cs.__dict__["halfkp"] = kernel
                cs.HALFKP_WEIGHTS = weights
                search = cs.Search(cs.TranspositionTable(16), {})
                started = time.perf_counter()
                previous = None
                for depth in range(1, args.depth + 1):
                    move, score, _ = search.search_root(board, depth, time.monotonic() + 120,
                                                        prev_score=previous)
                    previous = score
                elapsed = time.perf_counter() - started
                results[name] = {"move": move.uci(), "score": score, "nodes": search.nodes,
                                 "seconds": elapsed}
                totals[name] += elapsed
            for key in ("move", "score", "nodes"):
                assert results["before"][key] == results["after"][key], (board.fen(), results)
            comparisons.append({"fen": board.fen(), **results})
    finally:
        cs.__dict__["halfkp"] = original_kernel
        cs.HALFKP_WEIGHTS = original_weights
    result = {"integer_and_bitboard_comparisons": checked, "mismatches": 0,
              "search_comparisons": len(comparisons), "depth": args.depth, "cpu": args.cpu,
              "search_seconds": totals, "speed_ratio": totals["before"] / totals["after"],
              "comparisons": comparisons, "playing_strength_validated": False}
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
