"""Check aspiration windows and persistent tables across successive searches."""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path
from typing import Any

import chess

import agent
import chess_search as cs
from tools.compiled_search_check import snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--openings", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    agent._stop_pondering()
    spec = importlib.util.spec_from_file_location("reuse_reference", args.reference)
    assert spec is not None and spec.loader is not None
    reference: Any = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    reference.HALFKP_WEIGHTS = cs.HALFKP_WEIGHTS
    records = []
    for position in json.loads(args.openings.read_text())["positions"][:4]:
        board = chess.Board(position["fen"])
        history = {cs.hash_of_board(board): 1}
        tables = [reference.TranspositionTable(16), cs.TranspositionTable(16)]
        for turn in range(3):
            searches = [module.Search(table, history) for module, table in
                        zip((reference, cs), tables, strict=True)]
            previous: list[int | None] = [None, None]
            for depth in range(1, 8):
                results = []
                for index, search in enumerate(searches):
                    result = search.search_root(board, depth, time.monotonic() + 120,
                                                prev_score=previous[index])
                    previous[index] = result[1]
                    results.append(result)
                assert results[0] == results[1], (board.fen(), turn, depth, results)
                assert snapshot(searches[0]) == snapshot(searches[1]), (board.fen(), turn, depth)
                records.append({"fen": board.fen(), "turn": turn, "depth": depth,
                                "move": results[1][0].uci(), "score": results[1][1],
                                "nodes": searches[1].nodes})
            board.push(results[1][0])
            history[cs.hash_of_board(board)] = history.get(cs.hash_of_board(board), 0) + 1
        print(f"Opening completed: {position['fen']}", flush=True)
    args.out.write_text(json.dumps({"comparisons": len(records), "records": records,
                                   "playing_strength_validated": False}, indent=2) + "\n")
    print(f"{len(records)} exact iterative / persistent-table comparisons", flush=True)


if __name__ == "__main__":
    main()
