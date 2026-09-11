"""Reproduce a pruned draw-saving capture in a small endgame."""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path
from typing import Any

import chess

import chess_search as cs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("old_qendgame", args.reference)
    assert spec is not None and spec.loader is not None
    before: Any = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(before)
    # The old Python path supplies an independent reference without loading the
    # candidate compiled backend. This low-material position is classical anyway.
    before.HALFKP_WEIGHTS = None
    base = chess.Board("8/8/3k4/8/8/8/p7/K6b w - - 0 1")
    assert base.is_valid()
    saved = base.copy()
    saved.push_uci("a1a2")
    assert saved.is_insufficient_material()
    records = []
    reproduced = 0
    for board in (base, base.mirror()):
        for alpha in (-1000, -400, -200, -100, -50, -25, 0, 25):
            scores = {}
            for label, module in (("before", before), ("after", cs)):
                search = module.Search(module.TranspositionTable(12), {})
                search.deadline = time.monotonic() + 60
                scores[label] = search.quiescence(module.state_from_board(board),
                                                  alpha, alpha + 1, 0)
                assert not {k: n for k, n in search.seen.items() if n}
            expected = -cs.CONTEMPT
            if alpha < expected:
                assert scores["after"] >= alpha + 1, (board.fen(), alpha, scores)
                reproduced += int(scores["before"] < alpha + 1)
            else:
                assert scores["after"] == expected, (board.fen(), alpha, scores)
            records.append({"fen": board.fen(), "alpha": alpha, "beta": alpha + 1, **scores})
    assert reproduced >= 6
    result = {"checks": len(records), "old_draw_saving_failures_reproduced": reproduced,
              "all_new_bounds_correct": True, "records": records,
              "playing_strength_validated": False}
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
