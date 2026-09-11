"""Check hits, forced collisions and separate parameter contexts against fresh evaluation."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

import agent
import chess_eval as ce
import chess_search as cs
import chess_search_compiled as compiled


def main() -> None:
    agent._stop_pondering()
    assert cs.HALFKP_WEIGHTS is not None
    import chess

    with Path("../halfkp-trial/data/full-cache/heldout_fens.csv").open() as stream:
        fens = [row["fen"] for _, row in zip(range(1200), csv.DictReader(stream), strict=False)]
    boards = [chess.Board(fen) for fen in fens]
    checks = 0
    for blend in (0, 75, 100):
        ctx = compiled.create_context(compiled.Table(8), {}, ce.DEFAULT_PARAMS,
                                      cs.HALFKP_WEIGHTS, blend, None)
        fresh = compiled.create_context(compiled.Table(8), {}, ce.DEFAULT_PARAMS,
                                        cs.HALFKP_WEIGHTS, blend, None)
        ctx.eval_mask = 3  # deliberate heavy collisions, plus immediate hits
        for board in [*boards, *reversed(boards)]:
            state = cs.state_from_board(board)
            key = np.uint64(cs.hash_of_board(board))
            _, _, _, count = cs.legal_moves(state)
            expected = compiled.evaluate(state, count, fresh)
            assert compiled.evaluate_cached(state, count, key, ctx) == expected
            assert compiled.evaluate_cached(state, count, key, ctx) == expected
            checks += 2
        assert ctx.eval_hits >= len(boards) * 2 and ctx.eval_misses > 1000
    result = {"static_score_checks": checks, "blends": [0, 75, 100],
              "forced_collision_slots": 4, "mismatches": 0}
    Path("data/runs/eval-cache-check.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
