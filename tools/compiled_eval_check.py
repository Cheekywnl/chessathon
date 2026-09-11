"""Cross-check the compiled hybrid against the original evaluation entry point."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import chess

import agent
import chess_search as cs
import chess_search_compiled as compiled


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heldout", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    agent._stop_pondering()
    blends = (0, 25, 75, 100)
    original = cs.HALFKP_BLEND
    search = cs.Search(cs.TranspositionTable(8), {})
    comparisons = 0
    with args.heldout.open(newline="") as stream:
        for index, row in enumerate(csv.DictReader(stream)):
            if index >= 10000:
                break
            board = chess.Board(row["fen"])
            state = cs.state_from_board(board)
            mobility = cs.legal_moves(state)[3]
            for blend in blends:
                cs.HALFKP_BLEND = blend
                search._ctx.blend = blend
                expected = search.evaluate(state, mobility)
                actual = compiled.evaluate(state, mobility, search._ctx)
                assert expected == actual, (index, blend, board.fen(), expected, actual)
                comparisons += 1
    cs.HALFKP_BLEND = original
    result = {"positions": comparisons // len(blends), "blends": blends,
              "comparisons": comparisons, "mismatches": 0,
              "playing_strength_validated": False}
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
