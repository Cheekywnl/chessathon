"""Generate a reproducible, paired-test opening pool from the existing permitted book.

Choose lines before observing match outcomes, and reject grossly unbalanced positions
using only the preserved classical baseline. This is a local test pool, not the
competition's unpublished curated openings.
"""

from __future__ import annotations

import argparse
import importlib
import json
import random
import sys
import time
from pathlib import Path

import chess
import chess.polyglot

from tools.halfkp_data import file_hash
from tools.version_arena import OPENING_LINES


def book_exit_key(reader: chess.polyglot.MemoryMappedReader, position: chess.Board) -> str:
    board = position.copy()
    while board.fullmove_number <= 20 and not board.is_game_over(claim_draw=True):
        entry = max(reader.find_all(board), key=lambda item: item.weight, default=None)
        if entry is None:
            break
        board.push(entry.move)
    return " ".join(board.fen().split()[:4])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--cpu", type=int, default=6)
    parser.add_argument("--distinct-book-exits", action="store_true")
    parser.add_argument("--exclude-file", action="append", type=Path, default=[])
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    import psutil  # type: ignore[import-untyped]

    psutil.Process().cpu_affinity([args.cpu])
    baseline = args.baseline.resolve()
    sys.path.insert(0, str(baseline))
    cs = importlib.import_module("chess_search")
    assert cs.__file__ is not None
    assert Path(cs.__file__).parent == baseline
    rng = random.Random(args.seed)
    book = baseline / "book/codekiddy.bin"
    families = list(OPENING_LINES)
    seen: set[str] = set()
    seen_exits: set[str] = set()
    positions: list[dict[str, object]] = []
    attempts = 0
    with chess.polyglot.open_reader(book) as reader:
        if args.distinct_book_exits:
            for line in OPENING_LINES.values():
                prior = chess.Board()
                for san in line:
                    prior.push_san(san)
                seen_exits.add(book_exit_key(reader, prior))
        for path in args.exclude_file:
            for entry in json.loads(path.read_text(encoding="utf8"))["positions"]:
                prior = chess.Board(entry["fen"])
                seen.add(" ".join(prior.fen().split()[:4]))
                seen_exits.add(book_exit_key(reader, prior))
        while len(positions) < args.count:
            attempts += 1
            if attempts > args.count * 1000:
                raise RuntimeError("could not generate enough distinct balanced book positions")
            family = families[len(positions) % len(families)]
            board = chess.Board()
            for san in OPENING_LINES[family]:
                board.push_san(san)
            target_ply = rng.randrange(16, 29)
            while board.ply() < target_ply:
                entries = list(reader.find_all(board))
                if not entries:
                    break
                entry = rng.choices(entries, weights=[e.weight ** 0.5 for e in entries])[0]
                board.push(entry.move)
            key = " ".join(board.fen().split()[:4])
            if board.ply() < 14 or key in seen or board.is_game_over(claim_draw=True):
                continue
            if not board.is_valid() or len(board.piece_map()) < 26:
                continue
            exit_key = book_exit_key(reader, board)
            if args.distinct_book_exits and exit_key in seen_exits:
                continue
            search = cs.Search(cs.TranspositionTable(16), {})
            _move, score, _pv = search.search_root(board, 4, time.monotonic() + 60)
            if abs(score) > 125:
                continue
            seen.add(key)
            seen_exits.add(exit_key)
            prefix = f"{args.seed}_" if args.distinct_book_exits else ""
            positions.append({"name": f"{prefix}{len(positions):03d}_{family}", "family": family,
                              "fen": board.fen(), "baseline_depth4_stm_cp": score,
                              "book_exit_key": exit_key,
                              "uci_line": [move.uci() for move in board.move_stack]})
    result = {"seed": args.seed, "count": len(positions), "attempts": attempts,
              "book_sha256": file_hash(book), "baseline": str(baseline),
              "selection": "sqrt-weight book sampling; baseline depth4 abs(score)<=125cp",
              "distinct_book_exits": args.distinct_book_exits,
              "excluded_pools_sha256": {str(path): file_hash(path) for path in args.exclude_file},
              "positions": positions}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps({key: value for key, value in result.items() if key != "positions"}))
    print(f"opening suite SHA256 {file_hash(args.out)}")


if __name__ == "__main__":
    main()
