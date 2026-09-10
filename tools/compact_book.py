"""Keep the first maximum-weight record per Polyglot key, matching deterministic selection.

This never edits its source book. Actual position probes are checked separately,
because python-chess also filters illegal moves after matching a hash key.
"""

from __future__ import annotations

import argparse
import csv
import json
import struct
from pathlib import Path

import chess
import chess.polyglot
import numpy as np

from tools.halfkp_data import file_hash
from tools.version_arena import OPENING_LINES, _fen_for

ENTRY = struct.Struct(">QHHI")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--verify-data", type=Path, required=True)
    parser.add_argument("--positions", type=int, default=100000)
    parser.add_argument("--max-bytes", type=int,
                        help="Optionally drop the lowest-weight keys to meet a tested asset budget")
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("refusing to overwrite compact book")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    kept, input_rows, key_count = 0, 0, 0
    best: bytes | None = None
    prior_key = -1
    best_weight = 0
    with args.input.open("rb") as source, args.out.open("wb") as output:
        while raw := source.read(ENTRY.size):
            if len(raw) != ENTRY.size:
                raise ValueError("partial Polyglot entry")
            key, _move, weight, _learn = ENTRY.unpack(raw)
            input_rows += 1
            if key < prior_key:
                raise ValueError("unsorted Polyglot source")
            if key != prior_key:
                if best is not None:
                    output.write(best)
                    kept += 1
                key_count += 1
                best, best_weight = None, 0
                prior_key = key
            if weight > best_weight:
                best, best_weight = raw, weight
        if best is not None:
            output.write(best)
            kept += 1
    dropped_keys = 0
    if args.max_bytes is not None:
        if args.max_bytes < 16:
            raise ValueError("book byte budget must hold at least one entry")
        table = np.fromfile(args.out, dtype=[("key", ">u8"), ("move", ">u2"),
                                             ("weight", ">u2"), ("learn", ">u4")])
        maximum = args.max_bytes // 16
        if maximum < len(table):
            priority = np.lexsort((table["key"], -table["weight"].astype(np.int32)))
            selected = np.sort(priority[:maximum])
            table[selected].tofile(args.out)
            dropped_keys = len(table) - maximum
            kept = maximum
    matches = hits = lost_hits = 0
    with chess.polyglot.open_reader(args.input) as original, \
            chess.polyglot.open_reader(args.out) as compact:
        # Check every stored key's selected raw move/weight, including tie ordering.
        for entry in compact:
            expected = max(original.find_all(entry.key), key=lambda item: item.weight)
            assert entry == expected

        def check(fen: str) -> None:
            nonlocal matches, hits, lost_hits
            board = chess.Board(fen)
            left = max(original.find_all(board), key=lambda item: item.weight, default=None)
            right = max(compact.find_all(board), key=lambda item: item.weight, default=None)
            if right is None and left is not None and args.max_bytes is not None:
                lost_hits += 1
            else:
                assert left == right, fen
            matches += 1
            hits += int(left is not None)

        for moves in OPENING_LINES.values():
            check(_fen_for(moves))
        with args.verify_data.open(newline="", encoding="utf8") as source:
            reader = csv.reader(source)
            next(reader)
            for _, row in zip(range(args.positions), reader, strict=False):
                check(row[0])
    result = {"source": str(args.input.resolve()), "source_sha256": file_hash(args.input),
              "input_entries": input_rows, "input_keys": key_count, "output_entries": kept,
              "source_bytes": args.input.stat().st_size, "output_bytes": args.out.stat().st_size,
              "output_sha256": file_hash(args.out), "all_key_maxima_verified": True,
              "position_probes": matches, "covered_probes": hits, "probe_mismatches": 0,
              "dropped_keys": dropped_keys, "dropped_coverage_probes": lost_hits,
              "real_match_validation": "pending"}
    args.out.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
