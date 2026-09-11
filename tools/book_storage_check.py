"""Prove gzip book restoration and choices match the original Polyglot asset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import chess
import chess.polyglot

import agent
from tools.halfkp_data import file_hash


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", required=True, type=Path)
    parser.add_argument("--openings", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    assert agent._BOOK is not None and agent._BOOK_DIRECTORY is not None
    restored = Path(agent._BOOK_DIRECTORY.name) / "codekiddy.bin"
    assert file_hash(restored) == file_hash(args.original)
    document = json.loads(args.openings.read_text())
    boards = [chess.Board(item["fen"]) for item in document["positions"]]
    boards.append(chess.Board())
    checked = 0
    hits = 0
    with chess.polyglot.open_reader(args.original) as original:
        assert len(original) == len(agent._BOOK)
        for board in boards:
            for fullmove in (board.fullmove_number, 20, 21):
                board.fullmove_number = fullmove
                best = max(original.find_all(board), key=lambda entry: entry.weight, default=None)
                expected = best.move if best is not None and fullmove <= 20 else None
                actual = agent._book_move(board)
                assert actual == expected, board.fen()
                hits += int(actual is not None)
                checked += 1
        entry_count = len(original)
    result = {"identical_restored_sha256": file_hash(restored),
              "restored_bytes": restored.stat().st_size, "entries": entry_count,
              "choice_comparisons": checked, "book_hits": hits,
              "move_20_21_boundary": True, "all_bytes_identical": True}
    directory = restored.parent
    agent._close_book()
    assert not directory.exists()
    result["temporary_directory_cleanup"] = True
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
