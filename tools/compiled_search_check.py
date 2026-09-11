"""Compare complete and interrupted searches with this team's preceding source."""
from __future__ import annotations

import argparse
import importlib.util
import io
import json
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any

import chess
import chess.pgn
import numpy as np

import agent
import chess_draw as cd
import chess_search as cs


def snapshot(search: Any) -> dict[str, Any]:
    return {"seen": dict(search.seen), "killers": np.asarray(search.killers).tolist(),
            "history": dict(search.history), "tt": search.tt.table, "nodes": search.nodes}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--openings", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    agent._stop_pondering()
    spec = importlib.util.spec_from_file_location("python_search_reference", args.reference)
    assert spec is not None and spec.loader is not None
    reference: Any = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    reference.HALFKP_WEIGHTS = cs.HALFKP_WEIGHTS
    cs.warm_up()
    boards: list[tuple[chess.Board, dict[int, int]]] = [(chess.Board(row["fen"]), {}) for row in
              json.loads(args.openings.read_text())["positions"][:10]]
    probes = json.loads(Path("docs/validation/draw-choice-probe.json").read_text())["records"]
    draws = json.loads(Path("docs/validation/draw-analysis.json").read_text())["draws"]
    for probe in probes:
        row = next(row for row in draws if row["pair_index"] == probe["pair"]
                   and row["candidate_white"] == probe["white"])
        game = chess.pgn.read_game(io.StringIO(row["pgn"]))
        assert game is not None
        board = game.board()
        counts = Counter({cs.hash_of_board(board): 1})
        for move in list(game.mainline_moves())[:probe["ply"]]:
            board.push(move)
            counts[cs.hash_of_board(board)] += 1
        assert board.fen() == probe["fen"]
        boards.append((board, dict(counts)))
    boards += [(chess.Board(fen), {}) for fen in (
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
        "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
        "1r2k3/P7/8/8/8/8/7p/4K3 w - - 0 1",
        "1K1k4/1P6/8/8/8/8/r7/2R5 w - - 0 1",
        "7k/5K2/6Q1/8/8/8/8/8 b - - 0 90",
        "7k/6Q1/5K2/8/8/8/8/8 b - - 100 90",
        "6k1/8/8/8/8/8/3q4/6K1 w - - 99 50",
    )]
    records = []
    original_blend = cs.HALFKP_BLEND
    for blend in (0, 75, 100):
        cs.HALFKP_BLEND = reference.HALFKP_BLEND = blend
        selected = boards if blend == 75 else boards[:10]
        for index, (board, game_history) in enumerate(selected):
            history = game_history or {cs.hash_of_board(board): 1}
            claims, complete = cd.root_claims(
                board, history, list(board.legal_moves), time.monotonic() + 60,
            )
            assert complete
            repeated = cd.repeated_moves(board, history, list(board.legal_moves))
            results = {}
            order = (("python", reference), ("compiled", cs))[::1 if index % 2 else -1]
            for label, module in order:
                search = module.Search(module.TranspositionTable(16), history)
                search.root_draw_claims = dict(claims)
                search.root_repeated_moves = set(repeated)
                search.deadline = time.monotonic() + 120
                started = time.perf_counter()
                if board.is_game_over():
                    result = search.negamax(module.state_from_board(board), 6,
                                            -cs.MATE, cs.MATE, 0)
                else:
                    result = search.search_root(board, 6, search.deadline)
                elapsed = time.perf_counter() - started
                results[label] = {"result": result, "seconds": elapsed, **snapshot(search)}
            for field in ("result", "seen", "killers", "history", "tt", "nodes"):
                assert results["python"][field] == results["compiled"][field], (blend, index, field)
            record = {"blend": blend, "index": index, "fen": board.fen(),
                      "nodes": results["compiled"]["nodes"],
                      "python_seconds": results["python"]["seconds"],
                      "compiled_seconds": results["compiled"]["seconds"]}
            records.append(record)
            print(json.dumps(record), flush=True)
    cs.HALFKP_BLEND = reference.HALFKP_BLEND = original_blend

    # Exact boundary/normalization and colliding-slot replacement comparisons.
    table_before = reference.TranspositionTable(4)
    table_after = cs.TranspositionTable(4)
    table_checks = 0
    for key in (0, 2**64 - 1, 2**63, 16, 32, 2**63 + 16):
        for depth in (0, 3, 1, 4):
            for score in (-32000, -31001, -31000, -30999, 0, 30999, 31000, 31001, 32000):
                for flag in (0, 1, 2):
                    table_before.store(key, depth, score, flag, 1234, 7)
                    table_after.store(key, depth, score, flag, 1234, 7)
                    assert table_before.table == table_after.table
                    for ply in (0, 3, 15):
                        assert table_before.probe(key, depth, -100, 100, ply) == table_after.probe(
                            key, depth, -100, 100, ply)
                        table_checks += 1

    interruptions = []
    for index, (board, _) in enumerate(boards[:4]):
        history = {cs.hash_of_board(board): 1}
        for limit in (0, 1, 3, 31, 127, 511, 1023, 2047):
            before = reference.Search(reference.TranspositionTable(14), history)
            after = cs.Search(cs.TranspositionTable(14), history)

            def stop_at_limit(target: Any = before, bound: int = limit) -> None:
                target.nodes += 1
                if target.nodes > bound:
                    raise reference.TimeUp

            before._time_check = stop_at_limit
            after._ctx.limit = limit
            for search, exc in ((before, reference.TimeUp), (after, cs.TimeUp)):
                try:
                    search.search_root(board, 8, time.monotonic() + 120)
                except exc:
                    pass
                else:
                    raise AssertionError((index, limit, "did not abort"))
            assert snapshot(before) == snapshot(after), (index, limit, "abort state")
            assert {key: value for key, value in after.seen.items() if value} == history
            interruptions.append({"index": index, "limit": limit, "nodes": after.nodes})

    deadlines = []
    for board, _ in boards[:10]:
        history = {cs.hash_of_board(board): 1}
        search = cs.Search(cs.TranspositionTable(16), history)
        started = time.monotonic()
        try:
            search.search_root(board, 14, started + 0.05)
        except cs.TimeUp:
            pass
        else:
            raise AssertionError("deadline did not abort")
        elapsed = time.monotonic() - started
        assert elapsed < 0.30, elapsed
        assert {key: value for key, value in search.seen.items() if value} == history
        deadlines.append(elapsed)

    event = threading.Event()
    board = boards[0][0]
    search = cs.Search(cs.TranspositionTable(16), {cs.hash_of_board(board): 1}, stop_event=event)
    timer = threading.Timer(0.05, event.set)
    started = time.monotonic()
    timer.start()
    try:
        search.search_root(board, 16, started + 120)
    except cs.TimeUp:
        pass
    else:
        raise AssertionError("worker stop did not abort")
    finally:
        timer.join()
    cancellation_seconds = time.monotonic() - started
    assert cancellation_seconds < 0.30, cancellation_seconds
    result = {"fixed_depth": 6, "comparisons": records,
              "python_seconds": sum(row["python_seconds"] for row in records),
              "compiled_seconds": sum(row["compiled_seconds"] for row in records),
              "table_checks": table_checks, "exact_abort_comparisons": interruptions,
              "deadline_seconds": deadlines, "worker_cancellation_seconds": cancellation_seconds,
              "playing_strength_validated": False}
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in
                     ("comparisons", "exact_abort_comparisons")}), flush=True)


if __name__ == "__main__":
    main()
