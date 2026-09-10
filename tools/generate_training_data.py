"""Builds a Texel-tuning dataset: (fen, mobility, result) rows, one per sampled position, where
result is the eventual game outcome from White's perspective (1.0/0.5/0.0). Two sources:

- Self-play games at a fast, fixed time budget (games between the current engine and itself --
  don't need strong play for tuning, just positions that are roughly representative of real games
  and have a real eventual outcome attached).
- The real games already in games/real/, which have actual opponents and real results, as a
  supplement self-play alone can't provide.

Skips the first few plies (still following curated-position/opening theory, not informative for
material/structural eval terms) and samples periodically rather than every ply (consecutive
positions are near-duplicates and would just overweight whichever games ran long).

Usage:
    uv run python -m tools.generate_training_data --games 150 --out data/tune_positions.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import time
from pathlib import Path
from typing import Any, TextIO

import chess
import chess.pgn

import chess_eval as ce
import chess_movegen as mg
import chess_search as cs
import chess_state as cst

TT_SIZE_POWER = 17
MOVE_BUDGET_S = 0.15
SKIP_OPENING_PLIES = 10
SAMPLE_EVERY = 4


def _think(search: cs.Search, board: chess.Board, budget_s: float) -> chess.Move:
    deadline = time.monotonic() + budget_s
    best_move, completed, last_score, depth = next(iter(board.legal_moves)), 0, 0, 1
    while depth <= 12:
        prev_score = last_score if completed > 0 else None
        try:
            move, score, _ = search.search_root(board, depth, deadline, prev_score=prev_score)
        except cs.TimeUp:
            break
        best_move, completed, last_score = move, depth, score
        if abs(score) >= cs.MATE_THRESHOLD:
            break
        depth += 1
    return best_move


def _result_from_outcome(outcome: chess.Outcome | None) -> float | None:
    if outcome is None or outcome.winner is None:
        return 0.5 if outcome is not None else None
    return 1.0 if outcome.winner == chess.WHITE else 0.0


def self_play_games(games: int, writer: Any, out_file: TextIO) -> None:
    """Writes each game's sampled positions to `writer` as soon as that game finishes, and
    flushes immediately -- a long unattended run (this is meant to run for hours, generating
    data for tools/train_nnue.py) should never risk losing everything to a late crash or an
    impatient kill, and a partial file should always be safe to read mid-run."""
    for game in range(games):
        board = chess.Board()
        tt_white = cs.TranspositionTable(TT_SIZE_POWER)
        tt_black = cs.TranspositionTable(TT_SIZE_POWER)
        hist_white: dict[int, int] = {}
        hist_black: dict[int, int] = {}
        positions: list[tuple[str, int]] = []

        for ply in range(300):
            outcome = board.outcome(claim_draw=True)
            if outcome is not None:
                break
            legal = list(board.legal_moves)
            if not legal:
                break
            if ply >= SKIP_OPENING_PLIES and ply % SAMPLE_EVERY == 0 and not board.is_check():
                positions.append((board.fen(), len(legal)))

            is_white = board.turn == chess.WHITE
            tt = tt_white if is_white else tt_black
            hist = hist_white if is_white else hist_black
            key = cs.hash_of_board(board)
            hist[key] = hist.get(key, 0) + 1
            search = cs.Search(tt, hist)
            move = _think(search, board, MOVE_BUDGET_S)
            board.push(move)
            after_key = cs.hash_of_board(board)
            hist[after_key] = hist.get(after_key, 0) + 1

        result = _result_from_outcome(board.outcome(claim_draw=True))
        if result is None:
            continue
        for fen, mobility in positions:
            writer.writerow([fen, mobility, result])
        out_file.flush()
        print(
            f"self-play game {game + 1}/{games}: {len(positions)} positions, result={result}",
            flush=True,
        )


def real_game_rows() -> list[tuple[str, int, float]]:
    rows: list[tuple[str, int, float]] = []
    root = Path(__file__).resolve().parent.parent
    paths = sorted(glob.glob(str(root / "games" / "real" / "*.pgn"))) + sorted(
        glob.glob(str(root / "game_logs" / "*.pgn"))
    )
    for path in paths:
        with open(path) as f:
            game = chess.pgn.read_game(f)
        if game is None:
            continue
        result_header = game.headers.get("Result", "*")
        if result_header == "1-0":
            result = 1.0
        elif result_header == "0-1":
            result = 0.0
        elif result_header == "1/2-1/2":
            result = 0.5
        else:
            continue
        board = game.board()
        for ply, move in enumerate(game.mainline_moves()):
            if ply >= SKIP_OPENING_PLIES and ply % SAMPLE_EVERY == 0 and not board.is_check():
                rows.append((board.fen(), len(list(board.legal_moves)), result))
            board.push(move)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Texel-tuning / NNUE dataset.")
    parser.add_argument("--games", type=int, default=150)
    parser.add_argument("--out", type=Path, default=Path("data/tune_positions.csv"))
    parser.add_argument("--skip-self-play", action="store_true")
    arguments = parser.parse_args()

    ce.warm_up()
    mg.warm_up()
    cst.warm_up()

    real_rows = real_game_rows()
    print(f"real games: {len(real_rows)} positions", flush=True)

    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    with open(arguments.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fen", "mobility", "result"])
        writer.writerows(real_rows)
        f.flush()
        if not arguments.skip_self_play:
            self_play_games(arguments.games, writer, f)

    print(f"\nwrote data to {arguments.out}")


if __name__ == "__main__":
    main()
