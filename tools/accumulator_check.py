"""Compare cached raw sums with full feature refreshes after every legal move."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import chess
import numpy as np

import agent
import chess_eval as ce
import chess_halfkp_int as qi
import chess_search as cs
import chess_search_compiled as compiled
from chess_nnue_halfkp import active_features_halfkp


def check_position(board: chess.Board, ctx: Any) -> None:
    state = cs.state_from_board(board)
    white = np.empty(30, dtype=np.int64)
    black = np.empty(30, dtype=np.int64)
    count = active_features_halfkp(*state[:8], white, black)
    expected_white = ctx.b1.astype(np.int32) + ctx.w1[white[:count]].sum(axis=0, dtype=np.int32)
    expected_black = ctx.b1.astype(np.int32) + ctx.w1[black[:count]].sum(axis=0, dtype=np.int32)
    expected = qi.evaluate(*state[:9], ctx.w1, ctx.b1, ctx.w2, ctx.b2, ctx.w3, ctx.b3,
                           ctx.w4, ctx.b4, ctx.scale2, ctx.scale3, ctx.divisor)
    actual = compiled.cached_neural(state, ctx)
    assert actual == expected, (board.fen(), "score", expected, actual)
    assert np.array_equal(ctx.acc_white, expected_white), (board.fen(), "white raw sum")
    assert np.array_equal(ctx.acc_black, expected_black), (board.fen(), "black raw sum")
    assert np.array_equal(ctx.previous, np.asarray(state[:8], dtype=np.uint64))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    agent._stop_pondering()
    weights = qi.load_weights(args.weights)
    ctx = compiled.create_context(compiled.Table(8), {}, ce.DEFAULT_PARAMS, weights, 75, None)
    fens = (
        chess.STARTING_FEN,
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
        "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
        "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
        "4k3/8/8/8/3pP3/8/8/4K3 b - e3 0 1",
        "1r2k3/P7/8/8/8/8/7p/4K3 w - - 0 1",
        "4k3/7P/8/8/8/8/p7/1R2K3 b - - 0 1",
    )
    boards = [chess.Board(fen) for fen in fens]
    rng = random.Random(20260911)
    board = chess.Board()
    for index in range(1800):
        if board.is_game_over() or index % 150 == 0:
            board = chess.Board()
        board.push(rng.choice(list(board.legal_moves)))
        boards.append(board.copy(stack=False))
    transitions: Counter[str] = Counter()
    position_checks = 0
    for board in boards:
        assert board.is_valid(), board.fen()
        check_position(board, ctx)
        position_checks += 1
        for move in list(board.legal_moves):
            kind = ("castling" if board.is_castling(move) else
                    "en_passant" if board.is_en_passant(move) else
                    "capture_promotion" if move.promotion and board.is_capture(move) else
                    "promotion" if move.promotion else
                    "capture" if board.is_capture(move) else
                    "king" if board.piece_type_at(move.from_square) == chess.KING else "quiet")
            mover = "white" if board.turn else "black"
            board.push(move)
            check_position(board, ctx)
            board.pop()
            check_position(board, ctx)
            transitions[f"{mover}_{kind}"] += 1
            position_checks += 2
        # Changing the side to move changes dense-input order, not raw sums.
        board.push(chess.Move.null())
        check_position(board, ctx)
        board.pop()
        check_position(board, ctx)
        position_checks += 2
    for color in ("white", "black"):
        for kind in ("castling", "en_passant", "capture_promotion", "promotion", "capture",
                     "king", "quiet"):
            assert transitions[f"{color}_{kind}"] > 0, (color, kind)
    result = {"width": len(weights.b1), "parent_child_null_position_checks": position_checks,
              "legal_transitions": sum(transitions.values()), "move_types": dict(transitions),
              "mismatches": 0, "playing_strength_validated": False}
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
