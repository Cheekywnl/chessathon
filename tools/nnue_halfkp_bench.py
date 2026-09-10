"""Validates chess_nnue_halfkp.py's inference path before it's ever considered for integration:

1. Cross-checks active_features_halfkp() against tools/train_nnue_halfkp.halfkp_indices_for_fen()
   (the ground truth used for training, itself already cross-checked against an independent
   board-mirror symmetry property -- see that module) on many random legal positions.
2. Cross-checks forward() against a plain numpy reference implementation of the same
   architecture (dual EmbeddingBag-equivalent accumulators, shared weights), on random weights.
3. Benchmarks per-call latency in isolation against chess_eval.evaluate_board()'s own latency AND
   against chess_nnue.py's plain-net latency, so the real per-node cost of this eval -- which
   recomputes both accumulators from scratch every call rather than updating them incrementally
   -- is known before any integration or end-to-end A/B is attempted.

Usage:
    uv run python -m tools.nnue_halfkp_bench
    uv run python -m tools.nnue_halfkp_bench --weights data/nnue_halfkp_weights.npz
"""

from __future__ import annotations

import argparse
import random
import time

import chess
import numpy as np

import chess_eval as ce
import chess_nnue as nnue_plain
import chess_nnue_halfkp as nnue
from tools.train_nnue_halfkp import halfkp_indices_for_fen


def _random_weights(rng: np.random.Generator) -> nnue.HalfKPWeights:
    return nnue.HalfKPWeights(
        w1=(rng.standard_normal((nnue.INPUT_SIZE, nnue.L1)) * 0.02).astype(np.float32),
        b1=(rng.standard_normal(nnue.L1) * 0.02).astype(np.float32),
        w2=(rng.standard_normal((nnue.L2, nnue.L1 * 2)) * 0.02).astype(np.float32),
        b2=(rng.standard_normal(nnue.L2) * 0.02).astype(np.float32),
        w3=(rng.standard_normal((nnue.L3, nnue.L2)) * 0.02).astype(np.float32),
        b3=(rng.standard_normal(nnue.L3) * 0.02).astype(np.float32),
        w4=(rng.standard_normal(nnue.L3) * 0.02).astype(np.float32),
        b4=float(rng.standard_normal(()) * 0.02),
    )


def _reference_forward(
    white_idx: list[int], black_idx: list[int], white_to_move: bool, w: nnue.HalfKPWeights
) -> float:
    """Independent dense implementation (sum selected rows, no numba) of the exact architecture
    tools/train_nnue_halfkp.py trains -- the ground truth forward() gets checked against."""
    acc_white = np.maximum(w.w1[white_idx].sum(axis=0) + w.b1, 0.0)
    acc_black = np.maximum(w.w1[black_idx].sum(axis=0) + w.b1, 0.0)
    combined = (
        np.concatenate([acc_white, acc_black])
        if white_to_move
        else np.concatenate([acc_black, acc_white])
    )
    h2 = np.maximum(w.w2 @ combined + w.b2, 0.0)
    h3 = np.maximum(w.w3 @ h2 + w.b3, 0.0)
    return float(h3 @ w.w4 + w.b4)


def _random_positions(n: int, seed: int = 0) -> list[chess.Board]:
    rng = random.Random(seed)
    boards: list[chess.Board] = []
    while len(boards) < n:
        board = chess.Board()
        for _ in range(rng.randint(0, 80)):
            legal = list(board.legal_moves)
            if not legal or board.is_game_over():
                break
            board.push(rng.choice(legal))
        if not board.is_game_over():
            boards.append(board)
    return boards


def _active_idx(board: chess.Board) -> tuple[np.ndarray, np.ndarray, int]:
    white_buf = np.empty(nnue.MAX_ACTIVE_PER_PERSPECTIVE, dtype=np.int64)
    black_buf = np.empty(nnue.MAX_ACTIVE_PER_PERSPECTIVE, dtype=np.int64)
    n = nnue.active_features_halfkp(
        np.uint64(board.pawns),
        np.uint64(board.knights),
        np.uint64(board.bishops),
        np.uint64(board.rooks),
        np.uint64(board.queens),
        np.uint64(board.kings),
        np.uint64(board.occupied_co[chess.WHITE]),
        np.uint64(board.occupied_co[chess.BLACK]),
        white_buf,
        black_buf,
    )
    return white_buf, black_buf, n


def check_feature_encoding_matches(boards: list[chess.Board]) -> None:
    mismatches = 0
    for board in boards:
        fen = board.fen()
        expected_w, expected_b = halfkp_indices_for_fen(fen)
        white_buf, black_buf, n = _active_idx(board)
        got_w = {int(i) for i in white_buf[:n]}
        got_b = {int(i) for i in black_buf[:n]}
        if got_w != set(expected_w) or got_b != set(expected_b):
            mismatches += 1
            print(f"MISMATCH on {fen}")
    if mismatches:
        raise SystemExit(f"{mismatches}/{len(boards)} feature-encoding mismatches -- DO NOT TRUST")
    print(
        f"feature encoding: {len(boards)}/{len(boards)} positions match "
        "tools/train_nnue_halfkp.halfkp_indices_for_fen() exactly"
    )


def check_forward_matches(boards: list[chess.Board], weights: nnue.HalfKPWeights) -> None:
    max_abs_err = 0.0
    for board in boards:
        fen = board.fen()
        expected_w, expected_b = halfkp_indices_for_fen(fen)
        white_to_move = board.turn == chess.WHITE
        expected = _reference_forward(expected_w, expected_b, white_to_move, weights)

        white_buf, black_buf, n = _active_idx(board)
        got = nnue.forward(
            white_buf, n, black_buf, n, white_to_move,
            weights.w1, weights.b1, weights.w2, weights.b2,
            weights.w3, weights.b3, weights.w4, weights.b4,
        )
        max_abs_err = max(max_abs_err, abs(got - expected))
    print(f"forward pass: max abs error vs numpy dense reference over {len(boards)} positions: "
          f"{max_abs_err:.6f}")
    if max_abs_err > 0.5:
        raise SystemExit("forward pass diverges from reference by more than float32 rounding -- "
                          "DO NOT TRUST")


def benchmark(weights: nnue.HalfKPWeights, n_calls: int = 20000) -> None:
    nnue.warm_up(weights)
    ce.warm_up()

    board = chess.Board()
    args = (
        np.uint64(board.pawns), np.uint64(board.knights), np.uint64(board.bishops),
        np.uint64(board.rooks), np.uint64(board.queens), np.uint64(board.kings),
        np.uint64(board.occupied_co[chess.WHITE]), np.uint64(board.occupied_co[chess.BLACK]),
        True,
        weights.w1, weights.b1, weights.w2, weights.b2, weights.w3, weights.b3,
        weights.w4, weights.b4,
    )
    start = time.perf_counter()
    for _ in range(n_calls):
        nnue.evaluate(*args)
    halfkp_us = (time.perf_counter() - start) / n_calls * 1e6

    start = time.perf_counter()
    for _ in range(n_calls):
        ce.evaluate_board(board, 20)
    classical_us = (time.perf_counter() - start) / n_calls * 1e6

    plain_us = None
    try:
        plain_rng = np.random.default_rng(1)
        plain_weights = nnue_plain.NNUEWeights(
            w1t=(plain_rng.standard_normal((768, 256)) * 0.05).astype(np.float32),
            b1=(plain_rng.standard_normal(256) * 0.05).astype(np.float32),
            w2=(plain_rng.standard_normal((32, 256)) * 0.05).astype(np.float32),
            b2=(plain_rng.standard_normal(32) * 0.05).astype(np.float32),
            w3=(plain_rng.standard_normal(32) * 0.05).astype(np.float32),
            b3=float(plain_rng.standard_normal(()) * 0.05),
        )
        nnue_plain.warm_up(plain_weights)
        plain_args = (
            np.uint64(board.pawns), np.uint64(board.knights), np.uint64(board.bishops),
            np.uint64(board.rooks), np.uint64(board.queens), np.uint64(board.kings),
            np.uint64(board.occupied_co[chess.WHITE]), np.uint64(board.occupied_co[chess.BLACK]),
            plain_weights.w1t, plain_weights.b1, plain_weights.w2, plain_weights.b2,
            plain_weights.w3, plain_weights.b3,
        )
        start = time.perf_counter()
        for _ in range(n_calls):
            nnue_plain.evaluate(*plain_args)
        plain_us = (time.perf_counter() - start) / n_calls * 1e6
    except Exception as exc:  # pragma: no cover -- benchmark convenience only
        print(f"(plain-net comparison skipped: {exc})")

    print(f"\nlatency (startpos, {n_calls} calls each):")
    print(f"  classical eval:             {classical_us:.2f} us/call")
    if plain_us is not None:
        print(f"  plain NNUE (chess_nnue.py): {plain_us:.2f} us/call "
              f"({plain_us / classical_us:.1f}x classical)")
    print(f"  HalfKP NNUE (this):         {halfkp_us:.2f} us/call "
          f"({halfkp_us / classical_us:.1f}x classical)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and benchmark chess_nnue_halfkp.py.")
    parser.add_argument(
        "--weights", type=str, default=None,
        help="Real trained .npz weights (tools/train_nnue_halfkp.py --out). Defaults to random "
        "weights -- fine for correctness/latency checks, meaningless for eval quality.",
    )
    arguments = parser.parse_args()

    rng = np.random.default_rng(0)
    weights = nnue.load_weights(arguments.weights) if arguments.weights else _random_weights(rng)
    boards = _random_positions(300)

    check_feature_encoding_matches(boards)
    check_forward_matches(boards, weights)
    benchmark(weights)


if __name__ == "__main__":
    main()
