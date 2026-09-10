"""Validates chess_nnue.py's inference path before it's ever considered for integration into
chess_eval.py, chess_search.py, or agent.py:

1. Cross-checks active_features() against tools/train_nnue.fen_to_features() on many random
   legal positions -- if these two ever disagreed, weights trained on one feature encoding would
   get evaluated against a different one at inference time, silently and severely wrong.
2. Cross-checks forward() against a plain numpy dense-matmul reference implementation of the
   same architecture, on random weights.
3. Benchmarks per-call latency in isolation against chess_eval.evaluate_board()'s own latency, so
   the real per-node cost of this eval is known before any integration or end-to-end A/B.

Usage:
    uv run python -m tools.nnue_bench
    uv run python -m tools.nnue_bench --weights data/nnue_weights.npz  # real trained weights
"""

from __future__ import annotations

import argparse
import random
import time

import chess
import numpy as np

import chess_eval as ce
import chess_nnue as nnue
from tools.train_nnue import fen_to_features


def _random_weights(rng: np.random.Generator) -> nnue.NNUEWeights:
    return nnue.NNUEWeights(
        w1t=(rng.standard_normal((nnue.INPUT_SIZE, nnue.HIDDEN1)) * 0.05).astype(np.float32),
        b1=(rng.standard_normal(nnue.HIDDEN1) * 0.05).astype(np.float32),
        w2=(rng.standard_normal((nnue.HIDDEN2, nnue.HIDDEN1)) * 0.05).astype(np.float32),
        b2=(rng.standard_normal(nnue.HIDDEN2) * 0.05).astype(np.float32),
        w3=(rng.standard_normal(nnue.HIDDEN2) * 0.05).astype(np.float32),
        b3=float(rng.standard_normal(()) * 0.05),
    )


def _reference_forward(x: np.ndarray, w: nnue.NNUEWeights) -> float:
    """Independent dense-matmul implementation of the exact architecture tools/train_nnue.py
    trains -- the ground truth forward() gets checked against."""
    h1 = np.maximum(x @ w.w1t + w.b1, 0.0)
    h2 = np.maximum(h1 @ w.w2.T + w.b2, 0.0)
    return float(h2 @ w.w3 + w.b3)


def _random_positions(n: int, seed: int = 0) -> list[chess.Board]:
    rng = random.Random(seed)
    boards: list[chess.Board] = []
    while len(boards) < n:
        board = chess.Board()
        for _ in range(rng.randint(0, 60)):
            legal = list(board.legal_moves)
            if not legal or board.is_game_over():
                break
            board.push(rng.choice(legal))
        if not board.is_game_over():
            boards.append(board)
    return boards


def _active_idx(board: chess.Board) -> tuple[np.ndarray, int]:
    idx_buf = np.empty(nnue.MAX_ACTIVE_FEATURES, dtype=np.int64)
    n = nnue.active_features(
        np.uint64(board.pawns),
        np.uint64(board.knights),
        np.uint64(board.bishops),
        np.uint64(board.rooks),
        np.uint64(board.queens),
        np.uint64(board.kings),
        np.uint64(board.occupied_co[chess.WHITE]),
        np.uint64(board.occupied_co[chess.BLACK]),
        idx_buf,
    )
    return idx_buf, n


def check_feature_encoding_matches(boards: list[chess.Board]) -> None:
    mismatches = 0
    for board in boards:
        fen = board.fen()
        expected_idx = set(np.nonzero(fen_to_features(fen))[0].tolist())
        idx_buf, n = _active_idx(board)
        got_idx = {int(i) for i in idx_buf[:n]}
        if got_idx != expected_idx:
            mismatches += 1
            print(f"MISMATCH on {fen}: expected {sorted(expected_idx)}, got {sorted(got_idx)}")
    if mismatches:
        raise SystemExit(f"{mismatches}/{len(boards)} feature-encoding mismatches -- DO NOT TRUST")
    print(
        f"feature encoding: {len(boards)}/{len(boards)} positions match "
        "tools/train_nnue.fen_to_features() exactly"
    )


def check_forward_matches(boards: list[chess.Board], weights: nnue.NNUEWeights) -> None:
    max_abs_err = 0.0
    for board in boards:
        expected = _reference_forward(fen_to_features(board.fen()), weights)
        idx_buf, n = _active_idx(board)
        got = nnue.forward(
            idx_buf, n, weights.w1t, weights.b1, weights.w2, weights.b2, weights.w3, weights.b3
        )
        max_abs_err = max(max_abs_err, abs(got - expected))
    print(f"forward pass: max abs error vs numpy dense reference over {len(boards)} positions: "
          f"{max_abs_err:.6f}")
    if max_abs_err > 0.5:
        raise SystemExit("forward pass diverges from reference by more than float32 rounding -- "
                          "DO NOT TRUST")


def benchmark(weights: nnue.NNUEWeights, n_calls: int = 20000) -> None:
    nnue.warm_up(weights)
    ce.warm_up()

    board = chess.Board()
    nnue_args = (
        np.uint64(board.pawns), np.uint64(board.knights), np.uint64(board.bishops),
        np.uint64(board.rooks), np.uint64(board.queens), np.uint64(board.kings),
        np.uint64(board.occupied_co[chess.WHITE]), np.uint64(board.occupied_co[chess.BLACK]),
        weights.w1t, weights.b1, weights.w2, weights.b2, weights.w3, weights.b3,
    )
    start = time.perf_counter()
    for _ in range(n_calls):
        nnue.evaluate(*nnue_args)
    nnue_us = (time.perf_counter() - start) / n_calls * 1e6

    start = time.perf_counter()
    for _ in range(n_calls):
        ce.evaluate_board(board, 20)
    classical_us = (time.perf_counter() - start) / n_calls * 1e6

    print(f"\nlatency (startpos, {n_calls} calls each):")
    print(f"  classical eval: {classical_us:.2f} us/call")
    print(f"  nnue eval:      {nnue_us:.2f} us/call  ({nnue_us / classical_us:.1f}x classical)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and benchmark chess_nnue.py.")
    parser.add_argument(
        "--weights", type=str, default=None,
        help="Real trained .npz weights (tools/train_nnue.py --out). Defaults to random weights "
        "-- fine for correctness/latency checks, meaningless for eval quality.",
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
