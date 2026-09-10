"""Numba-jitted inference for the HalfKP-style, king-relative value network
tools/train_nnue_halfkp.py trains. See that module's docstring for the architecture and the
feature-indexing formulas (verified against chessprogramming.org/NNUE and the Stockfish NNUE
docs, and cross-checked here against that file's reference implementation, including an
independent board-mirror symmetry check -- not just the two implementations agreeing with each
other, which wouldn't catch a bug shared by both).

Not wired into chess_eval.py, chess_search.py, or agent.py yet -- this module only builds and
validates the inference path in isolation, same discipline as chess_nnue.py (the simpler,
plain-piece-square track, which lost a real 264-game SPRT test earlier tonight; this is the
higher-ceiling second attempt referenced in that result's activity-log entry).

Unlike chess_nnue.py, this network's output is already side-to-move-relative (the standard NNUE
convention -- concatenating [accumulator_side_to_move, accumulator_other_side] first is what
lets the net learn tempo), so evaluate() here needs no separate mover-relative sign flip at the
caller; chess_nnue.py's plain net is White-relative and does need one.

Implementation choice, stated explicitly: this recomputes both perspectives' accumulators from
scratch at every call (summing the ~30 active W1 rows per perspective, the same sparse-input
trick chess_nnue.py uses), not full incremental accumulator updates threaded through search's
make/unmake. Real NNUE engines use incremental updates for speed; skipping that here is a
deliberate scope decision to avoid its classic failure mode (stale accumulator state after a
king move, which forces a full recompute for that perspective anyway, or after any subtly wrong
add/remove bookkeeping) under time pressure -- benchmarked before deciding whether the recompute
cost alone is fast enough to be worth pursuing further.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numba import njit

PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = 1, 2, 3, 4, 5, 6

INPUT_SIZE = 40_960
L1 = 256
L2 = 32
L3 = 32
MAX_ACTIVE_PER_PERSPECTIVE = 30


@dataclass(frozen=True)
class HalfKPWeights:
    w1: np.ndarray  # (40960, 256) float32 -- EmbeddingBag weight, already row-per-feature
    b1: np.ndarray  # (256,) float32 -- shared feature-transformer bias
    w2: np.ndarray  # (32, 512) float32
    b2: np.ndarray  # (32,) float32
    w3: np.ndarray  # (32, 32) float32
    b3: np.ndarray  # (32,) float32
    w4: np.ndarray  # (32,) float32 -- flattened, output layer has a single unit
    b4: float


def load_weights(path: str | Path) -> HalfKPWeights:
    """Loads a tools/train_nnue_halfkp.py .npz output."""
    data = np.load(path)
    return HalfKPWeights(
        w1=np.ascontiguousarray(data["W1"], dtype=np.float32),
        b1=np.ascontiguousarray(data["b1"], dtype=np.float32),
        w2=np.ascontiguousarray(data["W2"], dtype=np.float32),
        b2=np.ascontiguousarray(data["b2"], dtype=np.float32),
        w3=np.ascontiguousarray(data["W3"], dtype=np.float32),
        b3=np.ascontiguousarray(data["b3"], dtype=np.float32),
        w4=np.ascontiguousarray(data["W4"].reshape(-1), dtype=np.float32),
        b4=float(data["b4"].reshape(-1)[0]),
    )


@njit(cache=False)
def active_features_halfkp(
    pawns: np.uint64,
    knights: np.uint64,
    bishops: np.uint64,
    rooks: np.uint64,
    queens: np.uint64,
    kings: np.uint64,
    white: np.uint64,
    black: np.uint64,
    white_out: np.ndarray,
    black_out: np.ndarray,
) -> int:
    """Fills white_out/black_out (each must have room for MAX_ACTIVE_PER_PERSPECTIVE) with this
    position's active HalfKP feature indices per perspective, returns how many were written.
    Piece-type numbering matches tools/train_nnue_halfkp.py exactly: PAWN=0, KNIGHT=1, BISHOP=2,
    ROOK=3, QUEEN=4 (kings are never their own feature, only the per-perspective anchor)."""
    white_king_sq = 0
    black_king_sq = 0
    for s in range(64):
        bit = np.uint64(1) << np.uint64(s)
        if kings & white & bit:
            white_king_sq = s
        if kings & black & bit:
            black_king_sq = s
    black_king_mirrored = black_king_sq ^ 56

    piece_bb = (pawns, knights, bishops, rooks, queens)
    n = 0
    for piece_type in range(5):
        bb = piece_bb[piece_type]
        for square in range(64):
            mask = np.uint64(1) << np.uint64(square)
            if bb & white & mask:
                rel_color_white = 0
                white_out[n] = square + (piece_type * 2 + rel_color_white + white_king_sq * 10) * 64
                mirrored_square = square ^ 56
                rel_color_black = 1
                bp_idx = piece_type * 2 + rel_color_black
                black_out[n] = mirrored_square + (bp_idx + black_king_mirrored * 10) * 64
                n += 1
            elif bb & black & mask:
                rel_color_white = 1
                white_out[n] = square + (piece_type * 2 + rel_color_white + white_king_sq * 10) * 64
                mirrored_square = square ^ 56
                rel_color_black = 0
                bp_idx = piece_type * 2 + rel_color_black
                black_out[n] = mirrored_square + (bp_idx + black_king_mirrored * 10) * 64
                n += 1
    return n


@njit(cache=False)
def _accumulate(
    idx: np.ndarray,
    n_active: int,
    w1: np.ndarray,
    b1: np.ndarray,
) -> np.ndarray:
    acc = np.empty(L1, dtype=np.float32)
    for j in range(L1):
        acc[j] = b1[j]
    for k in range(n_active):
        row = w1[idx[k]]
        for j in range(L1):
            acc[j] += row[j]
    for j in range(L1):
        if acc[j] < 0.0:
            acc[j] = 0.0
    return acc


@njit(cache=False)
def forward(
    white_idx: np.ndarray,
    n_white: int,
    black_idx: np.ndarray,
    n_black: int,
    white_to_move: bool,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    w3: np.ndarray,
    b3: np.ndarray,
    w4: np.ndarray,
    b4: float,
) -> float:
    acc_white = _accumulate(white_idx, n_white, w1, b1)
    acc_black = _accumulate(black_idx, n_black, w1, b1)

    combined = np.empty(L1 * 2, dtype=np.float32)
    if white_to_move:
        for j in range(L1):
            combined[j] = acc_white[j]
            combined[L1 + j] = acc_black[j]
    else:
        for j in range(L1):
            combined[j] = acc_black[j]
            combined[L1 + j] = acc_white[j]

    h2 = np.empty(L2, dtype=np.float32)
    for j in range(L2):
        acc = b2[j]
        row = w2[j]
        for k in range(L1 * 2):
            acc += row[k] * combined[k]
        h2[j] = acc if acc > 0.0 else 0.0

    h3 = np.empty(L3, dtype=np.float32)
    for j in range(L3):
        acc = b3[j]
        row = w3[j]
        for k in range(L2):
            acc += row[k] * h2[k]
        h3[j] = acc if acc > 0.0 else 0.0

    out = b4
    for k in range(L3):
        out += w4[k] * h3[k]
    return out


@njit(cache=False)
def evaluate(
    pawns: np.uint64,
    knights: np.uint64,
    bishops: np.uint64,
    rooks: np.uint64,
    queens: np.uint64,
    kings: np.uint64,
    white: np.uint64,
    black: np.uint64,
    white_to_move: bool,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    w3: np.ndarray,
    b3: np.ndarray,
    w4: np.ndarray,
    b4: float,
) -> int:
    """Side-to-move-relative centipawn score -- no sign flip needed at the caller, unlike
    chess_nnue.py's White-relative evaluate()."""
    white_buf = np.empty(MAX_ACTIVE_PER_PERSPECTIVE, dtype=np.int64)
    black_buf = np.empty(MAX_ACTIVE_PER_PERSPECTIVE, dtype=np.int64)
    n = active_features_halfkp(
        pawns, knights, bishops, rooks, queens, kings, white, black, white_buf, black_buf
    )
    raw = forward(
        white_buf, n, black_buf, n, white_to_move, w1, b1, w2, b2, w3, b3, w4, b4
    )
    return round(raw)


def warm_up(weights: HalfKPWeights) -> None:
    """Compiles the jitted signatures now, inside the init budget, not on the clock."""
    start_pawns = np.uint64(0x00FF00000000FF00)
    start_white = np.uint64(0x000000000000FFFF)
    start_black = np.uint64(0xFFFF000000000000)
    evaluate(
        start_pawns,
        np.uint64(0x4200000000000042),
        np.uint64(0x2400000000000024),
        np.uint64(0x8100000000000081),
        np.uint64(0x0800000000000008),
        np.uint64(0x1000000000000010),
        start_white,
        start_black,
        True,
        weights.w1,
        weights.b1,
        weights.w2,
        weights.b2,
        weights.w3,
        weights.b3,
        weights.w4,
        weights.b4,
    )
