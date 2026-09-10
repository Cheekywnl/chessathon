"""Numba-jitted inference for the small value network tools/train_nnue.py trains.

Not wired into chess_eval.py, chess_search.py, or agent.py yet -- this module only builds and
validates the inference path in isolation (correctness against a reference implementation, and
latency in the competition sandbox's terms: one CPU core, no GPU). Integration is a separate,
later step once real trained weights exist and this path has been benchmarked and validated end
to end, same discipline as everything else in this repo.

Architecture matches tools/train_nnue.py exactly: 768 binary input features (12 piece-planes x 64
squares, White's perspective) -> 256 (ReLU) -> 32 (ReLU) -> 1 linear output, a centipawn score
from White's perspective (chess_eval.py's evaluate() convention -- mover-relative sign flip
happens at the caller, not in here).

The one thing that makes this fast enough to call at every search node: the input layer is a
768-wide one-hot vector with only ~32 non-zero entries (one per piece on the board). Computing
x @ W1 densely is 768*256 multiply-adds; because x is one-hot, the real answer is just "sum the
W1 rows for the ~32 active features" -- ~32*256 additions, no multiplies, roughly 6x fewer
arithmetic ops and it skips 768-32=736 completely wasted multiply-by-zero rows. W1 is stored
transposed (768, 256) so each active feature's contribution is one contiguous row, not a
cache-hostile strided column read.

This is row-sparse recomputation, not Stockfish-style incremental accumulator updates (which
carry the hidden-layer sum across moves and only add/remove the handful of features that changed
between parent and child position). That's a further, larger optimization this doesn't attempt --
deliberately: it requires threading accumulator state through search's make/unmake, a much bigger
and riskier change to the hot path. Recomputing from scratch per node is simpler, easier to get
right under a deadline, and gets benchmarked on its own merits before anything decides whether
that extra complexity is worth it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numba import njit

PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = 1, 2, 3, 4, 5, 6

INPUT_SIZE = 768
HIDDEN1 = 256
HIDDEN2 = 32
MAX_ACTIVE_FEATURES = 32  # a legal position never has more than 32 pieces on the board


@dataclass(frozen=True)
class NNUEWeights:
    w1t: np.ndarray  # (768, 256) float32 -- W1 transposed, row i is feature i's contribution
    b1: np.ndarray  # (256,) float32
    w2: np.ndarray  # (32, 256) float32
    b2: np.ndarray  # (32,) float32
    w3: np.ndarray  # (32,) float32 -- flattened, output layer has a single unit
    b3: float


def load_weights(path: str | Path) -> NNUEWeights:
    """Loads a tools/train_nnue.py .npz output. PyTorch's nn.Linear.weight is (out, in), so W1
    comes out of training as (256, 768); transposed here, once, at load time, not per call."""
    data = np.load(path)
    return NNUEWeights(
        w1t=np.ascontiguousarray(data["W1"].T, dtype=np.float32),
        b1=np.ascontiguousarray(data["b1"], dtype=np.float32),
        w2=np.ascontiguousarray(data["W2"], dtype=np.float32),
        b2=np.ascontiguousarray(data["b2"], dtype=np.float32),
        w3=np.ascontiguousarray(data["W3"].reshape(-1), dtype=np.float32),
        b3=float(data["b3"].reshape(-1)[0]),
    )


@njit(cache=False)
def active_features(
    pawns: np.uint64,
    knights: np.uint64,
    bishops: np.uint64,
    rooks: np.uint64,
    queens: np.uint64,
    kings: np.uint64,
    white: np.uint64,
    black: np.uint64,
    out_idx: np.ndarray,
) -> int:
    """Fills out_idx (must have room for MAX_ACTIVE_FEATURES) with each occupied square's feature
    index and returns how many were written. Plane order matches tools/train_nnue.py's
    _PLANE_FOR_LETTER exactly: white P,N,B,R,Q,K = planes 0-5, black p,n,b,r,q,k = planes 6-11."""
    piece_bb = (pawns, knights, bishops, rooks, queens, kings)
    n = 0
    for piece_type in range(1, 7):
        bb = piece_bb[piece_type - 1]
        for square in range(64):
            mask = np.uint64(1) << np.uint64(square)
            if bb & white & mask:
                out_idx[n] = (piece_type - 1) * 64 + square
                n += 1
            elif bb & black & mask:
                out_idx[n] = (piece_type - 1 + 6) * 64 + square
                n += 1
    return n


@njit(cache=False)
def forward(
    active_idx: np.ndarray,
    n_active: int,
    w1t: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    w3: np.ndarray,
    b3: float,
) -> float:
    """Same math as tools/train_nnue.py's ValueNet.forward(), specialized for a single one-hot
    input built from active_idx instead of a dense 768-vector."""
    h1 = np.empty(HIDDEN1, dtype=np.float32)
    for j in range(HIDDEN1):
        h1[j] = b1[j]
    for k in range(n_active):
        idx = active_idx[k]
        row = w1t[idx]
        for j in range(HIDDEN1):
            h1[j] += row[j]
    for j in range(HIDDEN1):
        if h1[j] < 0.0:
            h1[j] = 0.0

    h2 = np.empty(HIDDEN2, dtype=np.float32)
    for j in range(HIDDEN2):
        acc = b2[j]
        w2_row = w2[j]
        for k in range(HIDDEN1):
            acc += w2_row[k] * h1[k]
        h2[j] = acc if acc > 0.0 else 0.0

    out = b3
    for k in range(HIDDEN2):
        out += w3[k] * h2[k]
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
    w1t: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    w3: np.ndarray,
    b3: float,
) -> int:
    """White-relative centipawn score -- same convention as chess_eval.evaluate(), so it's a
    drop-in replacement for the "positional" half of chess_eval.evaluate_board() once/if this
    gets integrated."""
    idx_buf = np.empty(MAX_ACTIVE_FEATURES, dtype=np.int64)
    n_active = active_features(pawns, knights, bishops, rooks, queens, kings, white, black, idx_buf)
    raw = forward(idx_buf, n_active, w1t, b1, w2, b2, w3, b3)
    return round(raw)


def warm_up(weights: NNUEWeights) -> None:
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
        weights.w1t,
        weights.b1,
        weights.w2,
        weights.b2,
        weights.w3,
        weights.b3,
    )
