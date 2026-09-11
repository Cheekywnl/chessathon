"""Shared parameters of this team's search."""

import math

import numpy as np

MATE = 32_000
MATE_THRESHOLD = MATE - 1_000
DRAW = 0
MAX_PLY = 128
NO_MOVE = -1
# Contempt: a draw isn't valued at a flat 0 -- it's scored as mildly bad for us specifically,
# not neutral, so the search prefers a continuation that keeps winning chances alive over one
# that settles for a repetition when the two look otherwise equal. Motivated by this project's
# own observed behaviour, not abstract theory: multiple real bugs this session (K+R vs K, 2Q vs
# K, a Lucena position) were the engine settling for a draw-by-repetition instead of continuing
# to press a won position, because a draw and "still winning but not yet resolved" scored the
# same at 0. Small and one-sided by construction (see _draw_score) -- it only ever nudges among
# moves that already look roughly equal, it can't override a real material/tactical verdict, and
# it never discourages accepting a draw when we're actually worse off, which would be irrational.
CONTEMPT = 20

FLAG_EXACT, FLAG_LOWER, FLAG_UPPER = 0, 1, 2

PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = 1, 2, 3, 4, 5, 6
PIECE_VALUES = np.array([100, 320, 330, 500, 900, 20_000], dtype=np.int64)

NULL_MOVE_REDUCTION = 2
NODES_PER_TIME_CHECK = 1024
REVERSE_FUTILITY_DEPTH = 3
REVERSE_FUTILITY_MARGIN_PER_PLY = 120
DELTA_MARGIN = 200
LATE_MOVE_PRUNING_DEPTH = 2
LATE_MOVE_PRUNING_BASE = 6
LATE_MOVE_PRUNING_PER_DEPTH = 3
FUTILITY_DEPTH = 3
FUTILITY_MARGIN_PER_PLY = 150
ASPIRATION_INITIAL_MARGIN = 25

# Late move reduction table: reduction grows with both depth and move index (log-product, the
# standard formula -- see chessprogramming.org/Late_Move_Reductions), replacing the previous
# flat "reduce by 1" rule. A move that's both late *and* found at a deep search gets reduced
# more than a merely-late move at a shallow one; the flat rule couldn't tell those apart.
# Precomputed once at import, not per-node -- log() in the hot loop would cost more than the
# search-tree savings this buys.
_LMR_MAX_DEPTH = 64
_LMR_MAX_MOVE_INDEX = 96
LMR_TABLE = np.zeros((_LMR_MAX_DEPTH + 1, _LMR_MAX_MOVE_INDEX + 1), dtype=np.int32)
for _d in range(1, _LMR_MAX_DEPTH + 1):
    for _m in range(1, _LMR_MAX_MOVE_INDEX + 1):
        LMR_TABLE[_d, _m] = int(0.5 + math.log(_d) * math.log(_m) / 2.0)
