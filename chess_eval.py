"""Position evaluation: tapered material + piece-square tables + structure, on raw bitboards.

Everything here reads python-chess's own bitboard attributes (board.pawns, board.knights, ...,
board.occupied_co[color]) rather than board.piece_map(), so the Python side does no per-square
work: it hands eight 64-bit ints to a jitted function and gets a centipawn score back, from
White's point of view. The search negates it for the side to move.

Piece-square tables are generated from a formula (centralisation, king safety, pawn advancement),
not copied from a published table, so every number here is derivable and checkable rather than
a block of magic constants nobody on the team can explain to a judge.

The hand-picked bonuses/penalties (material, mobility, pawn structure, rook files, king safety,
bishop pair, castling) live in one PARAMS array rather than as scattered literals, specifically
so they can be tuned automatically (see tune.py) instead of guessed one at a time -- manual
guesses were hitting diminishing returns. DEFAULT_PARAMS holds today's hand-picked values;
evaluate_board always uses them unless a tuned array is passed in explicitly.
"""

import chess
import numpy as np
from numba import njit

Bitboard = np.uint64

PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = 1, 2, 3, 4, 5, 6

# Indices into the params array. Kept as plain module constants (not an enum) so numba can
# inline them as compile-time literals when indexing the array inside the jitted function.
P_PAWN_MG, P_KNIGHT_MG, P_BISHOP_MG, P_ROOK_MG, P_QUEEN_MG = 0, 1, 2, 3, 4
P_PAWN_EG, P_KNIGHT_EG, P_BISHOP_EG, P_ROOK_EG, P_QUEEN_EG = 5, 6, 7, 8, 9
P_MOBILITY = 10
P_DOUBLED_MG, P_DOUBLED_EG = 11, 12
P_ISOLATED_MG, P_ISOLATED_EG = 13, 14
P_PASSED_MG_PER_RANK, P_PASSED_EG_PER_RANK = 15, 16
P_ROOK_OPEN_MG, P_ROOK_OPEN_EG = 17, 18
P_ROOK_SEMI_MG, P_ROOK_SEMI_EG = 19, 20
P_KING_OPEN_PENALTY, P_KING_SEMI_PENALTY = 21, 22
P_BISHOP_PAIR_MG, P_BISHOP_PAIR_EG = 23, 24
P_CASTLING_MG = 25
NUM_PARAMS = 26

DEFAULT_PARAMS = np.zeros(NUM_PARAMS, dtype=np.int32)
_MG_MATERIAL_IDX = [P_PAWN_MG, P_KNIGHT_MG, P_BISHOP_MG, P_ROOK_MG, P_QUEEN_MG]
_EG_MATERIAL_IDX = [P_PAWN_EG, P_KNIGHT_EG, P_BISHOP_EG, P_ROOK_EG, P_QUEEN_EG]
DEFAULT_PARAMS[_MG_MATERIAL_IDX] = [100, 320, 330, 500, 900]
DEFAULT_PARAMS[_EG_MATERIAL_IDX] = [120, 300, 320, 540, 950]
DEFAULT_PARAMS[P_MOBILITY] = 2
DEFAULT_PARAMS[P_DOUBLED_MG] = -8
DEFAULT_PARAMS[P_DOUBLED_EG] = -16
DEFAULT_PARAMS[P_ISOLATED_MG] = -10
DEFAULT_PARAMS[P_ISOLATED_EG] = -12
DEFAULT_PARAMS[P_PASSED_MG_PER_RANK] = 4
DEFAULT_PARAMS[P_PASSED_EG_PER_RANK] = 18
DEFAULT_PARAMS[P_ROOK_OPEN_MG] = 20
DEFAULT_PARAMS[P_ROOK_OPEN_EG] = 16
DEFAULT_PARAMS[P_ROOK_SEMI_MG] = 10
DEFAULT_PARAMS[P_ROOK_SEMI_EG] = 8
DEFAULT_PARAMS[P_KING_OPEN_PENALTY] = -12
DEFAULT_PARAMS[P_KING_SEMI_PENALTY] = -6
DEFAULT_PARAMS[P_BISHOP_PAIR_MG] = 30
DEFAULT_PARAMS[P_BISHOP_PAIR_EG] = 40
DEFAULT_PARAMS[P_CASTLING_MG] = 15

# Tapered-eval phase weight per piece type; starting position sums to 24.
PHASE_WEIGHT = np.array([0, 0, 1, 1, 2, 4, 0], dtype=np.int32)
MAX_PHASE = 24


def _square_tables() -> tuple[np.ndarray, np.ndarray]:
    """Build (mg, eg) piece-square tables of shape (7, 64), White's perspective, a1=0..h8=63."""
    mg = np.zeros((7, 64), dtype=np.int32)
    eg = np.zeros((7, 64), dtype=np.int32)
    for square in range(64):
        file = square % 8
        rank = square // 8
        center = 3.5 - (abs(file - 3.5) + abs(rank - 3.5)) / 2.0  # +3.5 center .. -3.5 corner

        mg[PAWN][square] = int(6 * max(0.0, rank - 1) + 4 * center) if 0 < rank < 7 else 0
        eg[PAWN][square] = int(14 * max(0.0, rank - 1)) if 0 < rank < 7 else 0

        mg[KNIGHT][square] = int(9 * center)
        eg[KNIGHT][square] = int(9 * center)

        mg[BISHOP][square] = int(6 * center)
        eg[BISHOP][square] = int(6 * center)

        mg[ROOK][square] = 12 if rank == 6 else 0
        mg[ROOK][square] += int(2 * center)
        eg[ROOK][square] = int(4 * center) + (8 if rank == 6 else 0)

        mg[QUEEN][square] = int(4 * center)
        eg[QUEEN][square] = int(8 * center)

        king_shield = 24 if rank == 0 and file in (0, 1, 2, 5, 6, 7) else 0
        mg[KING][square] = king_shield - int(10 * center)
        eg[KING][square] = int(14 * center)
    return mg, eg


MG_PST, EG_PST = _square_tables()


def _passed_masks() -> tuple[np.ndarray, np.ndarray]:
    """passed_mask[color][square]: squares that must be empty of enemy pawns for a passer."""
    white = np.zeros(64, dtype=np.uint64)
    black = np.zeros(64, dtype=np.uint64)
    for square in range(64):
        file = square % 8
        rank = square // 8
        files = [f for f in (file - 1, file, file + 1) if 0 <= f <= 7]
        w_mask = 0
        b_mask = 0
        for f in files:
            for r in range(rank + 1, 8):
                w_mask |= 1 << (r * 8 + f)
            for r in range(0, rank):
                b_mask |= 1 << (r * 8 + f)
        white[square] = np.uint64(w_mask)
        black[square] = np.uint64(b_mask)
    return white, black


PASSED_MASK_WHITE, PASSED_MASK_BLACK = _passed_masks()
FILE_MASK = np.array([np.uint64(0x0101010101010101 << f) for f in range(8)], dtype=np.uint64)


def _adjacent_file_masks() -> np.ndarray:
    masks = np.zeros(8, dtype=np.uint64)
    for f in range(8):
        mask = 0
        if f > 0:
            mask |= int(FILE_MASK[f - 1])
        if f < 7:
            mask |= int(FILE_MASK[f + 1])
        masks[f] = np.uint64(mask)
    return masks


ADJACENT_FILE_MASK = _adjacent_file_masks()


@njit(cache=False)
def _popcount(bb: np.uint64) -> int:
    count = 0
    while bb:
        bb &= bb - np.uint64(1)
        count += 1
    return count


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
    mg_pst: np.ndarray,
    eg_pst: np.ndarray,
    passed_white: np.ndarray,
    passed_black: np.ndarray,
    file_mask: np.ndarray,
    adjacent_file_mask: np.ndarray,
    white_castling_rights: int,
    black_castling_rights: int,
    params: np.ndarray,
) -> int:
    mg_value = params[P_PAWN_MG : P_QUEEN_MG + 1]
    eg_value = params[P_PAWN_EG : P_QUEEN_EG + 1]
    piece_bb = (pawns, knights, bishops, rooks, queens, kings)
    mg = 0
    eg = 0
    phase = 0
    bishop_count_w = 0
    bishop_count_b = 0

    for piece_type in range(1, 7):
        bb = piece_bb[piece_type - 1]
        w_bb = bb & white
        b_bb = bb & black
        w_count = _popcount(w_bb)
        b_count = _popcount(b_bb)
        phase += PHASE_WEIGHT[piece_type] * (w_count + b_count)

        if piece_type == BISHOP:
            bishop_count_w = w_count
            bishop_count_b = b_count

        for square in range(64):
            mask = np.uint64(1) << np.uint64(square)
            if w_bb & mask:
                if piece_type != KING:
                    mg += mg_value[piece_type - 1]
                    eg += eg_value[piece_type - 1]
                mg += mg_pst[piece_type][square]
                eg += eg_pst[piece_type][square]
                if piece_type == PAWN:
                    if pawns & passed_white[square] & black == 0:
                        rank = square // 8
                        mg += params[P_PASSED_MG_PER_RANK] * rank
                        eg += params[P_PASSED_EG_PER_RANK] * rank
                    if _popcount(pawns & white & file_mask[square % 8]) > 1:
                        mg += params[P_DOUBLED_MG]
                        eg += params[P_DOUBLED_EG]
                    if pawns & white & adjacent_file_mask[square % 8] == 0:
                        mg += params[P_ISOLATED_MG]
                        eg += params[P_ISOLATED_EG]
                elif piece_type == ROOK:
                    file_pawns = pawns & file_mask[square % 8]
                    if file_pawns == 0:
                        mg += params[P_ROOK_OPEN_MG]
                        eg += params[P_ROOK_OPEN_EG]
                    elif file_pawns & white == 0:
                        mg += params[P_ROOK_SEMI_MG]
                        eg += params[P_ROOK_SEMI_EG]
                elif piece_type == KING:
                    king_file = square % 8
                    lo = king_file - 1 if king_file > 0 else 0
                    hi = king_file + 1 if king_file < 7 else 7
                    for f in range(lo, hi + 1):
                        fmask = file_mask[f]
                        if pawns & white & fmask == 0:
                            mg += (
                                params[P_KING_OPEN_PENALTY]
                                if pawns & fmask == 0
                                else params[P_KING_SEMI_PENALTY]
                            )
            elif b_bb & mask:
                mirror = square ^ 56
                if piece_type != KING:
                    mg -= mg_value[piece_type - 1]
                    eg -= eg_value[piece_type - 1]
                mg -= mg_pst[piece_type][mirror]
                eg -= eg_pst[piece_type][mirror]
                if piece_type == PAWN:
                    if pawns & passed_black[square] & white == 0:
                        rank = 7 - (square // 8)
                        mg -= params[P_PASSED_MG_PER_RANK] * rank
                        eg -= params[P_PASSED_EG_PER_RANK] * rank
                    if _popcount(pawns & black & file_mask[square % 8]) > 1:
                        mg -= params[P_DOUBLED_MG]
                        eg -= params[P_DOUBLED_EG]
                    if pawns & black & adjacent_file_mask[square % 8] == 0:
                        mg -= params[P_ISOLATED_MG]
                        eg -= params[P_ISOLATED_EG]
                elif piece_type == ROOK:
                    file_pawns = pawns & file_mask[square % 8]
                    if file_pawns == 0:
                        mg -= params[P_ROOK_OPEN_MG]
                        eg -= params[P_ROOK_OPEN_EG]
                    elif file_pawns & black == 0:
                        mg -= params[P_ROOK_SEMI_MG]
                        eg -= params[P_ROOK_SEMI_EG]
                elif piece_type == KING:
                    king_file = square % 8
                    lo = king_file - 1 if king_file > 0 else 0
                    hi = king_file + 1 if king_file < 7 else 7
                    for f in range(lo, hi + 1):
                        fmask = file_mask[f]
                        if pawns & black & fmask == 0:
                            mg -= (
                                params[P_KING_OPEN_PENALTY]
                                if pawns & fmask == 0
                                else params[P_KING_SEMI_PENALTY]
                            )

    if bishop_count_w >= 2:
        mg += params[P_BISHOP_PAIR_MG]
        eg += params[P_BISHOP_PAIR_EG]
    if bishop_count_b >= 2:
        mg -= params[P_BISHOP_PAIR_MG]
        eg -= params[P_BISHOP_PAIR_EG]

    # Castling rights are worth real tempo/safety even before castling happens: an early,
    # unforced king move that forfeits them should cost more than a PST square difference
    # alone would charge it. Endgame-tapered away since the king wants to centralise there.
    mg += params[P_CASTLING_MG] * white_castling_rights
    mg -= params[P_CASTLING_MG] * black_castling_rights

    phase = min(phase, MAX_PHASE)
    tapered = (mg * phase + eg * (MAX_PHASE - phase)) // MAX_PHASE
    return int(tapered)


def _castling_rights_count(board: chess.Board, color: chess.Color) -> int:
    return int(board.has_kingside_castling_rights(color)) + int(
        board.has_queenside_castling_rights(color)
    )


def evaluate_board(board: chess.Board, mobility: int, params: np.ndarray = DEFAULT_PARAMS) -> int:
    """Centipawn score relative to the side to move (negamax convention): positive means the
    position favours whoever is about to play. `mobility` is that side's own legal-move count."""
    positional = evaluate(
        Bitboard(board.pawns),
        Bitboard(board.knights),
        Bitboard(board.bishops),
        Bitboard(board.rooks),
        Bitboard(board.queens),
        Bitboard(board.kings),
        Bitboard(board.occupied_co[chess.WHITE]),
        Bitboard(board.occupied_co[chess.BLACK]),
        MG_PST,
        EG_PST,
        PASSED_MASK_WHITE,
        PASSED_MASK_BLACK,
        FILE_MASK,
        ADJACENT_FILE_MASK,
        _castling_rights_count(board, chess.WHITE),
        _castling_rights_count(board, chess.BLACK),
        params,
    )
    mover_relative = int(positional) if board.turn == chess.WHITE else -int(positional)
    return mover_relative + int(params[P_MOBILITY]) * mobility


def warm_up() -> None:
    """Compile every jitted signature now, inside the init budget, not on the clock."""
    board = chess.Board()
    evaluate_board(board, 20)
    _popcount(np.uint64(2**64 - 1))
