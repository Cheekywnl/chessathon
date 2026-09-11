"""Raw bitboard game-state operations: apply a move, hash a position, and evaluate a capture
sequence (SEE) -- all without ever touching a chess.Board. This is what lets the search tree
recurse on eight uint64s and a handful of scalars instead of a chess.Board object, removing the
push()/pop() and chess.Move-construction overhead that persisted even after chess_movegen.py
replaced move generation itself.

State is threaded through the search as plain values (pawns, knights, ..., black, turn,
castling_rights, ep_square, halfmove_clock) -- immutable by convention, never mutated in place.
Recursion naturally "unmakes" a move simply by not propagating the child call's state back up;
there is deliberately no explicit unmake_move, which removes an entire class of bugs (forgetting
to restore some field on the way back out) at the cost of a few extra words copied per ply, which
is free next to what push()/pop() cost.

Moves are packed into a single int (from_sq | to_sq << 6 | promotion << 12) for cheap storage in
killer/history tables and TT entries, instead of chess.Move objects. chess.Move is only ever
constructed once, at the very root, to hand back a UCI string.
"""

import numpy as np
from numba import njit

import chess_movegen as mg
from chess_bits import lsb_index

PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = 1, 2, 3, 4, 5, 6

RawState = tuple[
    np.uint64, np.uint64, np.uint64, np.uint64, np.uint64, np.uint64,
    np.uint64, np.uint64, bool, np.uint64, int, int,
]

_bit = mg._bit
_sliding_attacks = mg._sliding_attacks
BISHOP_DIRS = mg.BISHOP_DIRS
ROOK_DIRS = mg.ROOK_DIRS
KNIGHT_ATTACKS = mg.KNIGHT_ATTACKS
KING_ATTACKS = mg.KING_ATTACKS
PAWN_ATTACKS_WHITE = mg.PAWN_ATTACKS_WHITE
PAWN_ATTACKS_BLACK = mg.PAWN_ATTACKS_BLACK

# Corner squares that carry castling rights, python-chess's own convention (a bitboard with a
# bit set per rook that still has its right, not per side).
A1, H1, A8, H8 = 0, 7, 56, 63


def pack_move(from_sq: int, to_sq: int, promotion: int) -> int:
    return from_sq | (to_sq << 6) | (promotion << 12)


def unpack_move(packed: int) -> tuple[int, int, int]:
    return packed & 0x3F, (packed >> 6) & 0x3F, (packed >> 12) & 0xF


@njit(cache=False)
def _piece_type_at(
    sq_bit: np.uint64,
    pawns: np.uint64,
    knights: np.uint64,
    bishops: np.uint64,
    rooks: np.uint64,
    queens: np.uint64,
) -> int:
    """0 means none-of-the-above (caller already knows it's a king, or the square is empty)."""
    if pawns & sq_bit:
        return PAWN
    if knights & sq_bit:
        return KNIGHT
    if bishops & sq_bit:
        return BISHOP
    if rooks & sq_bit:
        return ROOK
    if queens & sq_bit:
        return QUEEN
    return 0


@njit(cache=False)
def move_type_of(
    pawns: np.uint64,
    kings: np.uint64,
    from_sq: int,
    to_sq: int,
    ep_square: int,
) -> int:
    """0 normal, 1 en passant, 2 kingside castle, 3 queenside castle. Inferred from the move
    itself and the pre-move state, the same way python-chess's own push() infers it -- neither
    chess.Move nor the raw (from, to, promotion) triple carries this explicitly."""
    from_bit = _bit(from_sq)
    if kings & from_bit:
        diff = to_sq - from_sq
        if diff == 2:
            return 2
        if diff == -2:
            return 3
        return 0
    if (pawns & from_bit) and to_sq == ep_square and ep_square >= 0:
        return 1
    return 0


@njit(cache=False)
def make_move(
    pawns: np.uint64,
    knights: np.uint64,
    bishops: np.uint64,
    rooks: np.uint64,
    queens: np.uint64,
    kings: np.uint64,
    white: np.uint64,
    black: np.uint64,
    turn: bool,
    castling_rights: np.uint64,
    ep_square: int,
    halfmove_clock: int,
    from_sq: int,
    to_sq: int,
    promotion: int,
) -> tuple[
    np.uint64, np.uint64, np.uint64, np.uint64, np.uint64, np.uint64,
    np.uint64, np.uint64, bool, np.uint64, int, int,
]:
    """Apply one legal move to raw state, returning the complete new state. `turn` here is the
    mover's colour (True=white); the returned turn is already flipped."""
    from_bit = _bit(from_sq)
    to_bit = _bit(to_sq)
    mtype = move_type_of(pawns, kings, from_sq, to_sq, ep_square)

    is_capture = mtype == 1 or ((white | black) & to_bit) != 0
    moving_pawn = (pawns & from_bit) != 0
    new_halfmove = 0 if (is_capture or moving_pawn) else halfmove_clock + 1

    npawns, nknights, nbishops, nrooks, nqueens, nkings, nwhite, nblack = mg._apply_move_bitboards(
        pawns, knights, bishops, rooks, queens, kings, white, black,
        from_sq, to_sq, promotion, mtype, turn,
    )

    new_castling = castling_rights
    if kings & from_bit:
        home_mask = (_bit(A1) | _bit(H1)) if turn else (_bit(A8) | _bit(H8))
        new_castling &= ~home_mask
    new_castling &= ~from_bit
    new_castling &= ~to_bit

    new_ep = -1
    if moving_pawn:
        diff = to_sq - from_sq
        if diff == 16 or diff == -16:
            candidate = (from_sq + to_sq) // 2
            # Match python-chess's own ep_square/has_legal_en_passant distinction: only report
            # an en passant square when a pawn could actually (pseudo-legally) capture there --
            # this is what board._transposition_key() uses for repetition/hash purposes, and
            # what FEN output shows. A full legality check (pin/check safety of that capture) is
            # deliberately not done here to keep make_move cheap; generate_legal_moves_bb still
            # verifies full legality before ever offering the capture as a move.
            ep_table = PAWN_ATTACKS_WHITE if turn else PAWN_ATTACKS_BLACK
            capturer_side = nblack if turn else nwhite
            if ep_table[candidate] & npawns & capturer_side:
                new_ep = candidate

    return (
        npawns, nknights, nbishops, nrooks, nqueens, nkings, nwhite, nblack,
        not turn, new_castling, new_ep, new_halfmove,
    )


# --- Zobrist hashing: a fixed pseudo-random table, generated once at import with a fixed seed.
# Only needs to be internally consistent within one process (TT keys and repetition detection
# never need to match across process restarts), so determinism across runs isn't a requirement,
# just convenient for debugging. zobrist_hash remains the independent full-state reference.
# make_move_info updates a known parent hash from before/after bitboard differences, including
# castling rights and en passant. Both paths retain the same pseudo-legal en passant convention.
_rng = np.random.default_rng(20260907)
ZOBRIST_PIECE_SQUARE = _rng.integers(0, 2**63, size=(2, 6, 64), dtype=np.int64).astype(np.uint64)
ZOBRIST_CASTLING = _rng.integers(0, 2**63, size=64, dtype=np.int64).astype(np.uint64)
ZOBRIST_EP_FILE = _rng.integers(0, 2**63, size=8, dtype=np.int64).astype(np.uint64)
ZOBRIST_TURN = np.uint64(int(_rng.integers(0, 2**63, dtype=np.int64)))


@njit(cache=False)
def _lsb_index(bb: np.uint64) -> int:
    """Index of the lowest set bit using Numba's host compilation."""
    return lsb_index(bb)


@njit(cache=False)
def zobrist_hash(
    pawns: np.uint64,
    knights: np.uint64,
    bishops: np.uint64,
    rooks: np.uint64,
    queens: np.uint64,
    kings: np.uint64,
    white: np.uint64,
    black: np.uint64,
    turn: bool,
    castling_rights: np.uint64,
    ep_square: int,
    piece_square: np.ndarray,
    castling_table: np.ndarray,
    ep_file_table: np.ndarray,
    turn_key: np.uint64,
) -> np.uint64:
    h = np.uint64(0)
    piece_bb = (pawns, knights, bishops, rooks, queens, kings)
    for piece_type in range(1, 7):
        white_bb = piece_bb[piece_type - 1] & white
        while white_bb:
            sq = _lsb_index(white_bb)
            h ^= piece_square[0, piece_type - 1, sq]
            white_bb &= white_bb - np.uint64(1)
        black_bb = piece_bb[piece_type - 1] & black
        while black_bb:
            sq = _lsb_index(black_bb)
            h ^= piece_square[1, piece_type - 1, sq]
            black_bb &= black_bb - np.uint64(1)
    rights = castling_rights
    while rights:
        sq = _lsb_index(rights)
        h ^= castling_table[sq]
        rights &= rights - np.uint64(1)
    if ep_square >= 0:
        h ^= ep_file_table[ep_square % 8]
    if turn:
        h ^= turn_key
    return h


@njit(cache=False)
def make_move_info(
    state: RawState, parent_key: np.uint64, from_sq: int, to_sq: int, promotion: int,
) -> tuple[RawState, np.uint64, bool]:
    """Compute the child, its exact hash and check status in one compiled call.

    Hash changes follow the before/after bitboards, including castling, captures
    and promotions. This keeps all special-move rules in make_move itself.
    """
    child = make_move(
        state[0], state[1], state[2], state[3], state[4], state[5], state[6], state[7],
        state[8], state[9], state[10], state[11], from_sq, to_sq, promotion,
    )
    key = parent_key ^ ZOBRIST_TURN
    pieces_before = (state[0], state[1], state[2], state[3], state[4], state[5])
    pieces_after = (child[0], child[1], child[2], child[3], child[4], child[5])
    for piece in range(6):
        for color in range(2):
            occupied_before = state[6] if color == 0 else state[7]
            occupied_after = child[6] if color == 0 else child[7]
            delta = ((pieces_before[piece] & occupied_before)
                     ^ (pieces_after[piece] & occupied_after))
            while delta:
                square = _lsb_index(delta)
                key ^= ZOBRIST_PIECE_SQUARE[color, piece, square]
                delta &= delta - np.uint64(1)
    rights = state[9] ^ child[9]
    while rights:
        square = _lsb_index(rights)
        key ^= ZOBRIST_CASTLING[square]
        rights &= rights - np.uint64(1)
    if state[10] >= 0:
        key ^= ZOBRIST_EP_FILE[state[10] % 8]
    if child[10] >= 0:
        key ^= ZOBRIST_EP_FILE[child[10] % 8]
    own = child[6] if child[8] else child[7]
    king = _lsb_index(child[5] & own)
    check = mg.is_square_attacked(
        king, not child[8], child[0], child[1], child[2], child[3], child[4], child[5],
        child[6], child[7],
    )
    return child, key, check


def hash_state(
    pawns: int, knights: int, bishops: int, rooks: int, queens: int, kings: int,
    white: int, black: int, turn: bool, castling_rights: int, ep_square: int,
) -> int:
    return int(
        zobrist_hash(
            np.uint64(pawns), np.uint64(knights), np.uint64(bishops), np.uint64(rooks),
            np.uint64(queens), np.uint64(kings), np.uint64(white), np.uint64(black),
            turn, np.uint64(castling_rights), ep_square,
            ZOBRIST_PIECE_SQUARE, ZOBRIST_CASTLING, ZOBRIST_EP_FILE, ZOBRIST_TURN,
        )
    )


@njit(cache=False)
def attackers_to(
    sq: int,
    pawns: np.uint64,
    knights: np.uint64,
    bishops: np.uint64,
    rooks: np.uint64,
    queens: np.uint64,
    kings: np.uint64,
    white: np.uint64,
    black: np.uint64,
) -> np.uint64:
    """All pieces of either colour currently attacking `sq`, given the piece bitboards as they
    stand right now (used mid-exchange during SEE, where pieces are removed as they trade off)."""
    occupied = white | black
    attackers = PAWN_ATTACKS_BLACK[sq] & pawns & white
    attackers |= PAWN_ATTACKS_WHITE[sq] & pawns & black
    attackers |= KNIGHT_ATTACKS[sq] & knights
    attackers |= KING_ATTACKS[sq] & kings
    diag = _sliding_attacks(sq, occupied, BISHOP_DIRS)
    straight = _sliding_attacks(sq, occupied, ROOK_DIRS)
    attackers |= diag & (bishops | queens)
    attackers |= straight & (rooks | queens)
    return np.uint64(attackers)


@njit(cache=False)
def see_raw(
    pawns: np.uint64,
    knights: np.uint64,
    bishops: np.uint64,
    rooks: np.uint64,
    queens: np.uint64,
    kings: np.uint64,
    white: np.uint64,
    black: np.uint64,
    from_sq: int,
    to_sq: int,
    ep_square: int,
    moving_white: bool,
    piece_values: np.ndarray,
) -> int:
    """Static exchange evaluation on raw bitboards: net centipawn material if all attackers on
    `to_sq` trade off in least-valuable-first order. Does not verify that each intermediate
    recapture is itself legal (a pin or a king moving into check mid-exchange is not detected) --
    a standard simplification for a move-ordering heuristic, not a legality-critical path; every
    move actually played still goes through the full legal move generator regardless of what SEE
    says about it."""
    from_bit = _bit(from_sq)
    to_bit = _bit(to_sq)
    is_en_passant = (pawns & from_bit) != 0 and to_sq == ep_square and ep_square >= 0

    if is_en_passant:
        gain0 = piece_values[PAWN - 1]
    else:
        captured_type = _piece_type_at(to_bit, pawns, knights, bishops, rooks, queens)
        if captured_type == 0 and (kings & to_bit) == 0 and ((white | black) & to_bit) == 0:
            gain0 = 0
        elif captured_type == 0:
            gain0 = piece_values[KING - 1]
        else:
            gain0 = piece_values[captured_type - 1]

    attacker_type = _piece_type_at(from_bit, pawns, knights, bishops, rooks, queens)
    if attacker_type == 0:
        attacker_type = KING

    s_pawns, s_knights, s_bishops = pawns, knights, bishops
    s_rooks, s_queens, s_kings = rooks, queens, kings
    s_white, s_black = white, black

    if attacker_type == PAWN:
        s_pawns &= ~from_bit
    elif attacker_type == KNIGHT:
        s_knights &= ~from_bit
    elif attacker_type == BISHOP:
        s_bishops &= ~from_bit
    elif attacker_type == ROOK:
        s_rooks &= ~from_bit
    elif attacker_type == QUEEN:
        s_queens &= ~from_bit
    else:
        s_kings &= ~from_bit
    if moving_white:
        s_white &= ~from_bit
    else:
        s_black &= ~from_bit

    if is_en_passant:
        cap_sq = to_sq - 8 if moving_white else to_sq + 8
        cap_bit = _bit(cap_sq)
        s_pawns &= ~cap_bit
        if moving_white:
            s_black &= ~cap_bit
        else:
            s_white &= ~cap_bit
    else:
        s_pawns &= ~to_bit
        s_knights &= ~to_bit
        s_bishops &= ~to_bit
        s_rooks &= ~to_bit
        s_queens &= ~to_bit
        s_kings &= ~to_bit
        s_white &= ~to_bit
        s_black &= ~to_bit

    if attacker_type == PAWN:
        s_pawns |= to_bit
    elif attacker_type == KNIGHT:
        s_knights |= to_bit
    elif attacker_type == BISHOP:
        s_bishops |= to_bit
    elif attacker_type == ROOK:
        s_rooks |= to_bit
    elif attacker_type == QUEEN:
        s_queens |= to_bit
    else:
        s_kings |= to_bit
    if moving_white:
        s_white |= to_bit
    else:
        s_black |= to_bit

    gains = np.empty(32, dtype=np.int64)
    gains[0] = gain0
    depth = 0
    current_value = piece_values[attacker_type - 1]
    side_white = not moving_white

    while depth < 31:
        side_mask = s_white if side_white else s_black
        attackers = attackers_to(
            to_sq, s_pawns, s_knights, s_bishops, s_rooks, s_queens, s_kings, s_white, s_black
        )
        side_attackers = attackers & side_mask
        if side_attackers == 0:
            break

        best_sq = -1
        best_val = 1_000_000
        best_type = 0
        for s in range(64):
            if side_attackers & _bit(s):
                t = _piece_type_at(_bit(s), s_pawns, s_knights, s_bishops, s_rooks, s_queens)
                if t == 0:
                    t = KING
                v = piece_values[t - 1]
                if v < best_val:
                    best_val = v
                    best_sq = s
                    best_type = t

        depth += 1
        gains[depth] = current_value - gains[depth - 1]

        bsq_bit = _bit(best_sq)
        if best_type == PAWN:
            s_pawns &= ~bsq_bit
        elif best_type == KNIGHT:
            s_knights &= ~bsq_bit
        elif best_type == BISHOP:
            s_bishops &= ~bsq_bit
        elif best_type == ROOK:
            s_rooks &= ~bsq_bit
        elif best_type == QUEEN:
            s_queens &= ~bsq_bit
        else:
            s_kings &= ~bsq_bit
        if side_white:
            s_white &= ~bsq_bit
        else:
            s_black &= ~bsq_bit

        s_pawns &= ~to_bit
        s_knights &= ~to_bit
        s_bishops &= ~to_bit
        s_rooks &= ~to_bit
        s_queens &= ~to_bit
        s_kings &= ~to_bit
        s_white &= ~to_bit
        s_black &= ~to_bit

        if best_type == PAWN:
            s_pawns |= to_bit
        elif best_type == KNIGHT:
            s_knights |= to_bit
        elif best_type == BISHOP:
            s_bishops |= to_bit
        elif best_type == ROOK:
            s_rooks |= to_bit
        elif best_type == QUEEN:
            s_queens |= to_bit
        else:
            s_kings |= to_bit
        if side_white:
            s_white |= to_bit
        else:
            s_black |= to_bit

        current_value = best_val
        side_white = not side_white

    for i in range(depth - 1, -1, -1):
        neg = -gains[i]
        nxt = gains[i + 1]
        gains[i] = -(neg if neg > nxt else nxt)
    return int(gains[0])


def warm_up() -> None:
    """Compile every jitted signature now, at import, not on the clock."""
    import chess

    board = chess.Board()
    state = (
        np.uint64(board.pawns), np.uint64(board.knights), np.uint64(board.bishops),
        np.uint64(board.rooks), np.uint64(board.queens), np.uint64(board.kings),
        np.uint64(board.occupied_co[True]), np.uint64(board.occupied_co[False]),
    )
    make_move(*state, True, np.uint64(board.clean_castling_rights()), -1, 0, 12, 28, 0)
    parent_key = hash_state(
        board.pawns, board.knights, board.bishops, board.rooks, board.queens, board.kings,
        board.occupied_co[True], board.occupied_co[False], True, board.clean_castling_rights(), -1,
    )
    make_move_info((*state, True, np.uint64(board.clean_castling_rights()), -1, 0),
                   np.uint64(parent_key), 12, 28, 0)
    piece_values = np.array([100, 320, 330, 500, 900, 20_000], dtype=np.int64)
    see_raw(*state, 12, 28, -1, True, piece_values)
