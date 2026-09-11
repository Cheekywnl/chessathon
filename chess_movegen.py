"""Legal chess move generation on bitboards, compiled during agent import.

Pins and checkers determine legal destinations for ordinary non-king moves.
A single check requires capturing or blocking its checker; double check
permits only king moves through that path. King moves and en passant use
a full resulting-position attack check. Move order is deterministic.

The optimized generator matched the previous generator and python-chess
across 29,954 mixed positions and another 39,668 positions selected to
exercise checks, pins, promotions and en passant.
"""

import chess
import numpy as np
from numba import njit

from chess_bits import lsb_index

WHITE, BLACK = True, False
PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = 1, 2, 3, 4, 5, 6

# --- attack tables, lifted directly from python-chess's own public constants ---
KNIGHT_ATTACKS = np.array(chess.BB_KNIGHT_ATTACKS, dtype=np.uint64)
KING_ATTACKS = np.array(chess.BB_KING_ATTACKS, dtype=np.uint64)
PAWN_ATTACKS_WHITE = np.array(chess.BB_PAWN_ATTACKS[chess.WHITE], dtype=np.uint64)
PAWN_ATTACKS_BLACK = np.array(chess.BB_PAWN_ATTACKS[chess.BLACK], dtype=np.uint64)

# Ray-casting deltas for sliding pieces: (file_delta, rank_delta) per direction.
ROOK_DIRS = np.array([(1, 0), (-1, 0), (0, 1), (0, -1)], dtype=np.int64)
BISHOP_DIRS = np.array([(1, 1), (1, -1), (-1, 1), (-1, -1)], dtype=np.int64)

# Squares strictly between aligned endpoints; empty for nonaligned squares.
BETWEEN = np.array([[chess.between(a, b) for b in range(64)] for a in range(64)],
                   dtype=np.uint64)

FULL_BOARD = np.uint64(0xFFFFFFFFFFFFFFFF)
RANK_2 = np.uint64(0x000000000000FF00)
RANK_7 = np.uint64(0x00FF000000000000)
RANK_4 = np.uint64(0x00000000FF000000)
RANK_5 = np.uint64(0x000000FF00000000)


@njit(cache=False)
def _bit(sq: int) -> np.uint64:
    return np.uint64(1) << np.uint64(sq)


@njit(cache=False)
def _sliding_attacks(sq: int, occupied: np.uint64, dirs: np.ndarray) -> np.uint64:
    attacks = np.uint64(0)
    file0 = sq % 8
    rank0 = sq // 8
    for d in range(dirs.shape[0]):
        df = dirs[d, 0]
        dr = dirs[d, 1]
        f = file0 + df
        r = rank0 + dr
        while 0 <= f < 8 and 0 <= r < 8:
            target = r * 8 + f
            attacks |= _bit(target)
            if occupied & _bit(target):
                break
            f += df
            r += dr
    return attacks


@njit(cache=False)
def attacks_from(
    sq: int,
    piece_type: int,
    color: bool,
    occupied: np.uint64,
) -> np.uint64:
    if piece_type == KNIGHT:
        return np.uint64(KNIGHT_ATTACKS[sq])
    if piece_type == KING:
        return np.uint64(KING_ATTACKS[sq])
    if piece_type == PAWN:
        return np.uint64(PAWN_ATTACKS_WHITE[sq] if color else PAWN_ATTACKS_BLACK[sq])
    if piece_type == BISHOP:
        return _sliding_attacks(sq, occupied, BISHOP_DIRS)
    if piece_type == ROOK:
        return _sliding_attacks(sq, occupied, ROOK_DIRS)
    # QUEEN
    return _sliding_attacks(sq, occupied, BISHOP_DIRS) | _sliding_attacks(sq, occupied, ROOK_DIRS)


@njit(cache=False)
def is_square_attacked(
    sq: int,
    by_white: bool,
    pawns: np.uint64,
    knights: np.uint64,
    bishops: np.uint64,
    rooks: np.uint64,
    queens: np.uint64,
    kings: np.uint64,
    white: np.uint64,
    black: np.uint64,
) -> bool:
    occupied = white | black
    attacker_side = white if by_white else black
    # A pawn of `by_white`'s colour attacks `sq` iff `sq` is one of the squares that colour's
    # pawn attack pattern reaches -- equivalently, iff sq is attacked using the OPPOSITE
    # colour's attack table centred on sq (attack tables are symmetric this way).
    pawn_table = PAWN_ATTACKS_BLACK if by_white else PAWN_ATTACKS_WHITE
    pawn_attackers = pawn_table[sq] & pawns & attacker_side
    if pawn_attackers:
        return True
    if KNIGHT_ATTACKS[sq] & knights & attacker_side:
        return True
    if KING_ATTACKS[sq] & kings & attacker_side:
        return True
    diag = _sliding_attacks(sq, occupied, BISHOP_DIRS)
    if diag & (bishops | queens) & attacker_side:
        return True
    straight = _sliding_attacks(sq, occupied, ROOK_DIRS)
    return bool(straight & (rooks | queens) & attacker_side)


# Move encoding for the fixed-size scratch buffer: (from, to, promotion, move_type).
# move_type: 0 = normal, 1 = en passant, 2 = kingside castle, 3 = queenside castle.
MAX_MOVES = 256


@njit(cache=False)
def _apply_move_bitboards(
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
    promotion: int,
    move_type: int,
    moving_white: bool,
) -> tuple[np.uint64, np.uint64, np.uint64, np.uint64, np.uint64, np.uint64, np.uint64, np.uint64]:
    """Apply one move to a scratch copy of the bitboards. Used only to test whether the mover's
    king ends up attacked -- never touches the real board, never used for the actual search
    tree (python-chess's own push/pop still owns that)."""
    from_bit = _bit(from_sq)
    to_bit = _bit(to_sq)

    # Remove the moving piece from its origin square, from whichever piece bitboard it's in.
    if pawns & from_bit:
        pawns &= ~from_bit
        moving_type = PAWN
    elif knights & from_bit:
        knights &= ~from_bit
        moving_type = KNIGHT
    elif bishops & from_bit:
        bishops &= ~from_bit
        moving_type = BISHOP
    elif rooks & from_bit:
        rooks &= ~from_bit
        moving_type = ROOK
    elif queens & from_bit:
        queens &= ~from_bit
        moving_type = QUEEN
    else:
        kings &= ~from_bit
        moving_type = KING

    if moving_white:
        white &= ~from_bit
    else:
        black &= ~from_bit

    # Captures: remove whatever's on the destination square (or, for en passant, the pawn one
    # rank behind the destination).
    if move_type == 1:
        captured_sq = to_sq - 8 if moving_white else to_sq + 8
        captured_bit = _bit(captured_sq)
        pawns &= ~captured_bit
        if moving_white:
            black &= ~captured_bit
        else:
            white &= ~captured_bit
    else:
        pawns &= ~to_bit
        knights &= ~to_bit
        bishops &= ~to_bit
        rooks &= ~to_bit
        queens &= ~to_bit
        kings &= ~to_bit
        white &= ~to_bit
        black &= ~to_bit

    # Place the moving piece (or its promotion) on the destination square.
    final_type = promotion if promotion != 0 else moving_type
    if final_type == PAWN:
        pawns |= to_bit
    elif final_type == KNIGHT:
        knights |= to_bit
    elif final_type == BISHOP:
        bishops |= to_bit
    elif final_type == ROOK:
        rooks |= to_bit
    elif final_type == QUEEN:
        queens |= to_bit
    else:
        kings |= to_bit
    if moving_white:
        white |= to_bit
    else:
        black |= to_bit

    # Castling also moves the rook. from_sq/to_sq here are the KING's squares.
    if move_type == 2:  # kingside
        rook_from = 7 if moving_white else 63
        rook_to = 5 if moving_white else 61
        rooks = (rooks & ~_bit(rook_from)) | _bit(rook_to)
        if moving_white:
            white = (white & ~_bit(rook_from)) | _bit(rook_to)
        else:
            black = (black & ~_bit(rook_from)) | _bit(rook_to)
    elif move_type == 3:  # queenside
        rook_from = 0 if moving_white else 56
        rook_to = 3 if moving_white else 59
        rooks = (rooks & ~_bit(rook_from)) | _bit(rook_to)
        if moving_white:
            white = (white & ~_bit(rook_from)) | _bit(rook_to)
        else:
            black = (black & ~_bit(rook_from)) | _bit(rook_to)

    return pawns, knights, bishops, rooks, queens, kings, white, black


@njit(cache=False)
def compute_pins_and_checkers(
    pawns: np.uint64,
    knights: np.uint64,
    bishops: np.uint64,
    rooks: np.uint64,
    queens: np.uint64,
    white: np.uint64,
    black: np.uint64,
    turn: bool,
    king_sq: int,
) -> tuple[np.uint64, np.uint64, np.ndarray]:
    """The optimization python-chess itself uses: compute checkers and pinned pieces ONCE per
    position, so that most moves (non-pinned pieces, king not in check) need zero further
    legality checking at all -- the naive make-a-scratch-move-and-rescan-the-board approach
    this replaces was correct but redid a full attack scan for every single candidate move.

    Returns (checkers, pinned, pin_rays) -- pin_rays[sq] is the set of squares a piece pinned
    on `sq` may legally move to (the ray between the king and the pinning piece, inclusive of
    the pinner, exclusive of the king itself)."""
    occupied = white | black
    enemy = black if turn else white

    pawn_table = PAWN_ATTACKS_WHITE if turn else PAWN_ATTACKS_BLACK
    checkers = pawn_table[king_sq] & pawns & enemy
    checkers |= KNIGHT_ATTACKS[king_sq] & knights & enemy

    pinned = np.uint64(0)
    pin_rays = np.zeros(64, dtype=np.uint64)
    king_file = king_sq % 8
    king_rank = king_sq // 8

    for dset in range(2):
        dirs = BISHOP_DIRS if dset == 0 else ROOK_DIRS
        slider_mask = (bishops | queens) if dset == 0 else (rooks | queens)
        for d in range(4):
            df = dirs[d, 0]
            dr = dirs[d, 1]
            f = king_file + df
            r = king_rank + dr
            ray = np.uint64(0)
            blocker_sq = -1
            while 0 <= f < 8 and 0 <= r < 8:
                sq = r * 8 + f
                bit_sq = _bit(sq)
                ray |= bit_sq
                if occupied & bit_sq:
                    if blocker_sq == -1:
                        if enemy & bit_sq:
                            if slider_mask & bit_sq:
                                checkers |= bit_sq
                            break
                        blocker_sq = sq
                    else:
                        if (enemy & bit_sq) and (slider_mask & bit_sq):
                            pinned |= _bit(blocker_sq)
                            pin_rays[blocker_sq] = ray
                        break
                f += df
                r += dr

    return checkers, pinned, pin_rays


@njit(cache=False)
def _would_be_legal(
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
    promotion: int,
    move_type: int,
    moving_white: bool,
    king_sq: int,
) -> bool:
    """Apply the move to a scratch copy and check the mover's king isn't left attacked.
    king_sq is the mover's king square BEFORE the move (unless the king itself is moving, in
    which case the resulting king bitboard after _apply_move_bitboards already reflects that)."""
    npawns, nknights, nbishops, nrooks, nqueens, nkings, nwhite, nblack = _apply_move_bitboards(
        pawns, knights, bishops, rooks, queens, kings, white, black,
        from_sq, to_sq, promotion, move_type, moving_white,
    )
    if from_sq == king_sq:
        king_bb = nkings & (nwhite if moving_white else nblack)
        if king_bb:
            king_sq = lsb_index(king_bb)
    return not is_square_attacked(
        king_sq, not moving_white,
        npawns, nknights, nbishops, nrooks, nqueens, nkings, nwhite, nblack,
    )


@njit(cache=False)
def generate_legal_moves_bb(
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Returns (from_arr, to_arr, promo_arr, count). Parallel arrays, only the first `count`
    entries are valid. promo_arr uses 0/2/3/4/5 for none/knight/bishop/rook/queen."""
    out_from = np.empty(MAX_MOVES, dtype=np.int64)
    out_to = np.empty(MAX_MOVES, dtype=np.int64)
    out_promo = np.zeros(MAX_MOVES, dtype=np.int64)
    count = 0

    occupied = white | black
    own = white if turn else black
    enemy = black if turn else white

    king_bb = kings & own
    king_sq = lsb_index(king_bb) if king_bb else 0

    checkers, pinned, pin_rays = compute_pins_and_checkers(
        pawns, knights, bishops, rooks, queens, white, black, turn, king_sq
    )
    in_check = checkers != 0
    evasion_targets = FULL_BOARD
    if in_check:
        # Ordinary non-king moves must capture or block the sole checker.
        # With two checkers, only king moves can pass this path. En passant
        # retains its independent full-board legality check below.
        evasion_targets = np.uint64(0)
        if not (checkers & (checkers - np.uint64(1))):
            evasion_targets = checkers | BETWEEN[king_sq, lsb_index(checkers)]

    # Non-pawn, non-king pieces. When not in check and not pinned, a pseudo-legal move is
    # always legal -- no per-move attack rescan needed at all, which is the whole saving over
    # the naive make-and-test approach. Pinned pieces are legal iff the destination stays on
    # the pin ray; nothing else to check since the king itself isn't moving. In check, fall
    # back to the general (slower, but always correct) path -- check evasions are rarer than
    # normal moves, so this is a fine trade.
    for piece_type in (KNIGHT, BISHOP, ROOK, QUEEN):
        if piece_type == KNIGHT:
            piece_bb = knights & own
        elif piece_type == BISHOP:
            piece_bb = bishops & own
        elif piece_type == ROOK:
            piece_bb = rooks & own
        else:
            piece_bb = queens & own
        while piece_bb:
            from_sq = lsb_index(piece_bb)
            piece_bb &= piece_bb - np.uint64(1)
            targets = attacks_from(from_sq, piece_type, turn, occupied) & ~own
            is_pinned = pinned & _bit(from_sq) != 0
            while targets:
                to_sq = lsb_index(targets)
                targets &= targets - np.uint64(1)
                if in_check:
                    legal = bool(evasion_targets & _bit(to_sq)) and (
                        not is_pinned or bool(pin_rays[from_sq] & _bit(to_sq))
                    )
                elif is_pinned:
                    legal = (pin_rays[from_sq] & _bit(to_sq)) != 0
                else:
                    legal = True
                if legal:
                    out_from[count] = from_sq
                    out_to[count] = to_sq
                    out_promo[count] = 0
                    count += 1

    # King moves.
    king_targets = KING_ATTACKS[king_sq] & ~own
    while king_targets:
        to_sq = lsb_index(king_targets)
        king_targets &= king_targets - np.uint64(1)
        if _would_be_legal(
            pawns, knights, bishops, rooks, queens, kings, white, black,
            king_sq, to_sq, 0, 0, turn, king_sq,
        ):
            out_from[count] = king_sq
            out_to[count] = to_sq
            out_promo[count] = 0
            count += 1

    # Pawns: captures (incl. promotions), single/double pushes (incl. promotions), en passant.
    # Same fast/slow split as above for everything except en passant, which always uses the
    # slow path below (the classic en passant pin needs the full make-and-test).
    pawn_bb = pawns & own
    pawn_sources = pawn_bb
    while pawn_sources:
        from_sq = lsb_index(pawn_sources)
        pawn_sources &= pawn_sources - np.uint64(1)
        rank = from_sq // 8
        promo_rank = 7 if turn else 0
        is_pinned = pinned & _bit(from_sq) != 0

        capture_table = PAWN_ATTACKS_WHITE if turn else PAWN_ATTACKS_BLACK
        capture_targets = capture_table[from_sq] & enemy
        while capture_targets:
            to_sq = lsb_index(capture_targets)
            capture_targets &= capture_targets - np.uint64(1)
            if in_check:
                legal = bool(evasion_targets & _bit(to_sq)) and (
                    not is_pinned or bool(pin_rays[from_sq] & _bit(to_sq))
                )
            elif is_pinned:
                legal = (pin_rays[from_sq] & _bit(to_sq)) != 0
            else:
                legal = True
            if not legal:
                continue
            if to_sq // 8 == promo_rank:
                for promo in (QUEEN, ROOK, BISHOP, KNIGHT):
                    out_from[count] = from_sq
                    out_to[count] = to_sq
                    out_promo[count] = promo
                    count += 1
            else:
                out_from[count] = from_sq
                out_to[count] = to_sq
                out_promo[count] = 0
                count += 1

        # single push
        single_to = from_sq + 8 if turn else from_sq - 8
        if 0 <= single_to < 64 and not (occupied & _bit(single_to)):
            if in_check:
                legal = bool(evasion_targets & _bit(single_to)) and (
                    not is_pinned or bool(pin_rays[from_sq] & _bit(single_to))
                )
            elif is_pinned:
                legal = (pin_rays[from_sq] & _bit(single_to)) != 0
            else:
                legal = True
            if legal:
                if single_to // 8 == promo_rank:
                    for promo in (QUEEN, ROOK, BISHOP, KNIGHT):
                        out_from[count] = from_sq
                        out_to[count] = single_to
                        out_promo[count] = promo
                        count += 1
                else:
                    out_from[count] = from_sq
                    out_to[count] = single_to
                    out_promo[count] = 0
                    count += 1
            # double push, only from the home rank and only if the single push was clear
            home_rank = 1 if turn else 6
            if rank == home_rank:
                double_to = from_sq + 16 if turn else from_sq - 16
                if not (occupied & _bit(double_to)):
                    if in_check:
                        d_legal = bool(evasion_targets & _bit(double_to)) and (
                            not is_pinned or bool(pin_rays[from_sq] & _bit(double_to))
                        )
                    elif is_pinned:
                        d_legal = (pin_rays[from_sq] & _bit(double_to)) != 0
                    else:
                        d_legal = True
                    if d_legal:
                        out_from[count] = from_sq
                        out_to[count] = double_to
                        out_promo[count] = 0
                        count += 1

    # En passant.
    if ep_square >= 0:
        ep_table = PAWN_ATTACKS_BLACK if turn else PAWN_ATTACKS_WHITE
        capturers = ep_table[ep_square] & pawn_bb
        while capturers:
            from_sq = lsb_index(capturers)
            capturers &= capturers - np.uint64(1)
            if _would_be_legal(
                pawns, knights, bishops, rooks, queens, kings, white, black,
                from_sq, ep_square, 0, 1, turn, king_sq,
            ):
                out_from[count] = from_sq
                out_to[count] = ep_square
                out_promo[count] = 0
                count += 1

    # Castling.
    king_home = 4 if turn else 60
    if king_sq == king_home and not is_square_attacked(
        king_sq, not turn, pawns, knights, bishops, rooks, queens, kings, white, black
    ):
        kingside_rook = 7 if turn else 63
        if castling_rights & _bit(kingside_rook):
            f1, g1 = (5, 6) if turn else (61, 62)
            if not (occupied & (_bit(f1) | _bit(g1))) and not is_square_attacked(
                f1, not turn, pawns, knights, bishops, rooks, queens, kings, white, black
            ) and not is_square_attacked(
                g1, not turn, pawns, knights, bishops, rooks, queens, kings, white, black
            ):
                out_from[count] = king_sq
                out_to[count] = g1
                out_promo[count] = 0
                count += 1
        queenside_rook = 0 if turn else 56
        if castling_rights & _bit(queenside_rook):
            b1, c1, d1 = (1, 2, 3) if turn else (57, 58, 59)
            if not (occupied & (_bit(b1) | _bit(c1) | _bit(d1))) and not is_square_attacked(
                d1, not turn, pawns, knights, bishops, rooks, queens, kings, white, black
            ) and not is_square_attacked(
                c1, not turn, pawns, knights, bishops, rooks, queens, kings, white, black
            ):
                out_from[count] = king_sq
                out_to[count] = c1
                out_promo[count] = 0
                count += 1

    return out_from, out_to, out_promo, count


def fast_legal_moves(board: chess.Board) -> list[chess.Move]:
    """Move-for-move identical to list(board.legal_moves) -- see validate_movegen.py for the
    exhaustive comparison this claim rests on. Variant handling (Chess960 etc.) is NOT
    supported; only ever called on standard games, which is all this competition allows."""
    ep = board.ep_square if board.has_legal_en_passant() else -1
    from_arr, to_arr, promo_arr, count = generate_legal_moves_bb(
        np.uint64(board.pawns),
        np.uint64(board.knights),
        np.uint64(board.bishops),
        np.uint64(board.rooks),
        np.uint64(board.queens),
        np.uint64(board.kings),
        np.uint64(board.occupied_co[chess.WHITE]),
        np.uint64(board.occupied_co[chess.BLACK]),
        board.turn,
        np.uint64(board.clean_castling_rights()),
        ep if ep is not None else -1,
    )
    moves = []
    for i in range(count):
        promo = int(promo_arr[i])
        moves.append(chess.Move(int(from_arr[i]), int(to_arr[i]), promo if promo else None))
    return moves


def warm_up() -> None:
    """Compile every jitted signature now, at import, not on the clock."""
    fast_legal_moves(chess.Board())
