"""Negamax alpha-beta search: iterative deepening, a persistent transposition table, quiescence,
null-move pruning, reverse and regular futility pruning, late move reductions, and check
extensions.

The search tree operates entirely on raw bitboard state (pawns, knights, ..., black, turn,
castling_rights, ep_square, halfmove_clock) and packed-int moves (chess_state.pack_move) --
never a chess.Board, never a chess.Move -- for every recursive call. A chess.Board and
chess.Move are only ever constructed once each, at the very top and bottom of `search_root`,
to accept the caller's board and hand back a real move. This removes the board.push()/pop() and
Move-object-construction overhead that persisted through the first move-generation rewrite (see
chess_movegen.py's docstring): that rewrite only replaced move generation, and generation was
never the majority of the remaining cost once it got fast -- constructing chess.Move objects and
python-chess's own push()/pop() bookkeeping, happening on *every* node regardless of how moves
were generated, was.

State is immutable by convention (chess_state.make_move returns a complete new state rather than
mutating in place), so recursion unmakes a move simply by not propagating the child's state back
up -- there is no explicit unmake, which removes an entire class of bugs (some field not restored
on the way back out) at the cost of copying a dozen scalars per ply, which is free next to what
push()/pop() cost.

The transposition table and the real-game position history are owned by the caller and passed
in, because they are the two things worth keeping alive across moves in the same game (see
AGENTS.md). Everything else here -- killers, history heuristic, node count -- is scoped to one
`search_root` call and thrown away.

A timeout unwinds through this file as a `TimeUp` exception raised deep in the recursion.
"""

import math
import sys
import threading
import time
from pathlib import Path

import chess
import numpy as np

import chess_eval as ce
import chess_halfkp_int as halfkp
import chess_movegen as mg
import chess_state as cst

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

# Candidate blend, promoted only with a validated weight asset and real A/B evidence.
# An absent or malformed asset retains the classical evaluator exactly.
HALFKP_BLEND = 75
HALFKP_MIN_PIECES = 8
HALFKP_WEIGHTS: halfkp.QuantizedWeights | None = None
_halfkp_path = Path(__file__).resolve().parent / "weights" / "halfkp.npz"
if _halfkp_path.is_file():
    try:
        HALFKP_WEIGHTS = halfkp.load_weights(_halfkp_path)
        halfkp.warm_up(HALFKP_WEIGHTS)
    except Exception as _halfkp_error:
        HALFKP_WEIGHTS = None
        print(f"HalfKP load failed; using classical evaluation: {_halfkp_error}", file=sys.stderr)

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

# One flat tuple describes the game state everywhere in this file:
# (pawns, knights, bishops, rooks, queens, kings, white, black,
#  turn, castling_rights, ep_square, halfmove_clock)
State = tuple[
    np.uint64, np.uint64, np.uint64, np.uint64, np.uint64, np.uint64,
    np.uint64, np.uint64, bool, np.uint64, int, int,
]

TTEntry = tuple[int, int, int, int, int]  # key, depth, score, flag, move


class TimeUp(Exception):
    pass


def _score_to_tt(score: int, ply: int) -> int:
    if score >= MATE_THRESHOLD:
        return score + ply
    if score <= -MATE_THRESHOLD:
        return score - ply
    return score


def _score_from_tt(score: int, ply: int) -> int:
    if score >= MATE_THRESHOLD:
        return score - ply
    if score <= -MATE_THRESHOLD:
        return score + ply
    return score


class TranspositionTable:
    """A fixed-size, always-there hash table: no dict growth, so memory is bounded by
    construction. Keyed directly by the 64-bit Zobrist hash (chess_state.hash_state) -- the
    same industry-standard trade-off every real engine makes: a hash collision could in
    principle cause a false TT hit, but at 2^64 possible values it is not a practical risk
    within one game's search, and exact-key comparison (as the pre-rewrite table did, storing
    python-chess's own hashable position tuple) is no longer available now that the search
    never touches a chess.Board."""

    def __init__(self, size_power: int = 21) -> None:
        self.mask = (1 << size_power) - 1
        self.table: list[TTEntry | None] = [None] * (1 << size_power)

    def probe(
        self, key: int, depth: int, alpha: int, beta: int, ply: int
    ) -> tuple[int | None, int]:
        entry = self.table[key & self.mask]
        if entry is None or entry[0] != key:
            return None, NO_MOVE
        _, e_depth, e_score, e_flag, e_move = entry
        if e_depth >= depth:
            score = _score_from_tt(e_score, ply)
            if e_flag == FLAG_EXACT:
                return score, e_move
            if e_flag == FLAG_LOWER and score >= beta:
                return score, e_move
            if e_flag == FLAG_UPPER and score <= alpha:
                return score, e_move
        return None, e_move

    def store(self, key: int, depth: int, score: int, flag: int, move: int, ply: int) -> None:
        index = key & self.mask
        existing = self.table[index]
        if existing is None or existing[0] == key or existing[1] <= depth:
            self.table[index] = (key, depth, _score_to_tt(score, ply), flag, move)


def state_from_board(board: chess.Board) -> State:
    ep_square = board.ep_square
    ep = ep_square if (ep_square is not None and board.has_legal_en_passant()) else -1
    return (
        np.uint64(board.pawns), np.uint64(board.knights), np.uint64(board.bishops),
        np.uint64(board.rooks), np.uint64(board.queens), np.uint64(board.kings),
        np.uint64(board.occupied_co[chess.WHITE]), np.uint64(board.occupied_co[chess.BLACK]),
        board.turn, np.uint64(board.clean_castling_rights()), ep, board.halfmove_clock,
    )


def hash_of(state: State) -> int:
    # Calls the jitted zobrist_hash directly rather than going through chess_state.hash_state:
    # State's own fields are already np.uint64 (see chess_state.py's own docstring on this),
    # exactly what zobrist_hash wants, so hash_state's int()/np.uint64() round trip on every one
    # of them was pure conversion tax paid on the single hottest call in the whole search (one
    # per node, plus one per move actually searched) for no behavioural difference -- confirmed
    # by profiling before touching it: hash_state alone was ~18% of total search time on a
    # representative position. hash_state itself is untouched and still used by its own
    # warm_up() call, which is not hot enough for this to matter there.
    return int(
        cst.zobrist_hash(
            state[0], state[1], state[2], state[3], state[4], state[5], state[6], state[7],
            state[8], state[9], state[10],
            cst.ZOBRIST_PIECE_SQUARE, cst.ZOBRIST_CASTLING, cst.ZOBRIST_EP_FILE, cst.ZOBRIST_TURN,
        )
    )


def king_square(state: State, white: bool) -> int:
    own = int(state[6]) if white else int(state[7])
    king_bb = int(state[5]) & own
    return (king_bb & -king_bb).bit_length() - 1


def is_in_check(state: State) -> bool:
    turn = state[8]
    sq = king_square(state, turn)
    return bool(
        mg.is_square_attacked(
            sq, not turn, state[0], state[1], state[2], state[3], state[4], state[5],
            state[6], state[7],
        )
    )


def legal_moves(state: State) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    return mg.generate_legal_moves_bb(
        state[0], state[1], state[2], state[3], state[4], state[5], state[6], state[7],
        state[8], state[9], state[10],
    )


def apply_move(state: State, from_sq: int, to_sq: int, promotion: int) -> State:
    # numba returns plain Python ints (not np.uint64) across a nopython -> Python call boundary;
    # for values past int64's range that makes the type of a bare re-passed int ambiguous at the
    # *next* jitted call (observed in practice as numba inferring float64 and rejecting the bitwise
    # op). Re-wrapping here, once, keeps every State tuple genuinely np.uint64-typed everywhere
    # else in this module.
    result = cst.make_move(
        state[0], state[1], state[2], state[3], state[4], state[5], state[6], state[7],
        state[8], state[9], state[10], state[11], from_sq, to_sq, promotion,
    )
    return (
        np.uint64(result[0]), np.uint64(result[1]), np.uint64(result[2]), np.uint64(result[3]),
        np.uint64(result[4]), np.uint64(result[5]), np.uint64(result[6]), np.uint64(result[7]),
        bool(result[8]), np.uint64(result[9]), int(result[10]), int(result[11]),
    )


def apply_move_info(
    state: State, key: int, from_sq: int, to_sq: int, promotion: int,
) -> tuple[State, int, bool]:
    result, child_key, check = cst.make_move_info(
        state, np.uint64(key), from_sq, to_sq, promotion,
    )
    child: State = (
        np.uint64(result[0]), np.uint64(result[1]), np.uint64(result[2]), np.uint64(result[3]),
        np.uint64(result[4]), np.uint64(result[5]), np.uint64(result[6]), np.uint64(result[7]),
        bool(result[8]), np.uint64(result[9]), int(result[10]), int(result[11]),
    )
    return child, int(child_key), bool(check)


def apply_null(state: State) -> State:
    return (*state[:8], not state[8], state[9], -1, state[11] + 1)


def int_fields(state: State) -> tuple[int, int, int, int, int, int, int]:
    """Plain Python ints for the piece bitboards, extracted once per node. The per-move helpers
    below (is_capture_i, captured_type_i, moving_type_i) are called many times per node during
    move ordering and the search loop -- numpy uint64 scalar arithmetic has real per-operation
    overhead next to native Python int ops at this call frequency, measured directly as the
    single largest cost in this file before this split (profiling note, not a guess)."""
    return (
        int(state[0]), int(state[1]), int(state[2]), int(state[3]), int(state[4]),
        int(state[6]), int(state[7]),
    )


def is_capture_i(
    white: int, black: int, pawns: int, ep_square: int, from_sq: int, to_sq: int
) -> bool:
    if (white | black) & (1 << to_sq):
        return True
    return to_sq == ep_square and ep_square >= 0 and (pawns & (1 << from_sq)) != 0


def captured_type_i(
    pawns: int, knights: int, bishops: int, rooks: int, queens: int,
    ep_square: int, from_sq: int, to_sq: int,
) -> int:
    if to_sq == ep_square and ep_square >= 0 and (pawns & (1 << from_sq)):
        return PAWN
    bit = 1 << to_sq
    if pawns & bit:
        return PAWN
    if knights & bit:
        return KNIGHT
    if bishops & bit:
        return BISHOP
    if rooks & bit:
        return ROOK
    if queens & bit:
        return QUEEN
    return PAWN


def moving_type_i(
    pawns: int, knights: int, bishops: int, rooks: int, queens: int, from_sq: int
) -> int:
    bit = 1 << from_sq
    if pawns & bit:
        return PAWN
    if knights & bit:
        return KNIGHT
    if bishops & bit:
        return BISHOP
    if rooks & bit:
        return ROOK
    if queens & bit:
        return QUEEN
    return KING


def has_non_pawn_material(state: State) -> bool:
    own = state[6] if state[8] else state[7]
    return bool(own & ~state[0] & ~state[5])


def is_bare_king_endgame(state: State) -> bool:
    """True in exactly the situations chess_eval's mop-up term fires (kept in sync with it
    deliberately -- see its comment for why only a rook/queen counts as mating material, not
    a pawn or minor piece): one side has a rook or queen and the other has nothing but a king.
    This is the scenario a real regression showed late move reductions and futility pruning can
    break -- converting mate depends on a specific sequence of quiet king/rook moves that a
    reduction or an outright skip can hide, precisely when there's no material left to make
    captures or checks look tactically interesting instead."""
    pawns, knights, bishops, rooks, queens = state[0], state[1], state[2], state[3], state[4]
    white, black = state[6], state[7]
    major = rooks | queens
    losing_side_bare = pawns | knights | bishops | major
    white_is_bare = (losing_side_bare & white) == 0
    black_is_bare = (losing_side_bare & black) == 0
    return bool((major & white) != 0 and black_is_bare) or bool(
        (major & black) != 0 and white_is_bare
    )


def see(state: State, from_sq: int, to_sq: int) -> int:
    return cst.see_raw(
        state[0], state[1], state[2], state[3], state[4], state[5], state[6], state[7],
        from_sq, to_sq, state[10], state[8], PIECE_VALUES,
    )


def _castling_rights_counts(state: State) -> tuple[int, int]:
    castling_rights = int(state[9])
    white_count = (1 if castling_rights & 1 else 0) + (1 if castling_rights & 128 else 0)
    black_count = (1 if castling_rights & (1 << 56) else 0) + (
        1 if castling_rights & (1 << 63) else 0
    )
    return white_count, black_count


def hash_of_board(board: chess.Board) -> int:
    """For callers (agent.py) that still key their own persistent dicts -- the transposition
    table and the real-game history -- by position, without going through the search."""
    return hash_of(state_from_board(board))


def packed_to_move(packed: int) -> chess.Move | None:
    if packed == NO_MOVE:
        return None
    f, t, p = cst.unpack_move(packed)
    return chess.Move(f, t, p if p else None)


_DARK_SQUARES = 0xAA55AA55AA55AA55
_LIGHT_SQUARES = 0x55AA55AA55AA55AA


def _has_insufficient_material(
    own: int, other: int, pawns: int, knights: int, bishops: int, rooks: int, queens: int,
    kings: int,
) -> bool:
    """Direct bitboard port of chess.Board.has_insufficient_material, same rule, no allocation.
    `own`/`other` are that color's/the opponent's occupancy; every other argument is the
    all-pieces-of-that-type bitboard (both colors), exactly as python-chess reads them off
    self.pawns / self.bishops / etc. -- the bishop same-color and pawn/knight checks below are
    deliberately board-wide, not `own`-masked, matching the source rule."""
    if own & (pawns | rooks | queens):
        return False
    if own & knights:
        return own.bit_count() <= 2 and not (other & ~kings & ~queens)
    if own & bishops:
        same_color = not (bishops & _DARK_SQUARES) or not (bishops & _LIGHT_SQUARES)
        return same_color and not pawns and not knights
    return True


def insufficient_material(state: State) -> bool:
    """Only ever reached with <= 6 pieces on the board, but far from rare there -- profiling
    showed this was ~23% of total search time by way of reconstructing a throwaway chess.Board
    per call just to reuse its is_insufficient_material(). A direct bitboard port of that same
    rule (see _has_insufficient_material) needs no allocation and no chess.Board at all."""
    pawns, knights, bishops = int(state[0]), int(state[1]), int(state[2])
    rooks, queens, kings = int(state[3]), int(state[4]), int(state[5])
    white, black = int(state[6]), int(state[7])
    return (
        _has_insufficient_material(white, black, pawns, knights, bishops, rooks, queens, kings)
        and _has_insufficient_material(black, white, pawns, knights, bishops, rooks, queens, kings)
    )


class Search:
    def __init__(
        self,
        tt: TranspositionTable,
        game_history: dict[int, int],
        params: np.ndarray | None = None,
        stop_event: threading.Event | None = None,
    ) -> None:
        self.tt = tt
        self.params = params if params is not None else ce.DEFAULT_PARAMS
        self.seen: dict[int, int] = dict(game_history)
        self.killers: list[list[int]] = [[NO_MOVE, NO_MOVE] for _ in range(MAX_PLY)]
        self.history: dict[tuple[bool, int, int], int] = {}
        self.nodes = 0
        self.deadline = 0.0
        self.stop_event = stop_event
        # 1: the referee ends the game after this root move; 2: the opponent
        # can choose an immediate draw. Values agree with chess_draw constants.
        self.root_draw_claims: dict[int, int] = {}
        self.root_repeated_moves: set[int] = set()

    def classical_evaluate(self, state: State, mobility: int) -> int:
        white_count, black_count = _castling_rights_counts(state)
        positional = ce.evaluate(
            state[0], state[1], state[2], state[3], state[4], state[5], state[6], state[7],
            ce.MG_PST, ce.EG_PST, ce.PASSED_MASK_WHITE, ce.PASSED_MASK_BLACK,
            ce.FILE_MASK, ce.ADJACENT_FILE_MASK, white_count, black_count, self.params,
        )
        mover_relative = int(positional) if state[8] else -int(positional)
        return mover_relative + int(self.params[ce.P_MOBILITY]) * mobility

    def evaluate(self, state: State, mobility: int) -> int:
        weights = HALFKP_WEIGHTS
        if weights is None or HALFKP_BLEND <= 0:
            return self.classical_evaluate(state, mobility)
        occupied = int(state[6] | state[7])
        non_kings = int(state[0] | state[1] | state[2] | state[3] | state[4])
        if (occupied.bit_count() < HALFKP_MIN_PIECES
                or not (non_kings & int(state[6])) or not (non_kings & int(state[7]))):
            # Preserve all established low-material and bare-king technique exactly.
            return self.classical_evaluate(state, mobility)
        neural = halfkp.evaluate(
            state[0], state[1], state[2], state[3], state[4], state[5], state[6], state[7],
            state[8], weights.w1, weights.b1, weights.w2, weights.b2, weights.w3, weights.b3,
            weights.w4, weights.b4, weights.scale2, weights.scale3, weights.output_divisor,
        )
        bonus = int(ce.mop_up_bonus(*state[:8], self.params))
        neural = max(-20_000, min(20_000, neural)) + (bonus if state[8] else -bonus)
        if HALFKP_BLEND >= 100:
            return neural
        classical = self.classical_evaluate(state, mobility)
        return round((classical * (100 - HALFKP_BLEND) + neural * HALFKP_BLEND) / 100)

    def _time_check(self) -> None:
        self.nodes += 1
        if self.nodes % NODES_PER_TIME_CHECK == 0:
            if self.stop_event is not None and self.stop_event.is_set():
                raise TimeUp
            if time.monotonic() > self.deadline:
                raise TimeUp

    def _is_draw(self, state: State, key: int) -> bool:
        if state[11] >= 100:
            return True
        if self.seen.get(key, 0) >= 3:
            return True
        occupied = state[6] | state[7]
        # int.bit_count() (3.10+, the platform runs 3.12) does exactly what
        # bin(x).count("1") does -- count set bits -- natively rather than via a string
        # round-trip; same result, real cost difference at this call frequency (once per node).
        if int(occupied).bit_count() <= 6:
            return insufficient_material(state)
        return False

    @staticmethod
    def _draw_score(ply: int) -> int:
        """Mover-relative contempt score for a drawn position at this ply. search_root is
        always called with our own position to move (ply 0), so parity alone says whose move
        it is at any leaf -- even ply means it's our move here, odd means the opponent's --
        without needing to track an explicit colour through the whole recursive tree. Verified
        by construction, not just asserted: with an even number of plies between root and leaf,
        negamax's repeated negation nets to zero flips, so "bad for whoever moves at an even
        ply" is exactly "bad for us" once fully unwound back to the root, regardless of depth."""
        return -CONTEMPT if ply % 2 == 0 else CONTEMPT

    def _order_moves(
        self,
        state: State,
        from_arr: np.ndarray,
        to_arr: np.ndarray,
        promo_arr: np.ndarray,
        count: int,
        tt_move: int,
        ply: int,
    ) -> list[tuple[int, bool]]:
        """Returns (packed_move, is_capture) pairs -- is_capture is computed here, once per
        move, and handed back so the search loop never needs to recompute it."""
        k_ply = min(ply, MAX_PLY - 1)
        killer0, killer1 = self.killers[k_ply]
        mover = state[8]
        pawns, _, _, _, _, white, black = int_fields(state)
        ep_square = state[10]
        indexed = []

        for i in range(count):
            f, t, p = int(from_arr[i]), int(to_arr[i]), int(promo_arr[i])
            packed = cst.pack_move(f, t, p)
            is_capture = is_capture_i(white, black, pawns, ep_square, f, t)
            if packed == tt_move:
                score = 1_000_000
            elif is_capture:
                exchange = see(state, f, t)
                score = (100_000 + exchange) if exchange >= 0 else (-100_000 + exchange)
            elif packed == killer0:
                score = 90_000
            elif packed == killer1:
                score = 89_000
            else:
                score = self.history.get((mover, f, t), 0)
            indexed.append((score, packed, is_capture))

        indexed.sort(key=lambda x: x[0], reverse=True)
        return [(packed, is_capture) for _, packed, is_capture in indexed]

    def quiescence(
        self, state: State, alpha: int, beta: int, ply: int, key: int | None = None,
        in_check: bool | None = None,
    ) -> int:
        self._time_check()
        # `key` lets a caller that already hashed this exact state (negamax falling through to
        # quiescence at depth<=0, or quiescence's own move loop just below) pass it straight
        # through -- hash_of is a hot, non-free call (a jitted Zobrist fold over every field),
        # and re-hashing a position the caller just hashed a line earlier was a measured,
        # avoidable chunk of total search time.
        if key is None:
            key = hash_of(state)
        if self._is_draw(state, key):
            return self._draw_score(ply)

        if in_check is None:
            in_check = is_in_check(state)
        from_arr, to_arr, promo_arr, count = legal_moves(state)
        pawns, knights, bishops, rooks, queens, white, black = int_fields(state)
        ep_square = state[10]

        # is_capture is computed here, once per candidate, and carried alongside the move --
        # the loop below used to recompute it a second time per move via the exact same
        # is_capture_i call, found by profiling as a real, avoidable chunk of quiescence's
        # (a very hot path) total cost.
        if in_check:
            if count == 0:
                return -(MATE - ply)
            best = -MATE - 1
            cand = [
                (
                    int(from_arr[i]),
                    int(to_arr[i]),
                    int(promo_arr[i]),
                    is_capture_i(white, black, pawns, ep_square, int(from_arr[i]), int(to_arr[i])),
                )
                for i in range(count)
            ]
        else:
            stand_pat = self.evaluate(state, count)
            if stand_pat >= beta:
                return stand_pat
            if stand_pat > alpha:
                alpha = stand_pat
            best = stand_pat
            cand = []
            for i in range(count):
                f, t, p = int(from_arr[i]), int(to_arr[i]), int(promo_arr[i])
                capture = is_capture_i(white, black, pawns, ep_square, f, t)
                if p or capture:
                    cand.append((f, t, p, capture))

        def qscore(m: tuple[int, int, int, bool]) -> int:
            f, t, p, _ = m
            if p:
                return 200_000
            victim = captured_type_i(pawns, knights, bishops, rooks, queens, ep_square, f, t)
            attacker = moving_type_i(pawns, knights, bishops, rooks, queens, f)
            return int(PIECE_VALUES[victim - 1]) * 10 - int(PIECE_VALUES[attacker - 1])

        for f, t, p, is_capture in sorted(cand, key=qscore, reverse=True):
            # `and not p`: a capturing promotion's delta margin below only accounts for the
            # captured piece's value, never the ~800cp the promotion itself gains -- pruning it
            # on that alone would undervalue a move that's actually excellent. The pre-fix code
            # achieved this exclusion as a side effect of computing is_capture via `not p and
            # is_capture_i(...)`; is_capture here is the raw capture status (computed once, not
            # re-derived per move), so the exclusion has to be spelled out explicitly at the one
            # place it actually matters instead. Found by a second, independent review directly
            # reproducing the divergence (a hand-built position where a capturing promotion was
            # being wrongly pruned), not caught by this session's own move/score/node-count A/B
            # since that didn't happen to include a capturing promotion in the narrow alpha
            # window where it mattered.
            if not in_check and is_capture and not p:
                victim = captured_type_i(pawns, knights, bishops, rooks, queens, ep_square, f, t)
                if stand_pat + int(PIECE_VALUES[victim - 1]) + DELTA_MARGIN <= alpha:
                    continue
                if see(state, f, t) < 0:
                    continue

            child, child_key, child_in_check = apply_move_info(state, key, f, t, p)
            self.seen[child_key] = self.seen.get(child_key, 0) + 1
            try:
                score = -self.quiescence(
                    child, -beta, -alpha, ply + 1, child_key, in_check=child_in_check,
                )
            finally:
                self.seen[child_key] -= 1

            if score > best:
                best = score
            if score > alpha:
                alpha = score
            if alpha >= beta:
                break

        return best

    def negamax(
        self,
        state: State,
        depth: int,
        alpha: int,
        beta: int,
        ply: int,
        allow_null: bool = True,
        key: int | None = None,
        in_check: bool | None = None,
    ) -> int:
        self._time_check()
        # See quiescence's matching `key` parameter: a caller that already hashed this exact
        # state (the move loop below, or _search_root_pass) can pass it straight through.
        if key is None:
            key = hash_of(state)
        if self._is_draw(state, key):
            return self._draw_score(ply)

        tt_score, tt_move = self.tt.probe(key, depth, alpha, beta, ply)
        if tt_score is not None:
            return tt_score

        if depth <= 0:
            return self.quiescence(state, alpha, beta, ply, key, in_check=in_check)

        if in_check is None:
            in_check = is_in_check(state)
        from_arr, to_arr, promo_arr, count = legal_moves(state)
        if count == 0:
            return -(MATE - ply) if in_check else self._draw_score(ply)

        static_eval = None
        if not in_check:
            static_eval = self.evaluate(state, count)
            if depth <= REVERSE_FUTILITY_DEPTH and abs(beta) < MATE_THRESHOLD:
                margin = REVERSE_FUTILITY_MARGIN_PER_PLY * depth
                if static_eval - margin >= beta:
                    return static_eval - margin

        if allow_null and not in_check and depth >= 3 and has_non_pawn_material(state):
            null_state = apply_null(state)
            null_depth = depth - 1 - NULL_MOVE_REDUCTION
            null_score = -self.negamax(
                null_state, null_depth, -beta, -beta + 1, ply + 1, allow_null=False
            )
            if null_score >= beta:
                return null_score

        ordered = self._order_moves(state, from_arr, to_arr, promo_arr, count, tt_move, ply)
        original_alpha = alpha
        best_score = -MATE - 1
        best_move = NO_MOVE
        k_ply = min(ply, MAX_PLY - 1)
        mover = state[8]
        prunable = not is_bare_king_endgame(state)

        for i, (packed, is_capture) in enumerate(ordered):
            f, t, p = cst.unpack_move(packed)
            extension = 1 if in_check else 0
            is_killer = packed == self.killers[k_ply][0] or packed == self.killers[k_ply][1]

            child, child_key, child_in_check = apply_move_info(state, key, f, t, p)

            if (
                prunable
                and depth <= LATE_MOVE_PRUNING_DEPTH
                and i >= LATE_MOVE_PRUNING_BASE + LATE_MOVE_PRUNING_PER_DEPTH * depth
                and extension == 0
                and not is_capture
                and not p
                and not is_killer
                and best_score > -MATE_THRESHOLD
                and not child_in_check
            ):
                continue

            # Futility pruning: near the leaf, a quiet move that can't even reach alpha once
            # the position's current static assessment is padded by a generous margin is not
            # going to be the move that saves this node -- skip searching it entirely, rather
            # than reverse futility pruning's whole-node early return (which only fires when
            # the *whole node* looks good enough to already beat beta before any move is
            # tried). Requires i >= 1 so at least one move has gone through the full search
            # first -- best_score must reflect a real searched move, never a bare static_eval
            # guess, so a lopsided position doesn't get a fabricated score from pruning away
            # literally everything. Gated off entirely in a bare-king endgame (see `prunable`)
            # for the same reason LMR is below -- confirmed via a real regression, not a guess.
            if (
                prunable
                and static_eval is not None
                and i >= 1
                and depth <= FUTILITY_DEPTH
                and extension == 0
                and not is_capture
                and not p
                and not is_killer
                and abs(alpha) < MATE_THRESHOLD
                and static_eval + FUTILITY_MARGIN_PER_PLY * depth <= alpha
                and not child_in_check
            ):
                continue

            reduce = 0
            if (
                prunable
                and depth >= 3
                and i >= 3
                and extension == 0
                and not is_capture
                and not p
                and not is_killer
            ):
                d_idx = depth if depth < _LMR_MAX_DEPTH else _LMR_MAX_DEPTH
                m_idx = i if i < _LMR_MAX_MOVE_INDEX else _LMR_MAX_MOVE_INDEX
                reduce = min(int(LMR_TABLE[d_idx, m_idx]), depth - 1)

            self.seen[child_key] = self.seen.get(child_key, 0) + 1
            try:
                if i == 0:
                    score = -self.negamax(
                        child, depth - 1 + extension, -beta, -alpha, ply + 1,
                        key=child_key, in_check=child_in_check,
                    )
                else:
                    score = -self.negamax(
                        child, depth - 1 + extension - reduce, -alpha - 1, -alpha, ply + 1,
                        key=child_key, in_check=child_in_check,
                    )
                    if score > alpha:
                        score = -self.negamax(
                            child, depth - 1 + extension, -beta, -alpha, ply + 1,
                            key=child_key, in_check=child_in_check,
                        )
            finally:
                self.seen[child_key] -= 1

            if score > best_score:
                best_score = score
                best_move = packed
            if score > alpha:
                alpha = score
            if alpha >= beta:
                if not is_capture:
                    if packed != self.killers[k_ply][0]:
                        self.killers[k_ply][1] = self.killers[k_ply][0]
                        self.killers[k_ply][0] = packed
                    hist_key = (mover, f, t)
                    self.history[hist_key] = self.history.get(hist_key, 0) + depth * depth
                break

        flag = FLAG_EXACT
        if best_score <= original_alpha:
            flag = FLAG_UPPER
        elif best_score >= beta:
            flag = FLAG_LOWER
        self.tt.store(key, depth, best_score, flag, best_move, ply)
        return best_score

    def _search_root_pass(
        self,
        state: State,
        ordered: list[tuple[int, bool]],
        depth: int,
        alpha: int,
        beta: int,
    ) -> tuple[int, int, list[tuple[chess.Move, int]]]:
        """One full-width pass over every root move at the given (alpha, beta) window. Returns
        (best_move_packed, best_score, scored) -- best_score may fall outside (alpha, beta),
        which the caller (search_root) checks to decide whether to re-search with a wider
        window (a false narrow-window result is not trustworthy, only the fact that it needs
        widening is)."""
        best_move = ordered[0][0]
        best_score = -MATE - 1
        scored: list[tuple[chess.Move, int]] = []
        window_alpha = alpha
        key = hash_of(state)

        for i, (packed, _) in enumerate(ordered):
            f, t, p = cst.unpack_move(packed)
            claim = self.root_draw_claims.get(packed, 0)
            child, child_key, child_in_check = apply_move_info(state, key, f, t, p)
            self.seen[child_key] = self.seen.get(child_key, 0) + 1
            try:
                if claim == 1:
                    score = -CONTEMPT
                elif i == 0:
                    score = -self.negamax(
                        child, depth - 1, -beta, -window_alpha, 1,
                        key=child_key, in_check=child_in_check,
                    )
                else:
                    score = -self.negamax(
                        child, depth - 1, -window_alpha - 1, -window_alpha, 1,
                        key=child_key, in_check=child_in_check,
                    )
                    if score > window_alpha:
                        score = -self.negamax(
                            child, depth - 1, -beta, -window_alpha, 1,
                            key=child_key, in_check=child_in_check,
                        )
            finally:
                self.seen[child_key] -= 1

            if claim == 2:
                score = min(score, -CONTEMPT)
            # Prefer another positive continuation before a second occurrence
            # leaves only forced claims next turn. This is a root preference,
            # not an assertion that twofold repetition is a drawn game. Apply
            # it before updating alpha so alternatives are searched against
            # the adjusted score, rather than choosing from fail-low bounds.
            if packed in self.root_repeated_moves:
                score = min(score, 50)
            move_obj = chess.Move(f, t, p if p else None)
            scored.append((move_obj, score))
            if score > best_score:
                best_score = score
                best_move = packed
            if score > window_alpha:
                window_alpha = score

        return best_move, best_score, scored

    def search_root(
        self, board: chess.Board, depth: int, deadline: float, prev_score: int | None = None
    ) -> tuple[chess.Move, int, list[tuple[chess.Move, int]]]:
        """Full-width search of every root move at `depth`. Raises TimeUp if `deadline` passes
        before every root move has been searched -- callers should only trust the return value
        of a call that completes without raising.

        `prev_score` (the completed previous depth's score, when the caller has one) sets an
        aspiration window: search a narrow band around it first, since most of the time a
        position's evaluation doesn't swing much between adjacent depths, and a narrow window
        prunes far more than a full (-MATE, MATE) window would. Widens and re-searches on a
        fail-low or fail-high -- correctness never depends on the window being right, only
        speed does, since a failed narrow search is detected (best_score outside the window)
        and discarded rather than trusted."""
        self.deadline = deadline
        state = state_from_board(board)
        key = hash_of(state)
        from_arr, to_arr, promo_arr, count = legal_moves(state)
        _, tt_move = self.tt.probe(key, depth, -MATE, MATE, 0)
        ordered = self._order_moves(state, from_arr, to_arr, promo_arr, count, tt_move, 0)

        if prev_score is not None and abs(prev_score) < MATE_THRESHOLD and depth >= 4:
            margin = ASPIRATION_INITIAL_MARGIN
            alpha = max(prev_score - margin, -MATE)
            beta = min(prev_score + margin, MATE)
        else:
            alpha, beta = -MATE, MATE

        while True:
            best_move, best_score, scored = self._search_root_pass(
                state, ordered, depth, alpha, beta
            )
            if best_score <= alpha and alpha > -MATE:
                alpha = max(alpha - (beta - alpha), -MATE)
            elif best_score >= beta and beta < MATE:
                beta = min(beta + (beta - alpha), MATE)
            else:
                break

        self.tt.store(key, depth, best_score, FLAG_EXACT, best_move, 0)
        f, t, p = cst.unpack_move(best_move)
        return chess.Move(f, t, p if p else None), best_score, scored
