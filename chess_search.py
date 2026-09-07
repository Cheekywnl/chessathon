"""Negamax alpha-beta search: iterative deepening, a persistent transposition table, quiescence,
null-move pruning, late move reductions, and check extensions.

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

import threading
import time

import chess
import numpy as np

import chess_eval as ce
import chess_movegen as mg
import chess_state as cst

MATE = 32_000
MATE_THRESHOLD = MATE - 1_000
DRAW = 0
MAX_PLY = 128
NO_MOVE = -1

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
ASPIRATION_INITIAL_MARGIN = 25

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
    return cst.hash_state(
        int(state[0]), int(state[1]), int(state[2]), int(state[3]), int(state[4]), int(state[5]),
        int(state[6]), int(state[7]), state[8], int(state[9]), state[10],
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


def insufficient_material(state: State) -> bool:
    """Only ever reached with <= 6 pieces on the board -- rare enough that reconstructing a
    throwaway chess.Board to reuse its (fiddly, well-tested) insufficient-material rule is
    cheaper than re-deriving that rule correctly from scratch."""
    board = chess.Board.empty()
    piece_bb = ((PAWN, state[0]), (KNIGHT, state[1]), (BISHOP, state[2]), (ROOK, state[3]),
                (QUEEN, state[4]), (KING, state[5]))
    for piece_type, bb in piece_bb:
        bb_int = int(bb)
        while bb_int:
            sq = (bb_int & -bb_int).bit_length() - 1
            bb_int &= bb_int - 1
            color = bool(int(state[6]) & (1 << sq))
            board.set_piece_at(sq, chess.Piece(piece_type, color))
    return board.is_insufficient_material()


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

    def evaluate(self, state: State, mobility: int) -> int:
        white_count, black_count = _castling_rights_counts(state)
        positional = ce.evaluate(
            state[0], state[1], state[2], state[3], state[4], state[5], state[6], state[7],
            ce.MG_PST, ce.EG_PST, ce.PASSED_MASK_WHITE, ce.PASSED_MASK_BLACK,
            ce.FILE_MASK, ce.ADJACENT_FILE_MASK, white_count, black_count, self.params,
        )
        mover_relative = int(positional) if state[8] else -int(positional)
        return mover_relative + int(self.params[ce.P_MOBILITY]) * mobility

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
        if bin(int(occupied)).count("1") <= 6:
            return insufficient_material(state)
        return False

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

    def quiescence(self, state: State, alpha: int, beta: int, ply: int) -> int:
        self._time_check()
        key = hash_of(state)
        if self._is_draw(state, key):
            return DRAW

        in_check = is_in_check(state)
        from_arr, to_arr, promo_arr, count = legal_moves(state)
        pawns, knights, bishops, rooks, queens, white, black = int_fields(state)
        ep_square = state[10]

        if in_check:
            if count == 0:
                return -(MATE - ply)
            best = -MATE - 1
            cand = [(int(from_arr[i]), int(to_arr[i]), int(promo_arr[i])) for i in range(count)]
        else:
            stand_pat = self.evaluate(state, count)
            if stand_pat >= beta:
                return stand_pat
            if stand_pat > alpha:
                alpha = stand_pat
            best = stand_pat
            cand = [
                (int(from_arr[i]), int(to_arr[i]), int(promo_arr[i]))
                for i in range(count)
                if promo_arr[i]
                or is_capture_i(white, black, pawns, ep_square, int(from_arr[i]), int(to_arr[i]))
            ]

        def qscore(m: tuple[int, int, int]) -> int:
            f, t, p = m
            if p:
                return 200_000
            victim = captured_type_i(pawns, knights, bishops, rooks, queens, ep_square, f, t)
            attacker = moving_type_i(pawns, knights, bishops, rooks, queens, f)
            return int(PIECE_VALUES[victim - 1]) * 10 - int(PIECE_VALUES[attacker - 1])

        for f, t, p in sorted(cand, key=qscore, reverse=True):
            is_capture = not p and is_capture_i(white, black, pawns, ep_square, f, t)
            if not in_check and is_capture:
                victim = captured_type_i(pawns, knights, bishops, rooks, queens, ep_square, f, t)
                if stand_pat + int(PIECE_VALUES[victim - 1]) + DELTA_MARGIN <= alpha:
                    continue
                if see(state, f, t) < 0:
                    continue

            child = apply_move(state, f, t, p)
            child_key = hash_of(child)
            self.seen[child_key] = self.seen.get(child_key, 0) + 1
            try:
                score = -self.quiescence(child, -beta, -alpha, ply + 1)
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
    ) -> int:
        self._time_check()
        key = hash_of(state)
        if self._is_draw(state, key):
            return DRAW

        tt_score, tt_move = self.tt.probe(key, depth, alpha, beta, ply)
        if tt_score is not None:
            return tt_score

        if depth <= 0:
            return self.quiescence(state, alpha, beta, ply)

        in_check = is_in_check(state)
        from_arr, to_arr, promo_arr, count = legal_moves(state)
        if count == 0:
            return -(MATE - ply) if in_check else DRAW

        if not in_check and depth <= REVERSE_FUTILITY_DEPTH and abs(beta) < MATE_THRESHOLD:
            static_eval = self.evaluate(state, count)
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

        for i, (packed, is_capture) in enumerate(ordered):
            f, t, p = cst.unpack_move(packed)
            extension = 1 if in_check else 0
            is_killer = packed == self.killers[k_ply][0] or packed == self.killers[k_ply][1]

            child = apply_move(state, f, t, p)

            if (
                depth <= LATE_MOVE_PRUNING_DEPTH
                and i >= LATE_MOVE_PRUNING_BASE + LATE_MOVE_PRUNING_PER_DEPTH * depth
                and extension == 0
                and not is_capture
                and not p
                and not is_killer
                and best_score > -MATE_THRESHOLD
                and not is_in_check(child)
            ):
                continue

            reduce = 0
            if (
                depth >= 3
                and i >= 3
                and extension == 0
                and not is_capture
                and not p
                and not is_killer
            ):
                reduce = 1

            child_key = hash_of(child)
            self.seen[child_key] = self.seen.get(child_key, 0) + 1
            try:
                if i == 0:
                    score = -self.negamax(child, depth - 1 + extension, -beta, -alpha, ply + 1)
                else:
                    score = -self.negamax(
                        child, depth - 1 + extension - reduce, -alpha - 1, -alpha, ply + 1
                    )
                    if score > alpha:
                        score = -self.negamax(child, depth - 1 + extension, -beta, -alpha, ply + 1)
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

        for i, (packed, _) in enumerate(ordered):
            f, t, p = cst.unpack_move(packed)
            child = apply_move(state, f, t, p)
            child_key = hash_of(child)
            self.seen[child_key] = self.seen.get(child_key, 0) + 1
            try:
                if i == 0:
                    score = -self.negamax(child, depth - 1, -beta, -window_alpha, 1)
                else:
                    score = -self.negamax(child, depth - 1, -window_alpha - 1, -window_alpha, 1)
                    if score > window_alpha:
                        score = -self.negamax(child, depth - 1, -beta, -window_alpha, 1)
            finally:
                self.seen[child_key] -= 1

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
