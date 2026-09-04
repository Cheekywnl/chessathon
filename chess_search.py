"""Negamax alpha-beta search: iterative deepening, a persistent transposition table, quiescence,
null-move pruning, late move reductions, and check extensions.

The transposition table and the real-game position history are owned by the caller and passed
in, because they are the two things worth keeping alive across moves in the same game (see
AGENTS.md). Everything else here -- killers, history heuristic, node count -- is scoped to one
`search_root` call and thrown away.

A timeout unwinds through this file as a `TimeUp` exception raised deep in the recursion. Every
`board.push` here is paired with `board.pop()` in a `finally`, so the caller's board object is
always back in its original position even when a search is aborted mid-tree -- this is the one
invariant a bug here would break silently and expensively.
"""

import threading
import time
from collections.abc import Callable, Hashable

import chess

MATE = 32_000
MATE_THRESHOLD = MATE - 1_000
DRAW = 0
MAX_PLY = 128

FLAG_EXACT, FLAG_LOWER, FLAG_UPPER = 0, 1, 2

PIECE_VALUE = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20_000,
}

NULL_MOVE_REDUCTION = 2
NODES_PER_TIME_CHECK = 1024
REVERSE_FUTILITY_DEPTH = 3
REVERSE_FUTILITY_MARGIN_PER_PLY = 120

TTEntry = tuple[Hashable, int, int, int, chess.Move | None]


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
    construction."""

    def __init__(self, size_power: int = 21) -> None:
        self.mask = (1 << size_power) - 1
        self.table: list[TTEntry | None] = [None] * (1 << size_power)

    def probe(
        self, key: Hashable, depth: int, alpha: int, beta: int, ply: int
    ) -> tuple[int | None, chess.Move | None]:
        entry = self.table[hash(key) & self.mask]
        if entry is None or entry[0] != key:
            return None, None
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

    def store(
        self, key: Hashable, depth: int, score: int, flag: int, move: chess.Move | None, ply: int
    ) -> None:
        index = hash(key) & self.mask
        existing = self.table[index]
        if existing is None or existing[0] == key or existing[1] <= depth:
            self.table[index] = (key, depth, _score_to_tt(score, ply), flag, move)


def see(board: chess.Board, move: chess.Move) -> int:
    """Static exchange evaluation: net centipawn material if all attackers on the target
    square trade off in least-valuable-first order. Negative means the capture loses material."""
    square = move.to_square
    scratch = board.copy(stack=False)

    if scratch.is_en_passant(move):
        gains = [PIECE_VALUE[chess.PAWN]]
    else:
        captured = scratch.piece_type_at(square)
        gains = [PIECE_VALUE[captured] if captured is not None else 0]

    attacker_value = PIECE_VALUE[scratch.piece_type_at(move.from_square)]  # type: ignore[index]
    scratch.push(move)

    while True:
        attackers = scratch.attackers(scratch.turn, square)
        if not attackers:
            break
        least_square = min(attackers, key=lambda s: PIECE_VALUE[scratch.piece_type_at(s)])  # type: ignore[index]
        capture = chess.Move(least_square, square)
        if not scratch.is_legal(capture):
            capture = chess.Move(least_square, square, promotion=chess.QUEEN)
            if not scratch.is_legal(capture):
                break
        gains.append(attacker_value - gains[-1])
        attacker_value = PIECE_VALUE[scratch.piece_type_at(least_square)]  # type: ignore[index]
        scratch.push(capture)

    for i in range(len(gains) - 2, -1, -1):
        gains[i] = -max(-gains[i], gains[i + 1])
    return gains[0]


def _captured_piece_type(board: chess.Board, move: chess.Move) -> int:
    if board.is_en_passant(move):
        return chess.PAWN
    piece = board.piece_type_at(move.to_square)
    return piece if piece is not None else chess.PAWN


def _has_non_pawn_material(board: chess.Board) -> bool:
    side = board.occupied_co[board.turn]
    return bool(side & ~board.pawns & ~board.kings)


def _capture_and_promotion_moves(board: chess.Board) -> list[chess.Move]:
    """Captures and promotions only, via python-chess's targeted generators rather than
    generating every legal move and filtering -- verified move-for-move identical to that
    filter on a range of positions including en passant and promotions, ~3.5x faster."""
    moves = list(board.generate_legal_captures())
    promo_rank = chess.BB_RANK_7 if board.turn == chess.WHITE else chess.BB_RANK_2
    promoting_pawns = board.pawns & board.occupied_co[board.turn] & promo_rank
    if promoting_pawns:
        moves.extend(
            m
            for m in board.generate_legal_moves(from_mask=promoting_pawns)
            if m.promotion and not board.is_capture(m)
        )
    return moves


class Search:
    def __init__(
        self,
        tt: TranspositionTable,
        game_history: dict[Hashable, int],
        evaluate: Callable[[chess.Board, int], int],
        stop_event: threading.Event | None = None,
    ) -> None:
        self.tt = tt
        self.evaluate = evaluate
        self.seen: dict[Hashable, int] = dict(game_history)
        self.killers: list[list[chess.Move | None]] = [[None, None] for _ in range(MAX_PLY)]
        self.history: dict[tuple[bool, int, int], int] = {}
        self.nodes = 0
        self.deadline = 0.0
        self.stop_event = stop_event

    def _time_check(self) -> None:
        self.nodes += 1
        if self.nodes % NODES_PER_TIME_CHECK == 0:
            if self.stop_event is not None and self.stop_event.is_set():
                raise TimeUp
            if time.monotonic() > self.deadline:
                raise TimeUp

    def _is_draw(self, board: chess.Board) -> bool:
        if board.halfmove_clock >= 100:
            return True
        if self.seen.get(board._transposition_key(), 0) >= 3:
            return True
        if board.occupied.bit_count() <= 6:
            return board.is_insufficient_material()
        return False

    def _order_moves(
        self, board: chess.Board, moves: list[chess.Move], tt_move: chess.Move | None, ply: int
    ) -> list[chess.Move]:
        k_ply = min(ply, MAX_PLY - 1)
        killer0, killer1 = self.killers[k_ply]
        mover = board.turn

        def score(move: chess.Move) -> int:
            if move == tt_move:
                return 1_000_000
            if board.is_capture(move):
                victim = _captured_piece_type(board, move)
                attacker = board.piece_type_at(move.from_square)
                attacker_value = PIECE_VALUE[attacker] if attacker is not None else 0
                return 100_000 + PIECE_VALUE[victim] * 10 - attacker_value
            if move == killer0:
                return 90_000
            if move == killer1:
                return 89_000
            return self.history.get((mover, move.from_square, move.to_square), 0)

        return sorted(moves, key=score, reverse=True)

    def quiescence(self, board: chess.Board, alpha: int, beta: int, ply: int) -> int:
        self._time_check()
        if self._is_draw(board):
            return DRAW

        in_check = board.is_check()
        if in_check:
            moves = list(board.legal_moves)
            if not moves:
                return -(MATE - ply)
            best = -MATE - 1
        else:
            all_moves = list(board.legal_moves)
            stand_pat = self.evaluate(board, len(all_moves))
            if stand_pat >= beta:
                return stand_pat
            if stand_pat > alpha:
                alpha = stand_pat
            best = stand_pat
            moves = _capture_and_promotion_moves(board)

        def qscore(move: chess.Move) -> int:
            if move.promotion:
                return 200_000
            victim = _captured_piece_type(board, move)
            attacker = board.piece_type_at(move.from_square)
            attacker_value = PIECE_VALUE[attacker] if attacker is not None else 0
            return PIECE_VALUE[victim] * 10 - attacker_value

        for move in sorted(moves, key=qscore, reverse=True):
            is_capture = not move.promotion and board.is_capture(move)
            if not in_check and is_capture and see(board, move) < 0:
                continue
            board.push(move)
            key = board._transposition_key()
            self.seen[key] = self.seen.get(key, 0) + 1
            try:
                score = -self.quiescence(board, -beta, -alpha, ply + 1)
            finally:
                self.seen[key] -= 1
                board.pop()

            if score > best:
                best = score
            if score > alpha:
                alpha = score
            if alpha >= beta:
                break

        return best

    def negamax(
        self,
        board: chess.Board,
        depth: int,
        alpha: int,
        beta: int,
        ply: int,
        allow_null: bool = True,
    ) -> int:
        self._time_check()
        if self._is_draw(board):
            return DRAW

        key = board._transposition_key()
        tt_score, tt_move = self.tt.probe(key, depth, alpha, beta, ply)
        if tt_score is not None:
            return tt_score

        if depth <= 0:
            return self.quiescence(board, alpha, beta, ply)

        in_check = board.is_check()
        moves = list(board.legal_moves)
        if not moves:
            return -(MATE - ply) if in_check else DRAW

        if not in_check and depth <= REVERSE_FUTILITY_DEPTH and abs(beta) < MATE_THRESHOLD:
            static_eval = self.evaluate(board, len(moves))
            margin = REVERSE_FUTILITY_MARGIN_PER_PLY * depth
            if static_eval - margin >= beta:
                return static_eval - margin

        if allow_null and not in_check and depth >= 3 and _has_non_pawn_material(board):
            board.push(chess.Move.null())
            try:
                null_depth = depth - 1 - NULL_MOVE_REDUCTION
                null_score = -self.negamax(
                    board, null_depth, -beta, -beta + 1, ply + 1, allow_null=False
                )
            finally:
                board.pop()
            if null_score >= beta:
                return null_score

        ordered = self._order_moves(board, moves, tt_move, ply)
        original_alpha = alpha
        best_score = -MATE - 1
        best_move: chess.Move | None = None
        k_ply = min(ply, MAX_PLY - 1)

        for i, move in enumerate(ordered):
            mover = board.turn
            is_capture = board.is_capture(move)
            extension = 1 if in_check else 0

            reduce = 0
            if (
                depth >= 3
                and i >= 3
                and extension == 0
                and not is_capture
                and not move.promotion
                and move != self.killers[k_ply][0]
                and move != self.killers[k_ply][1]
            ):
                reduce = 1

            board.push(move)
            new_key = board._transposition_key()
            self.seen[new_key] = self.seen.get(new_key, 0) + 1
            try:
                if i == 0:
                    score = -self.negamax(board, depth - 1 + extension, -beta, -alpha, ply + 1)
                else:
                    score = -self.negamax(
                        board, depth - 1 + extension - reduce, -alpha - 1, -alpha, ply + 1
                    )
                    if score > alpha:
                        score = -self.negamax(board, depth - 1 + extension, -beta, -alpha, ply + 1)
            finally:
                self.seen[new_key] -= 1
                board.pop()

            if score > best_score:
                best_score = score
                best_move = move
            if score > alpha:
                alpha = score
            if alpha >= beta:
                if not is_capture:
                    if move != self.killers[k_ply][0]:
                        self.killers[k_ply][1] = self.killers[k_ply][0]
                        self.killers[k_ply][0] = move
                    hist_key = (mover, move.from_square, move.to_square)
                    self.history[hist_key] = self.history.get(hist_key, 0) + depth * depth
                break

        flag = FLAG_EXACT
        if best_score <= original_alpha:
            flag = FLAG_UPPER
        elif best_score >= beta:
            flag = FLAG_LOWER
        self.tt.store(key, depth, best_score, flag, best_move, ply)
        return best_score

    def search_root(
        self, board: chess.Board, depth: int, deadline: float
    ) -> tuple[chess.Move, int, list[tuple[chess.Move, int]]]:
        """Full-width search of every root move at `depth`. Raises TimeUp if `deadline` passes
        before every root move has been searched -- callers should only trust the return value
        of a call that completes without raising."""
        self.deadline = deadline
        moves = list(board.legal_moves)
        key = board._transposition_key()
        _, tt_move = self.tt.probe(key, depth, -MATE, MATE, 0)
        ordered = self._order_moves(board, moves, tt_move, 0)

        alpha, beta = -MATE, MATE
        best_move = ordered[0]
        best_score = -MATE - 1
        scored: list[tuple[chess.Move, int]] = []

        for i, move in enumerate(ordered):
            board.push(move)
            new_key = board._transposition_key()
            self.seen[new_key] = self.seen.get(new_key, 0) + 1
            try:
                if i == 0:
                    score = -self.negamax(board, depth - 1, -beta, -alpha, 1)
                else:
                    score = -self.negamax(board, depth - 1, -alpha - 1, -alpha, 1)
                    if score > alpha:
                        score = -self.negamax(board, depth - 1, -beta, -alpha, 1)
            finally:
                self.seen[new_key] -= 1
                board.pop()

            scored.append((move, score))
            if score > best_score:
                best_score = score
                best_move = move
            if score > alpha:
                alpha = score

        self.tt.store(key, depth, best_score, FLAG_EXACT, best_move, 0)
        return best_move, best_score, scored
