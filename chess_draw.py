"""Root move consequences under the referee's automatic FIDE draw claims.

The real game history includes both sides to move. Hypothetical pushes update a
private copy, so a claim is recognized before the third occurrence is played.
An immediate referee draw is exact; an opponent's drawing option only caps a
move's score and must never turn a losing continuation into a guaranteed draw.
"""

import time

import chess
import numpy as np
from numba import njit

import chess_movegen as mg
import chess_search as cs
import chess_state as cst

IMMEDIATE = 1
OPPONENT_CAN_DRAW = 2


@njit(cache=False)
def _legal(state: cs.State) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    f, t, p, n = mg.generate_legal_moves_bb(
        state[0], state[1], state[2], state[3], state[4], state[5], state[6], state[7],
        state[8], state[9], state[10],
    )
    return f, t, p, n


@njit(cache=False)
def _after(state: cs.State, f: int, t: int, p: int) -> cs.State:
    result = cst.make_move(
        state[0], state[1], state[2], state[3], state[4], state[5], state[6], state[7],
        state[8], state[9], state[10], state[11], f, t, p,
    )
    return (result[0], result[1], result[2], result[3], result[4], result[5],
            result[6], result[7], result[8], result[9], result[10], result[11])


@njit(cache=False)
def _was_seen_twice(state: cs.State, repeated: np.ndarray) -> bool:
    key = cst.zobrist_hash(
        state[0], state[1], state[2], state[3], state[4], state[5], state[6], state[7],
        state[8], state[9], state[10], cst.ZOBRIST_PIECE_SQUARE,
        cst.ZOBRIST_CASTLING, cst.ZOBRIST_EP_FILE, cst.ZOBRIST_TURN,
    )
    index = np.searchsorted(repeated, key)
    return bool(index < len(repeated) and repeated[index] == key)


@njit(cache=False)
def _claim(state: cs.State, repeated: np.ndarray) -> bool:
    f, t, p, count = _legal(state)
    if count == 0:
        own = state[6] if state[8] else state[7]
        king = cst._lsb_index(state[5] & own)
        return not mg.is_square_attacked(
            king, not state[8], state[0], state[1], state[2], state[3],
            state[4], state[5], state[6], state[7],
        )
    if state[11] >= 100 or _was_seen_twice(state, repeated):
        return True
    for index in range(count):
        child = _after(state, int(f[index]), int(t[index]), int(p[index]))
        if _was_seen_twice(child, repeated):
            return True
        if child[11] >= 100:
            _, _, _, child_count = _legal(child)
            if child_count > 0:
                return True
    return False


@njit(cache=False)
def _move_claim(state: cs.State, f: int, t: int, p: int, repeated: np.ndarray) -> int:
    child = _after(state, f, t, p)
    if _claim(child, repeated):
        return IMMEDIATE
    rf, rt, rp, count = _legal(child)
    for index in range(count):
        reply = _after(child, int(rf[index]), int(rt[index]), int(rp[index]))
        if reply[11] == 0:
            continue
        if _claim(reply, repeated):
            return OPPONENT_CAN_DRAW
    return 0


def _push(board: chess.Board, move: chess.Move, seen: dict[int, int]) -> int:
    board.push(move)
    key = cs.hash_of_board(board)
    seen[key] = seen.get(key, 0) + 1
    return key


def _pop(board: chess.Board, key: int, seen: dict[int, int]) -> None:
    seen[key] -= 1
    if seen[key] == 0:
        del seen[key]
    board.pop()


def automatic_draw(board: chess.Board, seen: dict[int, int]) -> bool:
    """Match outcome(claim_draw=True), with counts supplied across FEN requests."""
    legal = list(board.legal_moves)
    if not legal:
        return not board.is_check()  # Checkmate takes precedence over draw rules.
    if board.is_insufficient_material():
        return True
    if board.halfmove_clock >= 100 or seen.get(cs.hash_of_board(board), 0) >= 3:
        return True
    for move in legal:
        key = _push(board, move, seen)
        try:
            if seen[key] >= 3:
                return True
            if board.halfmove_clock >= 100 and any(board.legal_moves):
                return True
        finally:
            _pop(board, key, seen)
    return False


def root_claims(
    board: chess.Board, history: dict[int, int], moves: list[chess.Move], deadline: float,
) -> tuple[dict[int, int], bool]:
    """Classify exact claims and opponent draw options; keep proven partial results.

    Scan at most our move, their reply and a claim by intended next move. Root
    move ordering comes from the caller. The time limit leaves time for search;
    an unexamined move has no claimed bound, rather than an invented result.
    """
    if board.halfmove_clock < 97 and not any(count >= 2 for count in history.values()):
        return {}, True
    # No legal chess position can recur after only two plies. Therefore none of
    # these three hypothetical plies can revisit a new path position twice:
    # every possible third occurrence must target a position already seen twice
    # in the real game. A sorted array makes that fixed set cheap to probe in JIT.
    repeated = np.array(sorted(key for key, count in history.items() if count >= 2),
                        dtype=np.uint64)
    state = cs.state_from_board(board)
    claims: dict[int, int] = {}
    for move in moves:
        if time.monotonic() >= deadline:
            return claims, False
        # A zeroing move removes all repetition/fifty-move risk in this horizon.
        # Search already handles immediate stalemate and insufficient material.
        if board.is_zeroing(move):
            continue
        packed = cst.pack_move(move.from_square, move.to_square, move.promotion or 0)
        claim = int(_move_claim(state, move.from_square, move.to_square,
                                move.promotion or 0, repeated))
        if claim:
            claims[packed] = claim
    return claims, True


def warm_up() -> None:
    board = chess.Board()
    root_claims(board, {cs.hash_of_board(board): 2}, list(board.legal_moves), float("inf"))


def repeated_moves(
    board: chess.Board, history: dict[int, int], moves: list[chess.Move],
) -> set[int]:
    """Moves returning to a real position, before a cycle can become forced."""
    state = cs.state_from_board(board)
    repeated = set()
    for move in moves:
        child = cs.apply_move(state, move.from_square, move.to_square, move.promotion or 0)
        if history.get(cs.hash_of(child), 0) >= 1:
            repeated.add(cst.pack_move(move.from_square, move.to_square, move.promotion or 0))
    return repeated
