"""The submission entrypoint. The platform imports this file and calls get_move.

Search and evaluation live in chess_search.py / chess_eval.py; this file owns the state that
has to survive between moves in the same game -- the transposition table and the real-game
position history -- and the time budget that keeps a slow position from flagging the clock.
"""

import time

import chess

import chess_eval as ce
import chess_search as cs

SAFETY_MARGIN_MS = 300.0
MIN_THINK_MS = 50.0
MAX_SEARCH_DEPTH = 64

# Import time runs once per game, inside a 60 second budget, before your clock starts.
# The transposition table is sized to stay well under the 2 GB cap even fully populated
# (~0.5 GB at this size, measured empirically) -- an OOM kill is an instant loss, so this
# errs toward a smaller table and more hash collisions over a long game rather than a bigger
# one that risks the ceiling.
_TT = cs.TranspositionTable(size_power=20)
_GAME_HISTORY: dict[object, int] = {}

ce.warm_up()


def _time_budget(time_left_ms: float, fullmove_number: int) -> tuple[float, float]:
    """(soft_ms, hard_ms): soft is when iterative deepening stops starting new depths, hard is
    the absolute deadline passed into the search. Budgeted from the clock we were handed, not a
    constant, and never spends the increment before it has actually been credited."""
    moves_to_go = max(15, 45 - fullmove_number)
    soft_ms = max(time_left_ms / moves_to_go, MIN_THINK_MS)
    hard_ms = max(
        min(soft_ms * 4.0, time_left_ms * 0.5, time_left_ms - SAFETY_MARGIN_MS),
        MIN_THINK_MS,
    )
    return soft_ms, hard_ms


def get_move(fen: str, time_left_ms: int) -> str:
    """Return a legal move in UCI notation.

    fen           the position to move in; your colour is the side to move
    time_left_ms  your clock before this move, in milliseconds
    returns       "e2e4", or "e7e8q" for a promotion

    The process stays alive between your moves, so state you keep on a module or in a
    closure survives to the next call. It does not survive to the next game.
    """
    board = chess.Board(fen)
    key = board._transposition_key()
    _GAME_HISTORY[key] = _GAME_HISTORY.get(key, 0) + 1

    legal_moves = list(board.legal_moves)
    if len(legal_moves) == 1:
        return legal_moves[0].uci()

    start = time.monotonic()
    soft_ms, hard_ms = _time_budget(float(time_left_ms), board.fullmove_number)
    deadline = start + hard_ms / 1000.0
    soft_deadline = start + soft_ms / 1000.0

    _, tt_move = _TT.probe(key, 0, -cs.MATE, cs.MATE, 0)
    best_move = tt_move if tt_move in legal_moves else legal_moves[0]

    search = cs.Search(_TT, _GAME_HISTORY, ce.evaluate_board)
    depth = 1
    while depth <= MAX_SEARCH_DEPTH:
        try:
            move, score, _ = search.search_root(board, depth, deadline)
        except cs.TimeUp:
            break
        best_move = move
        if abs(score) >= cs.MATE_THRESHOLD or time.monotonic() >= soft_deadline:
            break
        depth += 1

    return best_move.uci()
