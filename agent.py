"""The submission entrypoint. The platform imports this file and calls get_move.

Search and evaluation live in chess_search.py / chess_eval.py; this file owns the state that
has to survive between moves in the same game -- the transposition table and the real-game
position history -- and the time budget that keeps a slow position from flagging the clock.
"""

import sys
import threading
import time
from pathlib import Path

import chess
import chess.syzygy

import chess_eval as ce
import chess_movegen as mg
import chess_search as cs
import chess_state as cst

SAFETY_MARGIN_MS = 300.0
MIN_THINK_MS = 50.0
MAX_SEARCH_DEPTH = 64
PONDER_TIME_CAP_S = 300.0
PONDER_JOIN_TIMEOUT_S = 1.0

# Real rated games showed the search choosing to repeat a position it was clearly winning --
# not a close call needing contempt's small nudge, but a rook or more ahead (round 76: +500cp
# the entire final stretch; round 80: as much as -641cp in our favour) thrown away for a draw.
# Contempt scores the repeat as mildly bad once the search sees it, but the search runs a fresh,
# time-boxed iterative deepening every move and isn't guaranteed to explore far enough down a
# specific repeating line to discover that consequence before its clock runs out -- especially
# when giving check keeps looking locally good move after move. This is a deterministic backstop
# instead of hoping deeper search finds it: after search_root already scored every legal move
# once, if the chosen move would create our own third occurrence of a position while we're
# clearly ahead, take the next-best move that doesn't, as long as it's still clearly winning.
# Costs no extra search time -- every candidate's score already exists in `scored`.
REPETITION_AVOIDANCE_THRESHOLD = 150
REPETITION_AVOIDANCE_MIN_SCORE = 50

# Import time runs once per game, inside a 60 second budget, before your clock starts.
# size_power=21 measures at ~1.0 GB fully populated plus ~0.12 GB baseline (interpreter,
# numpy/numba, python-chess) -- comfortably under the 2 GB cap with ~0.85 GB of margin left
# for board copies, recursion, and the ponder thread's own local state. An OOM kill is an
# instant loss, so this was re-measured empirically before raising it, not guessed.
_TT = cs.TranspositionTable(size_power=21)
_GAME_HISTORY: dict[int, int] = {}

# Syzygy 3-4 piece endgame tables (K+R vs K, K+B+N vs K, K+Q+Q vs K, and every other 3-4 piece
# ending): ~4 MB of WDL+DTZ data covering exactly the hard conversions this session's search
# alone couldn't reliably close out -- the mop-up and KBN-corner-target eval terms are heuristic
# guesses at the same problem this solves exactly, by table lookup instead of search. Explicitly
# permitted as shipped data (chess.syzygy ships in the base image for exactly this), distinct
# from shipping another engine's move/eval opinions: this is exact, retrograde-solved
# game-theoretic truth, not a heuristic. Directory may be absent in a stripped-down local
# checkout; fails open to plain search rather than crashing the game.
_SYZYGY_DIR = Path(__file__).resolve().parent / "syzygy"
_TABLEBASE = chess.syzygy.open_tablebase(str(_SYZYGY_DIR)) if _SYZYGY_DIR.is_dir() else None
MAX_TABLEBASE_PIECES = 4

# Pondering: while the opponent thinks, the harness blocks on stdin and this core sits idle
# unless we use it ourselves -- the rules explicitly allow this ("the process keeps its core
# while the opponent thinks"). A ponder thread keeps deepening on our predicted reply to our
# own move and banks results into the shared transposition table. It is always stopped and
# joined at the top of the *next* get_move, before any new work starts, so the ponder thread
# and the real search thread never touch shared state at the same time -- no true concurrent
# access to guard, just a clean handoff.
_ponder_thread: threading.Thread | None = None
_ponder_stop = threading.Event()

ce.warm_up()
mg.warm_up()
cst.warm_up()


def _time_budget(time_left_ms: float, fullmove_number: int) -> tuple[float, float]:
    """(soft_ms, hard_ms): soft is when iterative deepening stops starting new depths, hard is
    the absolute deadline passed into the search. Budgeted from the clock we were handed, not a
    constant, and never spends the increment before it has actually been credited.

    Real rated games showed the previous formula (moves_to_go floor 15, hard cap 4x soft, 50%
    of remaining time) burning most of the clock by move 30 and then playing out the rest of a
    long game -- these regularly run 100+ plies -- on 2-4 seconds a move for 40+ more moves.
    The hard cap let any single complex position eat a large multiple of the intended average,
    and that happened often enough, not as a rare exception, to be the actual failure mode.
    Tighter now: a higher moves-to-go floor and reference (don't assume the game is nearly over
    just because it's already gone long -- these games often haven't), a 2x hard-cap multiplier
    instead of 4x, and at most 25% of remaining time on any one move instead of 50%."""
    moves_to_go = max(20, 60 - fullmove_number)
    soft_ms = max(time_left_ms / moves_to_go, MIN_THINK_MS)
    hard_ms = max(
        min(soft_ms * 2.0, time_left_ms * 0.25, time_left_ms - SAFETY_MARGIN_MS),
        MIN_THINK_MS,
    )
    return soft_ms, hard_ms


def _stop_pondering() -> None:
    global _ponder_thread
    if _ponder_thread is not None:
        _ponder_stop.set()
        _ponder_thread.join(timeout=PONDER_JOIN_TIMEOUT_S)
        _ponder_thread = None


def _predict_reply(board_after_our_move: chess.Board) -> chess.Move | None:
    key = cs.hash_of_board(board_after_our_move)
    _, tt_move_packed = _TT.probe(key, 0, -cs.MATE, cs.MATE, 0)
    tt_move = cs.packed_to_move(tt_move_packed)
    if tt_move is not None and tt_move in board_after_our_move.legal_moves:
        return tt_move
    return None


def _ponder(board_to_ponder: chess.Board, stop_event: threading.Event) -> None:
    search = cs.Search(_TT, _GAME_HISTORY, stop_event=stop_event)
    deadline = time.monotonic() + PONDER_TIME_CAP_S
    depth = 1
    last_score: int | None = None
    while depth <= MAX_SEARCH_DEPTH:
        try:
            _, last_score, _ = search.search_root(
                board_to_ponder, depth, deadline, prev_score=last_score
            )
        except cs.TimeUp:
            return
        depth += 1


def _tablebase_move(board: chess.Board) -> chess.Move | None:
    """The provably best move by Syzygy WDL/DTZ, or None if this position isn't covered (too
    many pieces, castling rights still held -- Syzygy tables never contain those -- or a probe
    came back unreadable). Never guesses from a partial read: if any candidate move's outcome
    can't be read, the whole position is abandoned back to search rather than trusted halfway.

    Picks the move giving the best reachable result category (win > cursed win > draw > blessed
    loss > loss, all from our side's perspective). Among moves tied on a real win, prefers
    smaller |dtz| on the resulting (opponent-to-move) position -- fewer plies for them to reach
    a zeroing move means faster progress for us; ties elsewhere don't matter game-theoretically,
    so the first one found stands."""
    if (
        _TABLEBASE is None
        or chess.popcount(board.occupied) > MAX_TABLEBASE_PIECES
        or board.castling_rights
    ):
        return None

    best_move: chess.Move | None = None
    best_wdl = -3
    best_progress = 0
    for move in board.legal_moves:
        board.push(move)
        wdl = _TABLEBASE.get_wdl(board)
        if wdl is None:
            board.pop()
            return None
        dtz = _TABLEBASE.get_dtz(board)
        board.pop()
        our_wdl = -wdl
        progress = -abs(dtz) if (our_wdl > 0 and dtz is not None) else 0
        if best_move is None or (our_wdl, progress) > (best_wdl, best_progress):
            best_move, best_wdl, best_progress = move, our_wdl, progress
    return best_move


def _repeats_if_played(board: chess.Board, move: chess.Move) -> bool:
    board.push(move)
    key = cs.hash_of_board(board)
    board.pop()
    return _GAME_HISTORY.get(key, 0) >= 2


def _avoid_needless_repetition(
    board: chess.Board, best_move: chess.Move, best_score: int, scored: list[tuple[chess.Move, int]]
) -> chess.Move:
    if best_score < REPETITION_AVOIDANCE_THRESHOLD or not _repeats_if_played(board, best_move):
        return best_move
    for move, score in sorted(scored, key=lambda pair: -pair[1]):
        if score < REPETITION_AVOIDANCE_MIN_SCORE:
            break
        if not _repeats_if_played(board, move):
            return move
    return best_move


def _start_pondering(board: chess.Board, our_move: chess.Move) -> None:
    global _ponder_thread
    board_after_us = board.copy(stack=False)
    board_after_us.push(our_move)
    if board_after_us.is_game_over(claim_draw=False):
        return
    ponder_move = _predict_reply(board_after_us)
    if ponder_move is None:
        return
    board_to_ponder = board_after_us
    board_to_ponder.push(ponder_move)
    _ponder_stop.clear()
    _ponder_thread = threading.Thread(
        target=_ponder, args=(board_to_ponder, _ponder_stop), daemon=True
    )
    _ponder_thread.start()


def get_move(fen: str, time_left_ms: int) -> str:
    """Return a legal move in UCI notation.

    fen           the position to move in; your colour is the side to move
    time_left_ms  your clock before this move, in milliseconds
    returns       "e2e4", or "e7e8q" for a promotion

    The process stays alive between your moves, so state you keep on a module or in a
    closure survives to the next call. It does not survive to the next game.
    """
    _stop_pondering()

    board = chess.Board(fen)
    key = cs.hash_of_board(board)
    _GAME_HISTORY[key] = _GAME_HISTORY.get(key, 0) + 1

    legal_moves = list(board.legal_moves)
    if len(legal_moves) == 1:
        return legal_moves[0].uci()

    tablebase_move = _tablebase_move(board)
    if tablebase_move is not None:
        print(f"move={tablebase_move.uci()} source=tablebase", file=sys.stderr)
        _start_pondering(board, tablebase_move)
        return tablebase_move.uci()

    start = time.monotonic()
    soft_ms, hard_ms = _time_budget(float(time_left_ms), board.fullmove_number)
    deadline = start + hard_ms / 1000.0
    soft_deadline = start + soft_ms / 1000.0

    _, tt_move_packed = _TT.probe(key, 0, -cs.MATE, cs.MATE, 0)
    tt_move = cs.packed_to_move(tt_move_packed)
    best_move = tt_move if tt_move in legal_moves else legal_moves[0]

    search = cs.Search(_TT, _GAME_HISTORY)
    depth = 1
    completed_depth = 0
    last_score = 0
    scored: list[tuple[chess.Move, int]] = []
    while depth <= MAX_SEARCH_DEPTH:
        prev_score = last_score if completed_depth > 0 else None
        try:
            move, score, root_scores = search.search_root(
                board, depth, deadline, prev_score=prev_score
            )
        except cs.TimeUp:
            break
        best_move = move
        completed_depth = depth
        last_score = score
        scored = root_scores
        if abs(score) >= cs.MATE_THRESHOLD or time.monotonic() >= soft_deadline:
            break
        depth += 1

    if completed_depth > 0:
        best_move = _avoid_needless_repetition(board, best_move, last_score, scored)

    # Safe per the rules: stdout is redirected away from the protocol stream before this
    # module is even imported, so print() can never corrupt it. Discarded in rated games,
    # shown in the validation log -- cheap visibility into real games, not just local ones.
    elapsed_ms = (time.monotonic() - start) * 1000.0
    print(
        f"move={best_move.uci()} depth={completed_depth} score={last_score} "
        f"nodes={search.nodes} ms={elapsed_ms:.0f} time_left_ms={time_left_ms}",
        file=sys.stderr,
    )

    _start_pondering(board, best_move)
    return best_move.uci()
