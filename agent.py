"""The submission entrypoint. The platform imports this file and calls get_move.

Search and evaluation live in chess_search.py / chess_eval.py; this file owns the state that
has to survive between moves in the same game -- the transposition table and the real-game
position history -- and the time budget that keeps a slow position from flagging the clock.
"""

import atexit
import gzip
import shutil
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path

import chess
import chess.polyglot
import chess.syzygy

import chess_draw as cd
import chess_eval as ce
import chess_movegen as mg
import chess_search as cs
import chess_state as cst

SAFETY_MARGIN_MS = 300.0
MIN_THINK_MS = 50.0
MAX_SEARCH_DEPTH = 64
PONDER_TIME_CAP_S = 300.0
PONDER_JOIN_TIMEOUT_S = 1.0

# Claim analysis is charged to this turn and bounded independently of the search.
DRAW_SCAN_MAX_SECONDS = 0.25
DRAW_SCAN_BUDGET_FRACTION = 0.20

# Import time runs once per game, inside a 90 second budget, before your clock starts.
# The fixed 2**21-slot array table occupies about38MB regardless of fill level.
# Search compilation and asset loading complete before get_move is called.
_TT = cs.TranspositionTable(size_power=21)
_GAME_HISTORY: dict[int, int] = {}

# Syzygy endgame tables: every 3-4 piece ending (K+R vs K, K+B+N vs K, K+Q+Q vs K, ...) plus a
# handful of individually cherry-picked 5-piece endings common enough in real play to be worth
# their size specifically (KRPvKR's 29 MB alone would eat most of the remaining budget for one
# config, so it's deliberately not included -- the repetition backstop already resolves the
# canonical Lucena/Philidor case that config would cover, see tools/endgame_regression.py).
# ~26 MB of WDL+DTZ data covering exactly the hard conversions this session's search alone
# couldn't reliably close out -- the mop-up and KBN-corner-target eval terms are heuristic
# guesses at the same problem this solves exactly, by table lookup instead of search. Explicitly
# permitted as shipped data (chess.syzygy ships in the base image for exactly this), distinct
# from shipping another engine's move/eval opinions: this is exact, retrograde-solved
# game-theoretic truth, not a heuristic. A position matching a 5-piece config not among the ones
# actually downloaded just falls through to search -- get_wdl/get_dtz return None on a missing
# table rather than raising, and _tablebase_move treats that the same as no coverage at all. A
# *corrupted* table is a different failure mode get_wdl/get_dtz do NOT catch internally (they
# only handle the missing-table KeyError, and a bad file raises OSError instead) -- confirmed
# directly against a deliberately corrupted file, not assumed: _tablebase_move's own outer
# exception handler is what actually saves that case, falling back to plain search rather than
# crashing the game. Directory may be absent in a stripped-down local checkout; fails open too.
#
# Loading is wrapped, not called bare: this runs at import time, and the platform gives 90
# seconds before the clock starts but an *exception* here, not a slow load, would fail the
# import outright -- an agent that doesn't import loses every single game, not just the ones
# that would have used the tablebase. A corrupted file in the zip transfer, an unexpected
# filesystem quirk on the platform, or anything else improbable-but-not-impossible here should
# cost this one feature, never the whole submission. Same reasoning applies to the book below.
def _open_tablebase() -> "chess.syzygy.Tablebase | None":
    directory = Path(__file__).resolve().parent / "syzygy"
    if not directory.is_dir():
        return None
    try:
        return chess.syzygy.open_tablebase(str(directory))
    except Exception as exc:
        print(f"tablebase load failed, continuing without it: {exc}", file=sys.stderr)
        return None


_TABLEBASE = _open_tablebase()
MAX_TABLEBASE_PIECES = 5

# Opening book (CodeKiddy Polyglot collection, ~16 MB, compiled from a large human-games
# database -- not another engine's move/eval opinions, same permitted-as-shipped-data category
# as the tablebases above). Genuinely worth less here than in a normal book-vs-book match:
# rated games start from curated, non-standard positions (measured 9-20 plies deep, median 13,
# across this session's 47 real games) rather than the initial position a book is keyed from, so
# a hit requires this specific book to also cover that specific curated line -- measured directly
# against those same 47 real positions before shipping, not assumed: 20/47 (42.6%) covered by
# this book, the best of four candidates tried locally. Read-only lookup, no learning weights
# written back, so thread-unsafe concerns around the ponder thread don't apply. Load is
# wrapped for the same reason as the tablebase above: an exception here would fail the whole
# import, not just this feature.
_BOOK_DIRECTORY: "tempfile.TemporaryDirectory[str] | None" = None


def _open_book() -> "chess.polyglot.MemoryMappedReader | None":
    global _BOOK_DIRECTORY
    path = Path(__file__).resolve().parent / "book" / "codekiddy.bin"
    try:
        if not path.is_file():
            compressed = path.with_suffix(".bin.gz")
            if not compressed.is_file():
                return None
            # The platform directs tempfile to its writable /tmp. Restore the
            # exact Polyglot bytes once during init, then retain its normal mmap
            # reader. Compression changes storage, not book entries or choices.
            _BOOK_DIRECTORY = tempfile.TemporaryDirectory(prefix="chess-book-")
            path = Path(_BOOK_DIRECTORY.name) / "codekiddy.bin"
            with gzip.open(compressed, "rb") as source, path.open("wb") as output:
                shutil.copyfileobj(source, output)
        return chess.polyglot.open_reader(str(path))
    except Exception as exc:
        print(f"book load failed, continuing without it: {exc}", file=sys.stderr)
        return None


_BOOK = _open_book()


def _close_book() -> None:
    if _BOOK is not None:
        _BOOK.close()
    if _BOOK_DIRECTORY is not None:
        _BOOK_DIRECTORY.cleanup()


atexit.register(_close_book)

# The live platform suspends this process outside our turn. The bounded candidate
# disables worker startup below; foreground search and its persistent table remain.
# Legacy helpers remain available for diagnostic comparison with the parent.
_ponder_thread: threading.Thread | None = None
_ponder_stop = threading.Event()

ce.warm_up()
mg.warm_up()
cst.warm_up()
cd.warm_up()


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


def _book_move(board: chess.Board) -> chess.Move | None:
    """The highest-weight Polyglot book move for this exact position, or None if the book is
    absent or doesn't cover it (most positions here, since games start from curated lines a
    book keyed on human play may never have seen -- see the module-level comment). Weight is
    each move's frequency/success in the source games; taking the single best one rather than a
    weighted-random pick keeps this predictable and avoids preferring a rare, riskier try over
    the well-established main line, which is the whole point of using a book as a safety net.

    Any exception here falls back to None (plain search), never propagates: a rare runtime read
    failure on the memory-mapped book file should cost this one move's book lookup, not the
    game -- an uncaught exception on the clock is an instant loss, worse by a wide margin than
    the book simply not firing this once."""
    if _BOOK is None or board.fullmove_number > 20:
        return None
    try:
        best_entry = None
        for entry in _BOOK.find_all(board):
            if best_entry is None or entry.weight > best_entry.weight:
                best_entry = entry
        return best_entry.move if best_entry is not None else None
    except Exception as exc:
        print(f"book probe failed, falling back to search: {exc}", file=sys.stderr)
        return None


def _tablebase_move(
    board: chess.Board, claims: dict[int, int] | None = None,
) -> chess.Move | None:
    """The provably best move by Syzygy WDL/DTZ, or None if this position isn't covered (too
    many pieces, castling rights still held -- Syzygy tables never contain those -- or a probe
    came back unreadable). Never guesses from a partial read: if any candidate move's outcome
    can't be read, the whole position is abandoned back to search rather than trusted halfway.

    Picks the move giving the best reachable result category (win > cursed win > draw > blessed
    loss > loss, all from our side's perspective). Among moves tied on a real win, prefers
    the shortest distance to a pawn move or capture, measured BEFORE our move.
    An immediately zeroing move has distance one; a reversible move adds one to
    the child's absolute DTZ. Checkmate is preferred immediately. Draw claims
    take precedence over history-free tablebase WDL where applicable.

    Any exception falls back to None (plain search): a rare runtime read failure should cost
    this one lookup, not the game, and board.push/pop is wrapped in try/finally so a failure
    mid-loop can never leave the caller's board object corrupted for whatever runs next."""
    if (
        _TABLEBASE is None
        or chess.popcount(board.occupied) > MAX_TABLEBASE_PIECES
        or board.castling_rights
    ):
        return None

    try:
        best_move: chess.Move | None = None
        best_wdl = -3
        best_progress = 0
        for move in board.legal_moves:
            zeroing = board.is_zeroing(move)
            packed = cst.pack_move(move.from_square, move.to_square, move.promotion or 0)
            board.push(move)
            try:
                if board.is_checkmate():
                    return move
                wdl = _TABLEBASE.get_wdl(board)
                dtz = _TABLEBASE.get_dtz(board)
            finally:
                board.pop()
            if wdl is None:
                return None
            our_wdl = -wdl
            claim = claims.get(packed, 0) if claims else 0
            if claim == cd.IMMEDIATE:
                our_wdl = 0
            elif claim == cd.OPPONENT_CAN_DRAW:
                our_wdl = min(our_wdl, 0)
            distance = 1 if zeroing else abs(dtz) + 1 if dtz is not None else 1000
            progress = -distance if our_wdl > 0 else 0
            if best_move is None or (our_wdl, progress) > (best_wdl, best_progress):
                best_move, best_wdl, best_progress = move, our_wdl, progress
        return best_move
    except Exception as exc:
        print(f"tablebase probe failed, falling back to search: {exc}", file=sys.stderr)
        return None


def _record_our_move(board: chess.Board, move: chess.Move) -> None:
    """_GAME_HISTORY is only ever written from the fen the platform hands us -- always our own
    turn to move, per the contract. hash_of_board folds in a turn bit (chess_state.py), so the
    position immediately after we play `move` (opponent to move) hashes to something entirely
    different and would never appear in that dict on its own. Real rated games showed exactly
    this gap in practice: the position that actually recurred to draw three winning games was
    the one right after our own repeated check, which nothing was ever recording. Called once
    per real move we actually return (both here and from the tablebase path), this closes the
    gap by recording that position too, so a real recurrence of it is visible to
    root draw-claim analysis on a later move -- and to the search's own draw detection,
    which seeds `self.seen` from this same dict and had the identical blind spot."""
    irreversible = board.is_irreversible(move)
    board.push(move)
    key = cs.hash_of_board(board)
    board.pop()
    if irreversible:
        _GAME_HISTORY.clear()
    _GAME_HISTORY[key] = _GAME_HISTORY.get(key, 0) + 1


def _start_pondering(board: chess.Board, our_move: chess.Move) -> None:
    # The live runner suspends the process between turns. This bounded candidate
    # avoids startup/cancellation work for a worker with no opponent-time budget.
    # Keep the old implementation in Git as the independently tested reference.
    return


def get_move(fen: str, time_left_ms: int) -> str:
    """Return a legal move in UCI notation.

    fen           the position to move in; your colour is the side to move
    time_left_ms  your clock before this move, in milliseconds
    returns       "e2e4", or "e7e8q" for a promotion

    The process stays alive between your moves, so state you keep on a module or in a
    closure survives to the next call. It does not survive to the next game.
    """
    _stop_pondering()

    # Trusting the platform's own contract here (a valid fen, a position with at least one
    # legal move): if either of these two lines fails there is no legal-move list to fall back
    # to anyway, so nothing downstream could save the game regardless of how defensively it's
    # written -- the safety net below starts right after this, once a fallback move exists.
    board = chess.Board(fen)
    legal_moves = list(board.legal_moves)

    # Recorded unconditionally, even for a forced (single-legal-move) position -- this dict is
    # the real game's position history, and a forced move is still a real position in that
    # history. Splitting get_move into this wrapper and _choose_move once put this update on
    # the wrong side of the single-move early return, silently dropping every forced position
    # from _GAME_HISTORY -- caught by tools/endgame_regression.py regressing on exactly the
    # kind of position (forced king/rook shuffles) most likely to depend on it.
    key = cs.hash_of_board(board)
    if board.halfmove_clock == 0:
        # The opponent just moved a pawn or captured; older positions cannot recur.
        _GAME_HISTORY.clear()
    _GAME_HISTORY[key] = _GAME_HISTORY.get(key, 0) + 1

    if len(legal_moves) == 1:
        _record_our_move(board, legal_moves[0])
        return legal_moves[0].uci()

    try:
        return _choose_move(board, legal_moves, time_left_ms)
    except Exception as exc:
        # An uncaught exception here is an instant game loss (crash = loses that game, per the
        # rules) for a bug that could be anywhere in a genuinely complex pipeline -- search,
        # eval, the tablebase/book probes, time management. Every specific failure mode found
        # this session got its own targeted fix (the tablebase/book wrapping above, the
        # repetition-history fix, ...), but this is the backstop for whatever the *next* one
        # turns out to be: fall back to the first legal move rather than lose the whole game
        # over it. Full traceback to stderr for visibility in the validation log; the move
        # itself is still always legal, just not necessarily good.
        traceback.print_exc(file=sys.stderr)
        print(f"get_move crashed ({exc}); falling back to first legal move", file=sys.stderr)
        _record_our_move(board, legal_moves[0])
        return legal_moves[0].uci()


def _choose_move(board: chess.Board, legal_moves: list[chess.Move], time_left_ms: int) -> str:
    # get_move already recorded this position in _GAME_HISTORY (unconditionally, before the
    # single-legal-move early return) -- this key is only for the TT probe below.
    key = cs.hash_of_board(board)

    start = time.monotonic()
    soft_ms, hard_ms = _time_budget(float(time_left_ms), board.fullmove_number)
    deadline = start + hard_ms / 1000.0
    soft_deadline = start + soft_ms / 1000.0
    _, tt_move_packed = _TT.probe(key, 0, -cs.MATE, cs.MATE, 0)
    tt_move = cs.packed_to_move(tt_move_packed)
    ordered = sorted(legal_moves, key=lambda move: move != tt_move)
    claims, complete = cd.root_claims(
        board, _GAME_HISTORY, ordered,
        start + min(DRAW_SCAN_MAX_SECONDS, hard_ms / 1000 * DRAW_SCAN_BUDGET_FRACTION),
    )
    if claims or not complete:
        print(f"draw_claims={len(claims)} complete={int(complete)} "
              f"ms={(time.monotonic() - start) * 1000:.1f}", file=sys.stderr)

    book_move = _book_move(board)
    if book_move is not None and cst.pack_move(
        book_move.from_square, book_move.to_square, book_move.promotion or 0,
    ) in claims:
        book_move = None
    if book_move is not None:
        print(f"move={book_move.uci()} source=book", file=sys.stderr)
        _record_our_move(board, book_move)
        _start_pondering(board, book_move)
        return book_move.uci()

    tablebase_move = _tablebase_move(board, claims)
    if tablebase_move is not None:
        print(f"move={tablebase_move.uci()} source=tablebase", file=sys.stderr)
        _record_our_move(board, tablebase_move)
        _start_pondering(board, tablebase_move)
        return tablebase_move.uci()

    best_move = tt_move if tt_move in legal_moves else legal_moves[0]

    search = cs.Search(_TT, _GAME_HISTORY)
    search.root_draw_claims = claims
    search.root_repeated_moves = cd.repeated_moves(board, _GAME_HISTORY, legal_moves)
    depth = 1
    completed_depth = 0
    last_score = 0
    while depth <= MAX_SEARCH_DEPTH:
        prev_score = last_score if completed_depth > 0 else None
        try:
            move, score, _ = search.search_root(
                board, depth, deadline, prev_score=prev_score
            )
        except cs.TimeUp:
            break
        best_move = move
        completed_depth = depth
        last_score = score
        if abs(score) >= cs.MATE_THRESHOLD or time.monotonic() >= soft_deadline:
            break
        depth += 1

    # Safe per the rules: stdout is redirected away from the protocol stream before this
    # module is even imported, so print() can never corrupt it. Discarded in rated games,
    # shown in the validation log -- cheap visibility into real games, not just local ones.
    elapsed_ms = (time.monotonic() - start) * 1000.0
    print(
        f"move={best_move.uci()} depth={completed_depth} score={last_score} "
        f"nodes={search.nodes} ms={elapsed_ms:.0f} time_left_ms={time_left_ms}",
        file=sys.stderr,
    )

    _record_our_move(board, best_move)
    _start_pondering(board, best_move)
    return best_move.uci()
