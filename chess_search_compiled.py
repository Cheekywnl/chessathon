"""Compiled recursion for this team's own alpha-beta search.

The pruning, stable move order, evaluation, and replacement rules match the
Python implementation at f666661. Source remains ordinary readable Python;
Numba compiles it during import. No compiled cache or binary is shipped.

Timeouts set an abort flag. Every caller restores its repetition count before
propagating the flag, and no incomplete node updates the table or heuristics.
Only the Python root boundary raises TimeUp. A periodic object-mode clock check
also observes the optional worker cancellation event, with the GIL reacquired.
"""

from __future__ import annotations

import threading
import time
import weakref
from typing import Any

import numpy as np
from numba import njit, objmode, types
from numba.experimental import jitclass  # type: ignore[attr-defined]
from numba.typed import Dict

import chess_eval as ce
import chess_halfkp_int as qi
import chess_movegen as mg
import chess_search_limits as limits
import chess_state as cst

State = cst.RawState
MATE, MATE_THRESHOLD, NO_MOVE = (limits.MATE, limits.MATE_THRESHOLD, limits.NO_MOVE)
CONTEMPT, MAX_PLY = (limits.CONTEMPT, limits.MAX_PLY)
VALUES, LMR = (limits.PIECE_VALUES, limits.LMR_TABLE)
U1 = np.uint64(1)
_STOP_EVENTS: weakref.WeakValueDictionary[int, threading.Event] = weakref.WeakValueDictionary()


def register_stop(event: threading.Event | None) -> int:
    if event is None:
        return 0
    token = id(event)
    _STOP_EVENTS[token] = event
    return token


def should_stop(deadline: float, token: int) -> bool:
    event = _STOP_EVENTS.get(token)
    return time.monotonic() > deadline or (event is not None and event.is_set())


@jitclass(  # type: ignore[no-untyped-call]
    [
        ("keys", types.uint64[::1]),
        ("depths", types.int16[::1]),
        ("scores", types.int32[::1]),
        ("flags", types.int8[::1]),
        ("moves", types.int32[::1]),
        ("mask", types.int64),
    ]
)
class Table:
    def __init__(self, power: int) -> None:
        n = 1 << power
        self.keys = np.zeros(n, dtype=np.uint64)
        self.depths = np.zeros(n, dtype=np.int16)
        self.scores = np.zeros(n, dtype=np.int32)
        self.flags = np.full(n, -1, dtype=np.int8)
        self.moves = np.full(n, -1, dtype=np.int32)
        self.mask = n - 1


spec = [
    ("table", Table.class_type.instance_type),  # type: ignore[attr-defined]
    ("seen", types.DictType(types.uint64, types.int64)),  # type: ignore[no-untyped-call]
    ("killers", types.int64[:, ::1]),
    ("history", types.int64[:, :, ::1]),
    ("nodes", types.int64),
    ("limit", types.int64),
    ("params", types.int32[::1]),
    ("deadline", types.float64),
    ("stop_token", types.int64),
    ("aborted", types.boolean),
    ("w1", types.int16[:, ::1]),
    ("b1", types.int16[::1]),
    ("w2", types.int8[:, ::1]),
    ("b2", types.int32[::1]),
    ("w3", types.int8[:, ::1]),
    ("b3", types.int32[::1]),
    ("w4", types.int8[::1]),
    ("b4", types.int64),
    ("scale2", types.int64),
    ("scale3", types.int64),
    ("divisor", types.float64),
    ("blend", types.int64),
]


@jitclass(spec)  # type: ignore[no-untyped-call]
class Context:
    def __init__(
        self,
        table: Any,
        seen: Any,
        params: np.ndarray,
        w1: np.ndarray,
        b1: np.ndarray,
        w2: np.ndarray,
        b2: np.ndarray,
        w3: np.ndarray,
        b3: np.ndarray,
        w4: np.ndarray,
        b4: int,
        s2: int,
        s3: int,
        div: float,
        blend: int,
    ) -> None:
        self.table = table
        self.seen = seen
        self.killers = np.full((MAX_PLY, 2), -1, dtype=np.int64)
        self.history = np.zeros((2, 64, 64), dtype=np.int64)
        self.nodes, self.limit = (0, 2**62)
        self.deadline = 0.0
        self.stop_token = 0
        self.aborted = False
        self.params = params
        self.w1, self.b1, self.w2, self.b2 = (w1, b1, w2, b2)
        self.w3, self.b3, self.w4, self.b4 = (w3, b3, w4, b4)
        self.scale2, self.scale3, self.divisor, self.blend = (s2, s3, div, blend)


@njit(cache=False)
def key_of(s: State) -> np.uint64:
    return cst.zobrist_hash(
        s[0],
        s[1],
        s[2],
        s[3],
        s[4],
        s[5],
        s[6],
        s[7],
        s[8],
        s[9],
        s[10],
        cst.ZOBRIST_PIECE_SQUARE,
        cst.ZOBRIST_CASTLING,
        cst.ZOBRIST_EP_FILE,
        cst.ZOBRIST_TURN,
    )


@njit(cache=False)
def legal(s: State) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    return mg.generate_legal_moves_bb(
        s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[8], s[9], s[10]
    )


@njit(cache=False)
def check(s: State) -> bool:
    own = s[6] if s[8] else s[7]
    king = cst._lsb_index(s[5] & own)
    return mg.is_square_attacked(king, not s[8], s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7])


@njit(cache=False)
def after(s: State, key: np.uint64, f: int, t: int, p: int) -> tuple[State, np.uint64, bool]:
    return cst.make_move_info(s, key, f, t, p)


@njit(cache=False)
def capture(s: State, f: int, t: int) -> bool:
    return bool((s[6] | s[7]) & U1 << np.uint64(t) or (s[0] & U1 << np.uint64(f) and t == s[10]))


@njit(cache=False)
def piece_at(s: State, square: int) -> int:
    mask = U1 << np.uint64(square)
    pieces = (s[0], s[1], s[2], s[3], s[4], s[5])
    for i in range(6):
        if pieces[i] & mask:
            return i + 1
    return 6


@njit(cache=False)
def victim_at(s: State, f: int, t: int) -> int:
    if t == s[10] and s[0] & U1 << np.uint64(f):
        return 1
    kind = piece_at(s, t)
    return 1 if kind == 6 else kind


@njit(cache=False)
def see(s: State, f: int, t: int) -> int:
    return cst.see_raw(s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], f, t, s[10], s[8], VALUES)


@njit(cache=False)
def bare(s: State) -> bool:
    major = s[3] | s[4]
    nonking = s[0] | s[1] | s[2] | major
    return bool(major & s[6] and (not nonking & s[7])) or bool(
        major & s[7] and (not nonking & s[6])
    )


@njit(cache=False)
def insufficient_side(own: np.uint64, other: np.uint64, s: State) -> bool:
    if own & (s[0] | s[3] | s[4]):
        return False
    if own & s[1]:
        return ce._popcount(own) <= 2 and (not other & ~s[5] & ~s[4])
    if own & s[2]:
        same = not s[2] & np.uint64(12273903644374837845) or not s[2] & np.uint64(
            6172840429334713770
        )
        return same and (not s[0]) and (not s[1])
    return True


@njit(cache=False)
def is_draw(s: State, key: np.uint64, ctx: Any) -> bool:
    if s[11] >= 100 or ctx.seen.get(key, 0) >= 3:
        return not (check(s) and legal(s)[3] == 0)
    if s[11] == 99:
        f, t, p, n = legal(s)
        for i in range(n):
            child, _, _ = after(s, key, int(f[i]), int(t[i]), int(p[i]))
            if child[11] >= 100 and legal(child)[3] > 0:
                return True
    if ce._popcount(s[6] | s[7]) <= 6:
        return insufficient_side(s[6], s[7], s) and insufficient_side(s[7], s[6], s)
    return False


@njit(cache=False)
def draw_score(ply: int) -> int:
    return -CONTEMPT if ply % 2 == 0 else CONTEMPT


@njit(cache=False)
def evaluate(s: State, mobility: int, ctx: Any) -> int:
    wc = int(bool(s[9] & U1)) + int(bool(s[9] & np.uint64(128)))
    bc = int(bool(s[9] & np.uint64(1 << 56))) + int(bool(s[9] & np.uint64(1 << 63)))
    positional = ce.evaluate(
        s[0],
        s[1],
        s[2],
        s[3],
        s[4],
        s[5],
        s[6],
        s[7],
        ce.MG_PST,
        ce.EG_PST,
        ce.PASSED_MASK_WHITE,
        ce.PASSED_MASK_BLACK,
        ce.FILE_MASK,
        ce.ADJACENT_FILE_MASK,
        wc,
        bc,
        ctx.params,
    )
    classical = (int(positional) if s[8] else -int(positional)) + int(
        ctx.params[ce.P_MOBILITY]
    ) * mobility
    nonking = s[0] | s[1] | s[2] | s[3] | s[4]
    if (
        ctx.blend <= 0
        or ce._popcount(s[6] | s[7]) < 8
        or (not nonking & s[6])
        or (not nonking & s[7])
    ):
        return classical
    neural = qi.evaluate(
        s[0],
        s[1],
        s[2],
        s[3],
        s[4],
        s[5],
        s[6],
        s[7],
        s[8],
        ctx.w1,
        ctx.b1,
        ctx.w2,
        ctx.b2,
        ctx.w3,
        ctx.b3,
        ctx.w4,
        ctx.b4,
        ctx.scale2,
        ctx.scale3,
        ctx.divisor,
    )
    bonus = int(ce.mop_up_bonus(s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], ctx.params))
    neural = max(-20000, min(20000, neural)) + (bonus if s[8] else -bonus)
    return (
        neural
        if ctx.blend >= 100
        else round((classical * (100 - ctx.blend) + neural * ctx.blend) / 100)
    )


@njit(cache=False)
def pack(f: int, t: int, p: int) -> int:
    return int(f | t << 6 | p << 12)


@njit(cache=False)
def unpack(move: int) -> tuple[int, int, int]:
    return (move & 63, move >> 6 & 63, move >> 12 & 7)


@njit(cache=False)
def ordered(
    s: State,
    f: np.ndarray,
    t: np.ndarray,
    p: np.ndarray,
    n: int,
    ctx: Any,
    ttmove: int,
    ply: int,
    qmode: bool,
    incheck: bool,
) -> tuple[np.ndarray, np.ndarray, int]:
    moves = np.empty(n, dtype=np.int64)
    captures = np.empty(n, dtype=np.bool_)
    scores = np.empty(n, dtype=np.int64)
    used = 0
    kp = min(ply, MAX_PLY - 1)
    for i in range(n):
        ff, tt, pp = (int(f[i]), int(t[i]), int(p[i]))
        move = pack(ff, tt, pp)
        cap = capture(s, ff, tt)
        if qmode:
            if not incheck and (not pp) and (not cap):
                continue
            score = (
                200000
                if pp
                else VALUES[victim_at(s, ff, tt) - 1] * 10 - VALUES[piece_at(s, ff) - 1]
            )
        elif move == ttmove:
            score = 1000000
        elif cap:
            exchange = see(s, ff, tt)
            score = 100000 + exchange if exchange >= 0 else -100000 + exchange
        elif move == ctx.killers[kp, 0]:
            score = 90000
        elif move == ctx.killers[kp, 1]:
            score = 89000
        else:
            score = ctx.history[int(s[8]), ff, tt]
        moves[used], captures[used], scores[used] = (move, cap, score)
        used += 1
    # Stable insertion sort preserves legal-generation order when scores tie.
    for i in range(1, used):
        move, cap, score = (moves[i], captures[i], scores[i])
        j = i - 1
        while j >= 0 and scores[j] < score:
            moves[j + 1], captures[j + 1], scores[j + 1] = (moves[j], captures[j], scores[j])
            j -= 1
        moves[j + 1], captures[j + 1], scores[j + 1] = (move, cap, score)
    return (moves, captures, used)


@njit(cache=False)
def probe(
    ctx: Any, key: np.uint64, depth: int, alpha: int, beta: int, ply: int
) -> tuple[int, int, bool]:
    i = np.int64(key & np.uint64(ctx.mask))
    if ctx.flags[i] < 0 or ctx.keys[i] != key:
        return (0, NO_MOVE, False)
    move = int(ctx.moves[i])
    if ctx.depths[i] >= depth:
        score = int(ctx.scores[i])
        if score >= MATE_THRESHOLD:
            score -= ply
        elif score <= -MATE_THRESHOLD:
            score += ply
        flag = ctx.flags[i]
        if flag == 0 or (flag == 1 and score >= beta) or (flag == 2 and score <= alpha):
            return (score, move, True)
    return (0, move, False)


@njit(cache=False)
def put(ctx: Any, key: np.uint64, depth: int, score: int, flag: int, move: int, ply: int) -> None:
    i = np.int64(key & np.uint64(ctx.mask))
    if ctx.flags[i] < 0 or ctx.keys[i] == key or ctx.depths[i] <= depth:
        if score >= MATE_THRESHOLD:
            score += ply
        elif score <= -MATE_THRESHOLD:
            score -= ply
        ctx.keys[i], ctx.depths[i], ctx.scores[i], ctx.flags[i], ctx.moves[i] = (
            key,
            depth,
            score,
            flag,
            move,
        )


@njit(cache=False)
def tick(ctx: Any) -> bool:
    ctx.nodes += 1
    if ctx.nodes > ctx.limit:
        ctx.aborted = True
    elif ctx.nodes % limits.NODES_PER_TIME_CHECK == 0:
        deadline, token = (ctx.deadline, ctx.stop_token)
        with objmode(expired="boolean"):
            expired = should_stop(deadline, token)
        if expired:
            ctx.aborted = True
    return bool(ctx.aborted)


@njit(nogil=True)
def qsearch(
    s: State, alpha: int, beta: int, ply: int, key: np.uint64, incheck: bool, ctx: Any
) -> int:
    if tick(ctx):
        return 0
    if is_draw(s, key, ctx):
        return draw_score(ply)
    f, t, p, n = legal(s)
    if n == 0:
        return -(MATE - ply) if incheck else draw_score(ply)
    stand = 0
    if incheck:
        best = -MATE - 1
    else:
        stand = evaluate(s, n, ctx)
        if stand >= beta:
            return stand
        alpha = max(alpha, stand)
        best = stand
    moves, caps, count = ordered(s, f, t, p, n, ctx, NO_MOVE, ply, True, incheck)
    for i in range(count):
        ff, tt, pp = unpack(moves[i])
        if not incheck and caps[i] and (not pp):
            victim = victim_at(s, ff, tt)
            if stand + VALUES[victim - 1] + limits.DELTA_MARGIN <= alpha:
                continue
            if see(s, ff, tt) < 0:
                continue
        child, childkey, childcheck = after(s, key, ff, tt, pp)
        ctx.seen[childkey] = ctx.seen.get(childkey, 0) + 1
        score = -qsearch(child, -beta, -alpha, ply + 1, childkey, childcheck, ctx)
        ctx.seen[childkey] -= 1
        if ctx.aborted:
            return 0
        best = max(best, score)
        alpha = max(alpha, score)
        if alpha >= beta:
            break
    return best


@njit(nogil=True)
def negamax(
    s: State,
    depth: int,
    alpha: int,
    beta: int,
    ply: int,
    allow_null: bool,
    key: np.uint64,
    incheck: bool,
    ctx: Any,
) -> int:
    if tick(ctx):
        return 0
    if is_draw(s, key, ctx):
        return draw_score(ply)
    score, ttmove, hit = probe(ctx.table, key, depth, alpha, beta, ply)
    if hit:
        return score
    if depth <= 0:
        return qsearch(s, alpha, beta, ply, key, incheck, ctx)
    f, t, p, n = legal(s)
    if n == 0:
        return -(MATE - ply) if incheck else draw_score(ply)
    static = 0
    if not incheck:
        static = evaluate(s, n, ctx)
        if depth <= limits.REVERSE_FUTILITY_DEPTH and abs(beta) < MATE_THRESHOLD:
            margin = limits.REVERSE_FUTILITY_MARGIN_PER_PLY * depth
            if static - margin >= beta:
                return static - margin
    own = s[6] if s[8] else s[7]
    if allow_null and (not incheck) and (depth >= 3) and own & ~s[0] & ~s[5]:
        null = (s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], not s[8], s[9], -1, s[11] + 1)
        score = -negamax(
            null,
            depth - 1 - limits.NULL_MOVE_REDUCTION,
            -beta,
            -beta + 1,
            ply + 1,
            False,
            key_of(null),
            check(null),
            ctx,
        )
        if ctx.aborted:
            return 0
        if score >= beta:
            return score
    moves, caps, count = ordered(s, f, t, p, n, ctx, ttmove, ply, False, incheck)
    original_alpha = alpha
    best, bestmove = (-MATE - 1, NO_MOVE)
    kp = min(ply, MAX_PLY - 1)
    prunable = not bare(s)
    for i in range(count):
        move, cap = (moves[i], caps[i])
        ff, tt, pp = unpack(move)
        extension = 1 if incheck else 0
        killer = move == ctx.killers[kp, 0] or move == ctx.killers[kp, 1]
        child, childkey, childcheck = after(s, key, ff, tt, pp)
        if (
            prunable
            and depth <= limits.LATE_MOVE_PRUNING_DEPTH
            and (i >= limits.LATE_MOVE_PRUNING_BASE + limits.LATE_MOVE_PRUNING_PER_DEPTH * depth)
            and (not extension)
            and (not cap)
            and (not pp)
            and (not killer)
            and (best > -MATE_THRESHOLD)
            and (not childcheck)
        ):
            continue
        if (
            prunable
            and (not incheck)
            and (i >= 1)
            and (depth <= limits.FUTILITY_DEPTH)
            and (not extension)
            and (not cap)
            and (not pp)
            and (not killer)
            and (abs(alpha) < MATE_THRESHOLD)
            and (static + limits.FUTILITY_MARGIN_PER_PLY * depth <= alpha)
            and (not childcheck)
        ):
            continue
        reduction = 0
        if (
            prunable
            and depth >= 3
            and (i >= 3)
            and (not extension)
            and (not cap)
            and (not pp)
            and (not killer)
        ):
            reduction = min(
                int(LMR[min(depth, limits._LMR_MAX_DEPTH), min(i, limits._LMR_MAX_MOVE_INDEX)]),
                depth - 1,
            )
        ctx.seen[childkey] = ctx.seen.get(childkey, 0) + 1
        if i == 0:
            score = -negamax(
                child,
                depth - 1 + extension,
                -beta,
                -alpha,
                ply + 1,
                True,
                childkey,
                childcheck,
                ctx,
            )
        else:
            score = -negamax(
                child,
                depth - 1 + extension - reduction,
                -alpha - 1,
                -alpha,
                ply + 1,
                True,
                childkey,
                childcheck,
                ctx,
            )
            if not ctx.aborted and score > alpha:
                score = -negamax(
                    child,
                    depth - 1 + extension,
                    -beta,
                    -alpha,
                    ply + 1,
                    True,
                    childkey,
                    childcheck,
                    ctx,
                )
        ctx.seen[childkey] -= 1
        if ctx.aborted:
            return 0
        if score > best:
            best, bestmove = (score, move)
        alpha = max(alpha, score)
        if alpha >= beta:
            if not cap:
                if move != ctx.killers[kp, 0]:
                    ctx.killers[kp, 1] = ctx.killers[kp, 0]
                    ctx.killers[kp, 0] = move
                ctx.history[int(s[8]), ff, tt] += depth * depth
            break
    flag = 2 if best <= original_alpha else 1 if best >= beta else 0
    put(ctx.table, key, depth, best, flag, bestmove, ply)
    return best


def create_context(
    table: Any,
    history: dict[int, int],
    params: np.ndarray,
    weights: qi.QuantizedWeights,
    blend: int,
    stop: threading.Event | None,
) -> Any:
    seen = Dict.empty(key_type=types.uint64, value_type=types.int64)
    for key, count in history.items():
        seen[np.uint64(key)] = np.int64(count)
    ctx = Context(
        table,
        seen,
        params,
        weights.w1,
        weights.b1,
        weights.w2,
        weights.b2,
        weights.w3,
        weights.b3,
        weights.w4,
        weights.b4,
        weights.scale2,
        weights.scale3,
        weights.output_divisor,
        blend,
    )
    ctx.stop_token = register_stop(stop)
    return ctx
