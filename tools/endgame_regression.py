"""Plays out known-hard technical endgames through the real agent.get_move (so the repetition
backstop, tablebase probe, and everything else in agent.py is actually exercised, not bypassed
the way fast_arena.py and wac_test.py deliberately do for speed) against a plain-search
opponent, and asserts a win -- not a draw -- within a generous ply budget.

These are regression positions, not a strength benchmark: each one is here because the engine
previously failed it in a documented, specific way (see POSITIONS below). A pass proves the
specific failure hasn't come back; a full pass here says nothing about general endgame play the
way the WAC suite or fast_arena's A/B testing do.

Usage:
    uv run python -m tools.endgame_regression
"""

from __future__ import annotations

import time

import chess

import agent
import chess_eval as ce
import chess_movegen as mg
import chess_search as cs
import chess_state as cst

TT_SIZE_POWER = 18
PLY_BUDGET = 120
THINK_SECONDS = 4.0

# (name, fen, winning side, why this position is here)
POSITIONS: list[tuple[str, str, chess.Color, str]] = [
    (
        "Lucena position",
        "1K1k4/1P6/8/8/8/8/r7/2R5 w - - 0 1",
        chess.WHITE,
        "The contempt commit (e73bfb9) explicitly flagged this as unfixed: 'needs the actual "
        "bridge technique, not a small scoring nudge.' Confirmed independently before this file "
        "existed: plain search (no agent.py) draws by repetition at ply 13 every time. The "
        "repetition backstop (471a3fb, fixed in bbfc5bb) resolves it as a side effect -- not "
        "because the engine knows the bridge technique by name, but because it refuses the "
        "draw and keeps trying alternatives until one actually wins.",
    ),
]


def _opponent_move(
    board: chess.Board, tt: cs.TranspositionTable, history: dict[int, int]
) -> chess.Move:
    key = cs.hash_of_board(board)
    history[key] = history.get(key, 0) + 1
    search = cs.Search(tt, history)
    deadline = time.monotonic() + THINK_SECONDS
    best_move, completed, last_score, depth = next(iter(board.legal_moves)), 0, 0, 1
    while depth <= 40:
        prev_score = last_score if completed > 0 else None
        try:
            move, score, _ = search.search_root(board, depth, deadline, prev_score=prev_score)
        except cs.TimeUp:
            break
        best_move, completed, last_score = move, depth, score
        if abs(score) >= cs.MATE_THRESHOLD:
            break
        depth += 1
    after_key = cs.hash_of_board(_after(board, best_move))
    history[after_key] = history.get(after_key, 0) + 1
    return best_move


def _after(board: chess.Board, move: chess.Move) -> chess.Board:
    copy = board.copy(stack=False)
    copy.push(move)
    return copy


def play(name: str, fen: str, winning_side: chess.Color, why: str) -> bool:
    board = chess.Board(fen)
    agent._TT = cs.TranspositionTable(size_power=TT_SIZE_POWER)
    agent._GAME_HISTORY = {}
    opponent_tt = cs.TranspositionTable(size_power=TT_SIZE_POWER)
    opponent_history: dict[int, int] = {}

    for _ in range(PLY_BUDGET):
        outcome = board.outcome(claim_draw=True)
        if outcome is not None:
            won = outcome.winner == winning_side
            print(f"{name}: {'PASS' if won else 'FAIL'} ({outcome.termination.name.lower()})")
            if not won:
                print(f"  {why}")
            return won
        if board.turn == winning_side:
            move = chess.Move.from_uci(agent.get_move(board.fen(), 20_000))
        else:
            move = _opponent_move(board, opponent_tt, opponent_history)
        board.push(move)

    print(f"{name}: FAIL (no result within {PLY_BUDGET} plies)")
    print(f"  {why}")
    return False


def main() -> None:
    ce.warm_up()
    mg.warm_up()
    cst.warm_up()

    results = [play(*position) for position in POSITIONS]
    passed = sum(results)
    print(f"\n{passed}/{len(results)} endgame regression positions passed")
    if passed < len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
