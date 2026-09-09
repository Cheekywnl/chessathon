"""In-process A/B arena for eval-param changes: two engine identities (same search and move
generation code, different chess_eval params arrays) play each other directly through
chess_search.Search, in one Python process.

The harness's own arena (harness/arena.py) is the ground truth for protocol fidelity, but it
pays a subprocess spawn and a full numba JIT re-warm on every single game -- fine for validating
the real protocol, expensive for iterating on an eval change. This tool warms every jitted
function exactly once and then plays every game as plain in-process function calls, so a 40-game
run costs one warm-up instead of eighty. It is not a replacement for `make arena` / `make gate`:
those exercise the real subprocess protocol (crash/illegal/flag handling, stdout isolation) that
this tool bypasses entirely by calling the search directly. Use this for "is engine change A
stronger than engine change B", and the harness for "does this still speak the real protocol".

Time management is agent.py's real `_time_budget` formula (imported, not copied, so this can
never silently drift from what actually ships), applied per side with its own persistent
TranspositionTable and per-game position history -- exactly the lifetime agent.py gives them in
a real game (fresh per game, alive across moves within it). See AGENTS.md and
feedback-test-realistic-constraints in project memory: always A/B under a clock close to the real
one, and judge on full games, not single positions.

Usage:
    uv run python -m tools.fast_arena --games 40 \
        --zero P_ROOK_BEHIND_PASSED_MG P_ROOK_BEHIND_PASSED_EG
    uv run python -m tools.fast_arena --selfcheck --games 10   # sanity: should land near 50%
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field

import chess
import numpy as np

import agent
import chess_eval as ce
import chess_movegen as mg
import chess_search as cs
import chess_state as cst
from harness.rules import PLY_CAP

MAX_SEARCH_DEPTH = 64
# agent.py uses size_power=21 (~1 GB fully populated) because it gets the real 2 GB sandbox to
# itself for a single game. Here, two engines' TTs are alive at once, freshly allocated every
# game, on a dev machine shared with whatever else is running -- 21 measurably drove this
# machine into swap during real use (a 40-game run stalled to ~1 game/90min under heavy system
# memory pressure). 17 is what actually got validated locally before that was diagnosed; nodes
# at a 20s-per-side budget don't come close to filling even that, so this costs no real fidelity.
TT_SIZE_POWER = 17
FAILED_TERMINATIONS = frozenset({"illegal", "flag"})


@dataclass
class Engine:
    name: str
    params: np.ndarray
    tt: cs.TranspositionTable = field(
        default_factory=lambda: cs.TranspositionTable(TT_SIZE_POWER)
    )
    history: dict[int, int] = field(default_factory=dict)

    def fresh_for_new_game(self) -> Engine:
        return Engine(self.name, self.params)


def think(
    engine: Engine, board: chess.Board, time_left_ms: float, fullmove_number: int
) -> tuple[chess.Move, float]:
    legal = list(board.legal_moves)
    if len(legal) == 1:
        return legal[0], 0.0

    soft_ms, hard_ms = agent._time_budget(time_left_ms, fullmove_number)
    start = time.monotonic()
    deadline = start + hard_ms / 1000.0
    soft_deadline = start + soft_ms / 1000.0

    search = cs.Search(engine.tt, engine.history, params=engine.params)
    best_move, completed, last_score, depth = legal[0], 0, 0, 1
    while depth <= MAX_SEARCH_DEPTH:
        prev_score = last_score if completed > 0 else None
        try:
            move, score, _ = search.search_root(board, depth, deadline, prev_score=prev_score)
        except cs.TimeUp:
            break
        best_move, completed, last_score = move, depth, score
        if abs(score) >= cs.MATE_THRESHOLD or time.monotonic() >= soft_deadline:
            break
        depth += 1

    elapsed_ms = (time.monotonic() - start) * 1000.0
    return best_move, elapsed_ms


def play_game(
    white: Engine, black: Engine, base_ms: float, increment_ms: float, ply_cap: int
) -> tuple[str, str]:
    board = chess.Board()
    clock = {chess.WHITE: base_ms, chess.BLACK: base_ms}
    engines = {chess.WHITE: white, chess.BLACK: black}

    while True:
        outcome = board.outcome(claim_draw=True)
        if outcome is not None:
            result = "draw" if outcome.winner is None else ("white" if outcome.winner else "black")
            return result, outcome.termination.name.lower()
        if len(board.move_stack) >= ply_cap:
            return "draw", "ply_cap"

        mover = board.turn
        engine = engines[mover]
        key = cs.hash_of_board(board)
        engine.history[key] = engine.history.get(key, 0) + 1

        move, elapsed_ms = think(engine, board, clock[mover], board.fullmove_number)
        clock[mover] -= elapsed_ms
        if clock[mover] < 0:
            return ("black" if mover == chess.WHITE else "white"), "flag"
        if move not in board.legal_moves:
            return ("black" if mover == chess.WHITE else "white"), "illegal"

        board.push(move)
        clock[mover] += increment_ms


def run(
    candidate: Engine,
    baseline: Engine,
    games: int,
    base_ms: float,
    increment_ms: float,
    ply_cap: int,
) -> None:
    candidate_score = 0.0
    terminations: dict[str, int] = {}
    for game in range(games):
        candidate_is_white = game % 2 == 0
        white_src = candidate if candidate_is_white else baseline
        black_src = baseline if candidate_is_white else candidate
        white, black = white_src.fresh_for_new_game(), black_src.fresh_for_new_game()
        result, termination = play_game(white, black, base_ms, increment_ms, ply_cap)
        terminations[termination] = terminations.get(termination, 0) + 1

        if result == "draw":
            candidate_score += 0.5
            outcome = "="
        elif (result == "white") == candidate_is_white:
            candidate_score += 1.0
            outcome = "+"
        else:
            outcome = "-"
        running = candidate_score / (game + 1)
        print(
            f"game {game + 1}/{games}: {outcome} ({result} by {termination}), "
            f"candidate as {'white' if candidate_is_white else 'black'}, running {running:.1%}",
            flush=True,
        )

    score = candidate_score / games
    print(f"\n{candidate.name} vs {baseline.name} over {games} games")
    print(f"candidate score {score:.1%} ({candidate_score:.1f}/{games})")
    print("terminations: " + ", ".join(f"{name} {count}" for name, count in terminations.items()))
    broken = {name: count for name, count in terminations.items() if name in FAILED_TERMINATIONS}
    if broken:
        print(
            "\nwarning: non-clean terminations present -- treat this result as a methodology "
            "artifact, not a strength signal: " + ", ".join(f"{k} {v}" for k, v in broken.items()),
            file=sys.stderr,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="In-process A/B arena for chess_eval params.")
    parser.add_argument("--games", type=int, default=40)
    parser.add_argument("--base-ms", type=float, default=20_000)
    parser.add_argument("--increment-ms", type=float, default=300)
    parser.add_argument("--ply-cap", type=int, default=PLY_CAP)
    parser.add_argument(
        "--zero",
        nargs="+",
        default=[],
        metavar="P_NAME",
        help="chess_eval param names to zero for the baseline (candidate keeps DEFAULT_PARAMS).",
    )
    parser.add_argument(
        "--selfcheck",
        action="store_true",
        help="Ignore --zero; play DEFAULT_PARAMS against itself. Score should land near 50%.",
    )
    arguments = parser.parse_args()

    ce.warm_up()
    mg.warm_up()
    cst.warm_up()

    candidate_params = ce.DEFAULT_PARAMS.copy()
    if arguments.selfcheck:
        baseline_params = ce.DEFAULT_PARAMS.copy()
        candidate_name, baseline_name = "self-a", "self-b"
    else:
        baseline_params = ce.DEFAULT_PARAMS.copy()
        for name in arguments.zero:
            baseline_params[getattr(ce, name)] = 0
        candidate_name = "candidate (DEFAULT_PARAMS)"
        baseline_name = f"baseline (zeroed: {', '.join(arguments.zero) or 'none'})"

    candidate = Engine(candidate_name, candidate_params)
    baseline = Engine(baseline_name, baseline_params)
    run(
        candidate,
        baseline,
        arguments.games,
        arguments.base_ms,
        arguments.increment_ms,
        arguments.ply_cap,
    )


if __name__ == "__main__":
    main()
