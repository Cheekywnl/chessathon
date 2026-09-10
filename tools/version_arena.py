"""Real-protocol A/B arena between two agent directories (e.g. two git commits/worktrees of this
repo, or a candidate vs a baseline branch) across a small set of diverse opening positions, not
just the bare starting position.

harness/arena.py always starts from chess.STARTING_FEN. That's fine when testing against a
baseline agent whose own search is itself a source of game-to-game variety, but for two
deterministic engines (same hardware, same time budget, no RNG) it means "20 games" is really
just two distinct game trajectories (candidate-as-white, candidate-as-black) each replayed ten
times -- real-clock timing jitter adds a little variance, but not enough to call it 20 independent
trials. Playing from several different, well-known opening positions instead gives every game its
own actual decision tree, and it matches the platform's own rule that rated games start from
curated non-startpos positions rather than always the same one.

Deliberately does not import or modify anything in harness/ except the two functions actually
being reused (Agent process spawning via harness.sandbox.local, refereeing via
harness.referee.play_match) -- both already accept a start_fen, so this is pure composition, not
a fork of the real protocol. Every game is a real subprocess speaking the real stdin/stdout
protocol, so crash/illegal/flag handling is exactly what the platform (and harness/arena.py)
exercises, not the in-process shortcut tools/fast_arena.py takes for eval-param iteration.

Usage:
    uv run python -m tools.version_arena --agent . --opponent /path/to/old-worktree \
        --base-ms 20000 --increment-ms 300
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from functools import partial
from pathlib import Path

import chess

from harness.referee import FAILED_TERMINATIONS, play_match
from harness.rules import PLY_CAP
from harness.sandbox import Agent, local
from tools.platform_agent import local as platform_local

FAST_BASE_MS = 20_000
FAST_INCREMENT_MS = 300

# A small, diverse set of well-known, roughly balanced opening lines -- not the platform's own
# (undisclosed) curated set, just enough variety that "N games" actually explores N different
# decision trees instead of replaying the same two games repeatedly. Each entry is a sequence of
# SAN moves from the standard start position; the FEN is derived by replaying them, not
# hand-transcribed, so there is no risk of a typo'd FEN silently testing the wrong position.
OPENING_LINES: dict[str, list[str]] = {
    "startpos": [],
    "italian": ["e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5"],
    "ruy_lopez": ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6"],
    "sicilian_najdorf": ["e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6", "Nc3", "a6"],
    "french": ["e4", "e6", "d4", "d5", "Nc3", "Nf6"],
    "caro_kann": ["e4", "c6", "d4", "d5", "Nc3", "dxe4", "Nxe4", "Bf5"],
    "queens_gambit_declined": ["d4", "d5", "c4", "e6", "Nc3", "Nf6"],
    "kings_indian": ["d4", "Nf6", "c4", "g6", "Nc3", "Bg7", "e4", "d6"],
    "english": ["c4", "e5", "Nc3", "Nf6", "Nf3", "Nc6"],
    "nimzo_indian": ["d4", "Nf6", "c4", "e6", "Nc3", "Bb4"],
}


def _fen_for(moves: list[str]) -> str:
    board = chess.Board()
    for san in moves:
        board.push_san(san)
    return board.fen()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score one agent directory against another across diverse openings."
    )
    parser.add_argument("--agent", type=Path, default=Path("."))
    parser.add_argument("--opponent", type=Path, required=True)
    parser.add_argument("--base-ms", type=int, default=FAST_BASE_MS)
    parser.add_argument("--increment-ms", type=int, default=FAST_INCREMENT_MS)
    parser.add_argument("--ply-cap", type=int, default=PLY_CAP)
    parser.add_argument("--platform-cpu", type=int,
                        help="Pin to this core and suspend each agent outside its turn")
    parser.add_argument("--engine-python", help="Pinned CPU interpreter for platform subprocesses")
    parser.add_argument("--jsonl", type=Path, help="Write exact game records and PGNs")
    parser.add_argument(
        "--openings",
        nargs="+",
        default=list(OPENING_LINES),
        choices=list(OPENING_LINES),
        help="Which named openings to play (default: all).",
    )
    arguments = parser.parse_args()

    factory: Callable[[Path], Agent] = local
    if arguments.platform_cpu is not None:
        factory = partial(platform_local, cpu=arguments.platform_cpu,
                          python=arguments.engine_python)

    agent = arguments.agent.resolve()
    opponent = arguments.opponent.resolve()
    wins = draws = losses = 0
    terminations: dict[str, int] = {}
    games = len(arguments.openings) * 2

    game_num = 0
    for name in arguments.openings:
        fen = _fen_for(OPENING_LINES[name])
        for agent_plays_white in (True, False):
            game_num += 1
            white, black = (agent, opponent) if agent_plays_white else (opponent, agent)
            white_process, black_process = factory(white), factory(black)
            outcome = play_match(
                white_process,
                black_process,
                arguments.base_ms,
                arguments.increment_ms,
                ply_cap=arguments.ply_cap,
                start_fen=fen,
            )
            terminations[outcome.termination] = terminations.get(outcome.termination, 0) + 1
            if arguments.jsonl:
                arguments.jsonl.parent.mkdir(parents=True, exist_ok=True)
                with arguments.jsonl.open("a", encoding="utf8") as log:
                    log.write(json.dumps({"game": game_num, "opening": name,
                                          "agent_white": agent_plays_white,
                                          "base_ms": arguments.base_ms,
                                          "increment_ms": arguments.increment_ms,
                                          "result": outcome.result,
                                          "termination": outcome.termination, "pgn": outcome.pgn,
                                          "white_log": white_process.stderr_tail,
                                          "black_log": black_process.stderr_tail}) + "\n")
            if outcome.termination in FAILED_TERMINATIONS or outcome.result == "void":
                print(white_process.stderr_tail, black_process.stderr_tail)
                raise SystemExit(f"Invalid strength test: game {game_num} {outcome.termination}")
            if outcome.result == "draw":
                draws += 1
            elif (outcome.result == "white") == agent_plays_white:
                wins += 1
            else:
                losses += 1
            side = "white" if agent_plays_white else "black"
            print(
                f"game {game_num}/{games} [{name}, agent={side}]: "
                f"{outcome.result} by {outcome.termination}",
                flush=True,
            )

    score = (wins + draws / 2) / games
    n_openings = len(arguments.openings)
    print(f"\n{arguments.agent} vs {arguments.opponent} over {games} games, {n_openings} openings")
    print(f"+{wins} ={draws} -{losses}, score {score:.1%}")
    print("terminations: " + ", ".join(f"{n} {c}" for n, c in terminations.items()))
    broken = {n: c for n, c in terminations.items() if n in FAILED_TERMINATIONS}
    if broken:
        print(
            "\nwarning: non-clean terminations present -- treat this result as a methodology "
            "artifact, not a pure strength signal: "
            + ", ".join(f"{k} {v}" for k, v in broken.items())
        )


if __name__ == "__main__":
    main()
