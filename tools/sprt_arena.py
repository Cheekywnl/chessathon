"""SPRT (Sequential Probability Ratio Test) A/B arena -- the actual methodology real engine
developers use to validate changes (fishtest, OpenBench, cutechess-cli all implement this),
rather than committing to a fixed game count and eyeballing a confidence interval the way
tools/version_arena.py's plain result does.

Runs games one at a time between two agent directories (real subprocess protocol, same
harness.referee.play_match / harness.sandbox.local primitives version_arena.py uses, and the
same diverse-opening set so two deterministic engines still get real game-to-game variety) and
recomputes a log-likelihood ratio after every game. Stops as soon as there's enough evidence to
accept H1 (candidate is at least elo1 stronger than baseline) or H0 (candidate is not
meaningfully better than elo0 -- typically 0, "no improvement"), rather than always playing a
fixed number of games. A clearly bad or clearly good change is often decided in far fewer games
than a fixed-N test would need; a genuinely close one correctly keeps running instead of
resolving to a coin flip.

Method: the GSPRT (generalized SPRT) normal approximation to the sequential likelihood ratio,
the same one cutechess-cli and fishtest use (see Michel Van den Bergh's "the SPRT applied to
chess engine testing", https://hardy.uhasselt.be/Toga/GSPRT_approximation.pdf). Per-game score
x in {0, 0.5, 1}; t0/t1 are the expected scores implied by elo0/elo1 via the standard logistic
Elo formula; after n games with sample mean s_bar and variance var:

    LLR = n * (t1 - t0) / var * (s_bar - (t0 + t1) / 2)

Stop and accept H1 when LLR >= ln((1-beta)/alpha); stop and accept H0 when
LLR <= ln(beta/(1-alpha)). Default alpha = beta = 0.05 (standard 5% false-positive/negative rate).

Usage:
    uv run python -m tools.sprt_arena --agent . --opponent /path/to/baseline
    uv run python -m tools.sprt_arena --agent . --opponent /path/to/baseline --elo0 0 --elo1 10
    uv run python -m tools.sprt_arena --agent . --opponent . --selfcheck  # sanity: should
        # reach H0 ("no difference") relatively quickly, since there is truly zero Elo gap
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Callable
from functools import partial
from pathlib import Path

from harness.referee import FAILED_TERMINATIONS, play_match
from harness.rules import PLY_CAP
from harness.sandbox import Agent, local
from tools.platform_agent import local as platform_local
from tools.version_arena import FAST_BASE_MS, FAST_INCREMENT_MS, OPENING_LINES, _fen_for


def elo_to_score(elo: float) -> float:
    return 1.0 / (1.0 + math.pow(10.0, -elo / 400.0))


# A real, short winning (or losing) streak gives zero empirical variance -- dividing by that
# is undefined, but the streak itself is strong evidence, not "no evidence": returning 0 there
# (as an early version of this did) is backwards, silently treating the most decisive possible
# result as inconclusive. Flooring variance instead keeps the LLR large-but-finite in that case,
# so a real streak still drives the test toward a correct, prompt conclusion. The floor matches
# the natural variance of a heavily lopsided-but-not-literally-certain outcome (a ~90/10 coin),
# not an arbitrary small number picked to force a particular answer.
_MIN_VARIANCE = 0.09 * 0.91


def compute_llr(results: list[float], elo0: float, elo1: float) -> float:
    n = len(results)
    if n < 2:
        return 0.0
    mean = sum(results) / n
    var = max(sum((x - mean) ** 2 for x in results) / n, _MIN_VARIANCE)
    t0 = elo_to_score(elo0)
    t1 = elo_to_score(elo1)
    return n * (t1 - t0) / var * (mean - (t0 + t1) / 2.0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SPRT A/B test between two agent directories, stopping as soon as the "
        "evidence supports accepting or rejecting the candidate."
    )
    parser.add_argument("--agent", type=Path, default=Path("."))
    parser.add_argument("--opponent", type=Path, required=True)
    parser.add_argument("--base-ms", type=int, default=FAST_BASE_MS)
    parser.add_argument("--increment-ms", type=int, default=FAST_INCREMENT_MS)
    parser.add_argument("--ply-cap", type=int, default=PLY_CAP)
    parser.add_argument("--platform-cpu", type=int,
                        help="Pin to this core and suspend each agent outside its turn")
    parser.add_argument("--engine-python", help="Pinned CPU interpreter for platform subprocesses")
    parser.add_argument("--elo0", type=float, default=0.0, help="H0: no better than this.")
    parser.add_argument("--elo1", type=float, default=5.0, help="H1: at least this much better.")
    parser.add_argument("--alpha", type=float, default=0.05, help="False-positive rate.")
    parser.add_argument("--beta", type=float, default=0.05, help="False-negative rate.")
    parser.add_argument(
        "--max-games",
        type=int,
        default=400,
        help="Safety cap -- report inconclusive rather than run forever if neither bound is hit.",
    )
    parser.add_argument(
        "--selfcheck",
        action="store_true",
        help="Ignore --opponent's identity for interpretation purposes; just a labeled run "
        "intended for --agent and --opponent pointing at the same build, as a sanity check "
        "that the tool correctly finds no difference rather than a spurious one.",
    )
    arguments = parser.parse_args()

    factory: Callable[[Path], Agent] = local
    if arguments.platform_cpu is not None:
        factory = partial(platform_local, cpu=arguments.platform_cpu,
                          python=arguments.engine_python)

    agent = arguments.agent.resolve()
    opponent = arguments.opponent.resolve()
    lower_bound = math.log(arguments.beta / (1 - arguments.alpha))
    upper_bound = math.log((1 - arguments.beta) / arguments.alpha)

    print(
        f"SPRT: elo0={arguments.elo0} elo1={arguments.elo1} alpha={arguments.alpha} "
        f"beta={arguments.beta} bounds=[{lower_bound:.3f}, {upper_bound:.3f}]"
    )

    results: list[float] = []
    terminations: dict[str, int] = {}
    opening_names = list(OPENING_LINES)
    game_num = 0

    while game_num < arguments.max_games:
        for name in opening_names:
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

                if outcome.termination in FAILED_TERMINATIONS or outcome.result == "void":
                    print(white_process.stderr_tail, black_process.stderr_tail)
                    raise SystemExit(f"Invalid SPRT: game {game_num} {outcome.termination}")
                if outcome.result == "draw":
                    score = 0.5
                elif (outcome.result == "white") == agent_plays_white:
                    score = 1.0
                else:
                    score = 0.0
                results.append(score)

                llr = compute_llr(results, arguments.elo0, arguments.elo1)
                mean = sum(results) / len(results)
                side = "white" if agent_plays_white else "black"
                print(
                    f"game {game_num} [{name}, agent={side}]: {outcome.result} by "
                    f"{outcome.termination} -- score {mean:.1%}, llr={llr:.3f}",
                    flush=True,
                )

                if llr >= upper_bound:
                    _report(
                        results, terminations, llr, upper_bound,
                        "H1 ACCEPTED -- candidate is stronger",
                    )
                    return
                if llr <= lower_bound:
                    _report(
                        results, terminations, llr, lower_bound,
                        "H0 ACCEPTED -- candidate is not distinguishable from the baseline "
                        "at this bound (not necessarily identical, just not proven better)",
                    )
                    return
                if game_num >= arguments.max_games:
                    _report(results, terminations, llr, None, "INCONCLUSIVE -- max games reached")
                    return


def _report(
    results: list[float],
    terminations: dict[str, int],
    llr: float,
    bound: float | None,
    verdict: str,
) -> None:
    n = len(results)
    mean = sum(results) / n
    bound_note = f" (bound {bound:.3f})" if bound is not None else ""
    print(f"\n{verdict}")
    print(f"{n} games, score {mean:.1%}, final llr={llr:.3f}{bound_note}")
    print("terminations: " + ", ".join(f"{name} {count}" for name, count in terminations.items()))
    broken = {name: count for name, count in terminations.items() if name in FAILED_TERMINATIONS}
    if broken:
        print(
            "\nwarning: non-clean terminations present -- treat this result as a methodology "
            "artifact, not a pure strength signal: "
            + ", ".join(f"{k} {v}" for k, v in broken.items())
        )


if __name__ == "__main__":
    main()
