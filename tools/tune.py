"""Texel tuning: fits chess_eval.py's DEFAULT_PARAMS against real game outcomes.

Minimizes the error between sigmoid(K * static_eval(position)) and the position's actual game
result (1.0/0.5/0.0 from White's perspective) via coordinate-ascent local search -- the classic
"Texel tuning" method (named for the engine that popularized it): try nudging each parameter up
or down, keep the change if total error drops, repeat until nothing helps at the current step
size, then halve the step and continue. Purely a fitting procedure over already-played or
self-played positions and their eventual outcomes -- no search or judgment call involved beyond
choosing the dataset, unlike hand-picking a magnitude and hoping a real A/B validates it.

Only ever prints a candidate params array. Nothing in chess_eval.py changes unless that candidate
is copied in by hand after clearing the same bar as every other change this session: gate, WAC,
the endgame regression suite, and a real A/B against the values it would replace.

Usage:
    uv run python -m tools.generate_training_data --games 150   # build the dataset first
    uv run python -m tools.tune --data data/tune_positions.csv --iterations 6
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import chess
import numpy as np

import chess_eval as ce

Position = tuple[chess.Board, int, float]


def load_dataset(path: Path) -> list[Position]:
    positions: list[Position] = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            board = chess.Board(row["fen"])
            positions.append((board, int(row["mobility"]), float(row["result"])))
    return positions


def _white_eval(board: chess.Board, mobility: int, params: np.ndarray) -> int:
    score = ce.evaluate_board(board, mobility, params)
    return score if board.turn == chess.WHITE else -score


def _sigmoid(x: float, k: float) -> float:
    return 1.0 / (1.0 + math.exp(-k * x / 400.0))


def total_error(
    positions: list[Position],
    params: np.ndarray,
    k: float,
    l2_weight: float = 0.0,
    prior: np.ndarray | None = None,
) -> float:
    """Mean squared error against real outcomes, plus an optional L2 term pulling params back
    toward `prior` (DEFAULT_PARAMS as it stood before this tuning run, not zero -- an already
    reasonably-tuned starting point is a much better anchor than "no knowledge"). Deviations are
    normalised by 100 before squaring so the penalty applies comparably to a 900-point queen
    value and a -5-point isolated-pawn penalty, rather than implicitly regularising small terms
    far harder than large ones. Exists because unconstrained tuning on ~6k self-play positions
    inverted the sign of several well-established terms (bishop pair, isolated-pawn endgame,
    king safety) and lost a real 40-game A/B 16.2% (6.5/40) -- a collinearity/overfitting
    signature, not a dataset-size one (more of the same shallow self-play wouldn't fix
    parameters trading off against each other); anchoring to the prior is the direct fix."""
    total = 0.0
    for board, mobility, result in positions:
        predicted = _sigmoid(_white_eval(board, mobility, params), k)
        total += (result - predicted) ** 2
    mse = total / len(positions)
    if l2_weight <= 0.0 or prior is None:
        return mse
    deviation = ((params.astype(np.float64) - prior.astype(np.float64)) / 100.0) ** 2
    return mse + l2_weight * float(np.mean(deviation))


def find_best_k(positions: list[Position], params: np.ndarray) -> float:
    best_k, best_error = 1.0, float("inf")
    k = 0.1
    while k <= 2.0:
        error = total_error(positions, params, k)
        if error < best_error:
            best_error, best_k = error, k
        k += 0.05
    return best_k


def tune(
    positions: list[Position],
    params: np.ndarray,
    k: float,
    iterations: int,
    step: int,
    l2_weight: float = 0.0,
    prior: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    params = params.copy()
    current_error = total_error(positions, params, k, l2_weight, prior)
    print(f"initial error: {current_error:.6f} (k={k:.2f}, l2_weight={l2_weight})")
    for iteration in range(iterations):
        improved_this_iteration = False
        for idx in range(len(params)):
            original = params[idx]
            best_for_idx = original
            for delta in (step, -step):
                params[idx] = original + delta
                error = total_error(positions, params, k, l2_weight, prior)
                if error < current_error:
                    current_error = error
                    best_for_idx = params[idx]
                    improved_this_iteration = True
            params[idx] = best_for_idx
        print(f"iteration {iteration + 1}/{iterations}: error={current_error:.6f}, step={step}")
        if not improved_this_iteration:
            if step <= 1:
                break
            step = max(1, step // 2)
    return params, current_error


def main() -> None:
    parser = argparse.ArgumentParser(description="Texel-tune chess_eval.py's DEFAULT_PARAMS.")
    parser.add_argument("--data", type=Path, default=Path("data/tune_positions.csv"))
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--step", type=int, default=8)
    parser.add_argument(
        "--l2-weight",
        type=float,
        default=0.02,
        help="Pull toward the current DEFAULT_PARAMS; 0 disables (the unregularised mode that "
        "inverted several established terms' signs on this project's own dataset -- see "
        "total_error's docstring).",
    )
    parser.add_argument("--save", type=Path, default=Path("data/tuned_params.npy"))
    arguments = parser.parse_args()

    positions = load_dataset(arguments.data)
    print(f"loaded {len(positions)} positions from {arguments.data}")

    baseline_params = ce.DEFAULT_PARAMS.copy()
    k = find_best_k(positions, baseline_params)
    baseline_error = total_error(positions, baseline_params, k)
    print(f"baseline (current DEFAULT_PARAMS) error: {baseline_error:.6f}\n")

    tuned_params, tuned_error_with_l2 = tune(
        positions,
        baseline_params,
        k,
        arguments.iterations,
        arguments.step,
        arguments.l2_weight,
        baseline_params,
    )
    tuned_error = total_error(positions, tuned_params, k)

    print(f"\nbaseline error (no L2):        {baseline_error:.6f}")
    print(f"tuned error (no L2, for comparison): {tuned_error:.6f}")
    print(f"tuned error (with L2, what was actually minimised): {tuned_error_with_l2:.6f}")
    print(f"improvement (no-L2 basis):     {(baseline_error - tuned_error) / baseline_error:.2%}\n")

    print("candidate params (only apply after a real A/B validates it):")
    for name, old, new in zip(_param_names(), baseline_params, tuned_params, strict=True):
        marker = "  <-- changed" if old != new else ""
        print(f"  {name:28s} {int(old):6d} -> {int(new):6d}{marker}")

    np.save(arguments.save, tuned_params)
    print(f"\nsaved candidate params to {arguments.save}")


def _param_names() -> list[str]:
    """Index-ordered, not dir()'s alphabetical order -- P_* names are indices into the params
    array (multiple names can share one index, e.g. P_PAWN_MG, P_KNIGHT_MG = 0, 1), so the name
    printed next to each value has to come from a lookup by index, not string sort order."""
    by_index: dict[int, list[str]] = {}
    for name in dir(ce):
        if name.startswith("P_"):
            value = getattr(ce, name)
            if isinstance(value, int):
                by_index.setdefault(value, []).append(name)
    return ["/".join(sorted(by_index[i])) for i in range(ce.NUM_PARAMS)]


if __name__ == "__main__":
    main()
