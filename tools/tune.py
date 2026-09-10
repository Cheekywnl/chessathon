"""Texel tuning: fits chess_eval.py's DEFAULT_PARAMS against real game outcomes.

Minimizes the error between sigmoid(K * static_eval(position)) and the position's actual game
result (1.0/0.5/0.0 from White's perspective) via coordinate-ascent local search -- the classic
"Texel tuning" method (named for the engine that popularized it): try nudging each parameter up
or down, keep the change if total error drops, repeat until nothing helps at the current step
size, then halve the step and continue. Purely a fitting procedure over already-played or
self-played positions and their eventual outcomes -- no search or judgment call involved beyond
choosing the dataset, unlike hand-picking a magnitude and hoping a real A/B validates it.

Accepts two CSV formats transparently, same auto-detection tools/train_nnue.py uses (and freely
mixable in one --data glob):

- tools/generate_training_data.py's output ("fen,mobility,result"): self-play/real games, a hard
  0/0.5/1 outcome, mobility already computed.
- tools/prepare_lichess_eval.py's output ("fen,depth,cp"): real Stockfish evaluations -- the
  target becomes sigmoid(cp/400) (same K/400 convention as tools/train_nnue.py's _cp_to_target),
  a soft "how good is this position" label instead of a hard game outcome, and mobility gets
  computed from the FEN since this format doesn't carry it. Explicitly the "training on positions
  an existing engine labelled" the competition rules allow -- this only ever prints a candidate
  params array for chess_eval.py, the same engine code this project already ships, refit against
  more data than the two previous attempts had (4,283 and ~6-23k positions -- both of which
  document collinearity/overfitting symptoms this dataset's real size should resolve).

Only ever prints a candidate params array. Nothing in chess_eval.py changes unless that candidate
is copied in by hand after clearing the same bar as every other change this session: gate, WAC,
the endgame regression suite, and a real A/B against the values it would replace.

Usage:
    uv run python -m tools.generate_training_data --games 150   # build a self-play dataset
    uv run python -m tools.tune --data data/tune_positions.csv --iterations 6
    uv run python -m tools.tune --data data/lichess_sample.csv --limit 300000 --iterations 6
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
from pathlib import Path

import chess
import numpy as np

import chess_eval as ce
from tools.train_nnue import _cp_to_target

Position = tuple[chess.Board, int, float]


def load_dataset(pattern: str, limit: int | None = None) -> list[Position]:
    """--limit caps rows taken from EACH matched file (not the combined total) -- this is a
    coordinate-ascent local search, not a lazy/batched training loop, so the whole dataset lives
    in memory as chess.Board objects throughout; capping it is how this stays tractable against a
    20M-row source file (each total_error() pass costs one chess_eval.evaluate_board() call per
    position, and tune() calls total_error() roughly 2 * NUM_PARAMS * iterations times, so this
    is deliberately a sample, not a full pass over everything tools/prepare_lichess_eval.py
    produced -- see the module docstring for what that dataset actually is)."""
    paths: list[str] = []
    for part in pattern.split(","):
        paths.extend(sorted(glob.glob(part.strip())))
    if not paths:
        raise SystemExit(f"no files matched {pattern!r}")

    positions: list[Position] = []
    for path in paths:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames is not None
            is_cp_format = list(reader.fieldnames) == ["fen", "depth", "cp"]
            n_before = len(positions)
            for row in reader:
                if limit is not None and len(positions) - n_before >= limit:
                    break
                board = chess.Board(row["fen"])
                if is_cp_format:
                    mobility = len(list(board.legal_moves))
                    target = _cp_to_target(float(row["cp"]))
                else:
                    mobility = int(row["mobility"])
                    target = float(row["result"])
                positions.append((board, mobility, target))
            print(f"  {path}: {len(positions) - n_before} positions "
                  f"({'cp' if is_cp_format else 'wdl'})")
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
    parser.add_argument(
        "--data", type=str, default="data/tune_positions.csv",
        help="Glob pattern (or comma-separated patterns) matching CSV files.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Cap rows taken from each matched file -- see load_dataset's docstring.",
    )
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

    positions = load_dataset(arguments.data, arguments.limit)
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
