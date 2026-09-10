"""Trains a small value network on (fen, result) pairs from tools/generate_training_data.py's
CSV output, meant to run on a GPU (this machine's own self-play data generation runs on CPU in
parallel, in the background -- see data/selfplay_shards/). This is a training script only: it
does NOT touch agent.py, chess_search.py, or chess_eval.py, and produces nothing that ships
until a separate, later integration step is written and validated (see the module docstring's
last paragraph). Safe to develop and run without any risk to the current, working submission.

Architecture: deliberately small and simple for a first attempt, not Serendipity-scale --
768 binary input features (12 piece-planes x 64 squares, White's perspective, no board
mirroring) -> 256 (ReLU) -> 32 (ReLU) -> 1 linear output, interpreted as a centipawn score from
White's perspective, exactly the same convention chess_eval.py's own evaluate() uses (mover-
relative sign flip happens at the caller, not in this network).

Loss: sigmoid(output / 400) compared against the game result (0/0.5/1) via MSE -- the same
K/400 sigmoid scaling tools/tune.py already uses for Texel tuning, so a trained network's
output lands on the same centipawn scale the rest of this codebase already assumes.

Usage (on a GPU machine or Colab):
    pip install torch numpy python-chess
    uv run python -m tools.train_nnue --data "data/selfplay_shards/*.csv" --epochs 30 \
        --out data/nnue_weights.npz

Re-run this periodically as more self-play data accumulates in data/selfplay_shards/ -- each
run is independent (loads whatever CSVs currently match --data), so there's no need to babysit
a single long training job. --data accepts a glob pattern and/or multiple explicit paths.

Output: a single .npz with the raw weight/bias matrices (W1, b1, W2, b2, W3, b3), not a .pt
file -- inference in the actual search hot path needs a hand-written, numba-jitted forward
pass (matching how every other hot-path function in this codebase works, and far faster than
going through torch's Python dispatch for a single-position forward call at every search node),
which is a separate piece of work to write and validate against real games (via
tools/sprt_arena.py, the same discipline as every other change this session) before it's ever
considered for chess_eval.py or agent.py. Nothing here ships on its own.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import chess
import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError as exc:
    raise SystemExit(
        "This script needs torch (pip install torch) -- it's meant to run on a GPU machine "
        "or Colab, not the CPU-only competition sandbox this repo otherwise targets."
    ) from exc

PIECE_ORDER = [
    (chess.PAWN, chess.WHITE), (chess.KNIGHT, chess.WHITE), (chess.BISHOP, chess.WHITE),
    (chess.ROOK, chess.WHITE), (chess.QUEEN, chess.WHITE), (chess.KING, chess.WHITE),
    (chess.PAWN, chess.BLACK), (chess.KNIGHT, chess.BLACK), (chess.BISHOP, chess.BLACK),
    (chess.ROOK, chess.BLACK), (chess.QUEEN, chess.BLACK), (chess.KING, chess.BLACK),
]
INPUT_SIZE = 768
HIDDEN1 = 256
HIDDEN2 = 32


def fen_to_features(fen: str) -> np.ndarray:
    board = chess.Board(fen)
    features = np.zeros(INPUT_SIZE, dtype=np.float32)
    for plane, (piece_type, color) in enumerate(PIECE_ORDER):
        for square in board.pieces(piece_type, color):
            features[plane * 64 + square] = 1.0
    return features


class ValueNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(INPUT_SIZE, HIDDEN1)
        self.fc2 = nn.Linear(HIDDEN1, HIDDEN2)
        self.fc3 = nn.Linear(HIDDEN2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        output: torch.Tensor = self.fc3(x).squeeze(-1)
        return output


def load_dataset(pattern: str) -> tuple[np.ndarray, np.ndarray]:
    paths: list[str] = []
    for part in pattern.split(","):
        paths.extend(sorted(glob.glob(part.strip())))
    if not paths:
        raise SystemExit(f"no files matched {pattern!r}")

    fens: list[str] = []
    results: list[float] = []
    for path in paths:
        with open(path) as f:
            next(f)  # header
            for line in f:
                parts = line.rstrip("\n").rsplit(",", 2)
                if len(parts) != 3:
                    continue
                fen, _mobility, result = parts
                fens.append(fen)
                results.append(float(result))
    print(f"loaded {len(fens)} positions from {len(paths)} file(s)")

    print("extracting features...")
    x = np.stack([fen_to_features(fen) for fen in fens]).astype(np.float32)
    y = np.array(results, dtype=np.float32)
    return x, y


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a small value network.")
    parser.add_argument(
        "--data", type=str, required=True,
        help="Glob pattern (or comma-separated patterns) matching CSV files.",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--out", type=Path, default=Path("data/nnue_weights.npz"))
    arguments = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    x, y = load_dataset(arguments.data)
    n = len(x)
    rng = np.random.default_rng(42)
    perm = rng.permutation(n)
    x, y = x[perm], y[perm]
    n_val = max(1, int(n * arguments.val_fraction))
    x_train, y_train = x[n_val:], y[n_val:]
    x_val, y_val = x[:n_val], y[:n_val]
    print(f"train: {len(x_train)}, val: {len(x_val)}")

    model = ValueNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=arguments.lr)

    x_train_t = torch.from_numpy(x_train).to(device)
    y_train_t = torch.from_numpy(y_train).to(device)
    x_val_t = torch.from_numpy(x_val).to(device)
    y_val_t = torch.from_numpy(y_val).to(device)

    n_train = len(x_train_t)
    for epoch in range(1, arguments.epochs + 1):
        model.train()
        epoch_perm = torch.randperm(n_train, device=device)
        total_loss = 0.0
        for start in range(0, n_train, arguments.batch_size):
            idx = epoch_perm[start : start + arguments.batch_size]
            batch_x, batch_y = x_train_t[idx], y_train_t[idx]
            optimizer.zero_grad()
            raw_output = model(batch_x)
            predicted = torch.sigmoid(raw_output / 400.0)
            loss = torch.mean((predicted - batch_y) ** 2)
            loss.backward()  # type: ignore[no-untyped-call]  # torch's own stub gap, not ours
            optimizer.step()
            total_loss += loss.item() * len(idx)
        train_loss = total_loss / n_train

        model.eval()
        with torch.no_grad():
            val_predicted = torch.sigmoid(model(x_val_t) / 400.0)
            val_loss = torch.mean((val_predicted - y_val_t) ** 2).item()

        print(
            f"epoch {epoch}/{arguments.epochs}: "
            f"train_loss={train_loss:.5f} val_loss={val_loss:.5f}"
        )

    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        arguments.out,
        W1=model.fc1.weight.detach().cpu().numpy(), b1=model.fc1.bias.detach().cpu().numpy(),
        W2=model.fc2.weight.detach().cpu().numpy(), b2=model.fc2.bias.detach().cpu().numpy(),
        W3=model.fc3.weight.detach().cpu().numpy(), b3=model.fc3.bias.detach().cpu().numpy(),
    )
    print(f"\nsaved weights to {arguments.out}")


if __name__ == "__main__":
    main()
