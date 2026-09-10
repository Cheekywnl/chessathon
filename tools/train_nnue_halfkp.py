"""Trains a HalfKP-style, king-relative value network -- the real NNUE architecture, as a
second, higher-ceiling track alongside tools/train_nnue.py's simpler plain-piece-square net.
Meant to run on a GPU. Same two accepted CSV formats as tools/train_nnue.py, same auto-detection.

Why HalfKP over plain piece-square features (see tools/train_nnue.py): a plain one-hot net has
to learn "a knight on f3 is worth X" as one fixed number, the same in every position. HalfKP
features are (our_king_square, piece_square, piece_type, relative_color) tuples instead of just
(piece_square, piece_type, color) -- the network can learn a *different* value for the same
piece on the same square depending on where its own king is (e.g. a knight defending a castled
king vs. the same knight with the king elsewhere). This is the single biggest lever real NNUE
nets use over a plain piece-square net, and is why this is being attempted as a second, riskier,
higher-ceiling track after the plain net (chess_nnue.py's architecture) lost a real 264-game
SPRT test earlier tonight (see viewer/activity_log.jsonl) -- that result was a real, validated
negative for the simpler architecture and generic training data, not proof a better-designed
network can't help; this is the more serious attempt referenced there.

Feature indexing and the black-perspective mirror, verified against chessprogramming.org/NNUE
and the Stockfish NNUE docs before writing any code (not guessed from memory):
    p_idx = piece_type * 2 + relative_color            # relative_color: 0 = ours, 1 = enemy
    feature_index = piece_square + (p_idx + king_square * 10) * 64
    # range [0, 64*64*10 - 1] = [0, 40959] per perspective; kings are never their own feature
For Black's perspective specifically: every square (including the king) is XORed with 0b111000
(56) -- the same vertical-mirror trick chess_eval.py already uses for black PST lookups -- and
every piece's relative_color flips (black pieces become "ours", white become "enemy"). White's
perspective uses raw squares and colors, no transform.

Architecture (matches documented early/small Stockfish NNUE sizing, not full-scale Stockfish):
    two 40960-wide sparse inputs (one per perspective) -> a SHARED 256-wide linear feature
    transformer (implemented as nn.EmbeddingBag(sum) since materializing a dense (batch, 40960)
    tensor for ~30-nonzero rows would be enormous and almost entirely wasted) -> concatenate
    [accumulator_side_to_move, accumulator_other_side] (512) -> 32 -> 32 -> 1.

Output convention is intentionally different from tools/train_nnue.py's: side-to-move relative
(the standard NNUE convention -- "the side-to-move accumulator is listed first" -- and it lets
the net learn tempo), not White-relative. Training targets are converted accordingly: the
Lichess eval database's cp is White-relative (verified, not assumed -- see the commit this
shipped in), so it gets negated when Black is to move before the sigmoid(cp/400) conversion;
self-play's WDL result is also White-relative per tools/generate_training_data.py's own
docstring, so it becomes 1-result when Black is to move.

This is a training script only, same as tools/train_nnue.py: does not touch agent.py,
chess_search.py, or chess_eval.py. Nothing here ships until a separate inference module
(chess_nnue_halfkp.py) is built, cross-validated against this file's reference feature
extraction, benchmarked for speed, and only then considered for integration -- same discipline
as the plain-net track, which is exactly the discipline that caught its Lucena regression before
it shipped.

Usage (on a GPU machine or Colab):
    uv run python -m tools.train_nnue_halfkp --data "data/lichess/lichess_eval_large.csv" \
        --epochs 20 --out data/nnue_halfkp_weights.npz
"""

from __future__ import annotations

import argparse
import csv
import glob
from collections.abc import Iterator
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError as exc:
    raise SystemExit(
        "This script needs torch (pip install torch) -- it's meant to run on a GPU machine "
        "or Colab, not the CPU-only competition sandbox this repo otherwise targets."
    ) from exc

try:
    from numba import njit
except ImportError as exc:
    raise SystemExit(
        "This script needs numba (pip install numba) -- feature extraction is jitted, the "
        "same real bottleneck fix tools/train_nnue.py needed."
    ) from exc

INPUT_SIZE = 40_960  # 64 king squares * 64 piece squares * 5 piece types * 2 relative colors
L1 = 256  # per-perspective accumulator width (shared feature-transformer weights)
L2 = 32
L3 = 32
MAX_ACTIVE_PER_PERSPECTIVE = 30  # 32 pieces max on board, minus the 2 kings

# Piece-type index for each FEN letter, case-insensitive, kings excluded (they're the anchor,
# never their own feature). Purely an internal numbering -- doesn't need to match any external
# convention, only needs to be self-consistent between this file and chess_nnue_halfkp.py, which
# is exactly what tools/nnue_halfkp_bench.py cross-checks.
PAWN, KNIGHT, BISHOP, ROOK, QUEEN = 0, 1, 2, 3, 4
_PIECE_TYPE_FOR_LETTER = {
    "p": PAWN, "n": KNIGHT, "b": BISHOP, "r": ROOK, "q": QUEEN,
}


def _parse_fen_pieces(fen: str) -> tuple[list[tuple[int, int, int]], int, int]:
    """Returns (pieces, white_king_square, black_king_square). pieces is a list of
    (square, piece_type, color) for every non-king piece; color 0=white, 1=black."""
    placement = fen.split(" ", 1)[0]
    pieces: list[tuple[int, int, int]] = []
    white_king_sq = -1
    black_king_sq = -1
    rank, file = 7, 0
    for ch in placement:
        if ch == "/":
            rank -= 1
            file = 0
        elif ch.isdigit():
            file += int(ch)
        else:
            square = rank * 8 + file
            lower = ch.lower()
            if lower == "k":
                if ch == "K":
                    white_king_sq = square
                else:
                    black_king_sq = square
            else:
                color = 0 if ch.isupper() else 1
                pieces.append((square, _PIECE_TYPE_FOR_LETTER[lower], color))
            file += 1
    if white_king_sq < 0 or black_king_sq < 0:
        raise ValueError(f"fen missing a king: {fen!r}")
    return pieces, white_king_sq, black_king_sq


def halfkp_indices_for_fen(fen: str) -> tuple[list[int], list[int]]:
    """Reference (plain Python) HalfKP feature extraction -- the ground truth
    chess_nnue_halfkp.py's jitted, bitboard-based version gets cross-checked against. Returns
    (white_perspective_indices, black_perspective_indices)."""
    pieces, white_king_sq, black_king_sq = _parse_fen_pieces(fen)
    white_idx: list[int] = []
    black_idx: list[int] = []
    for square, piece_type, color in pieces:
        rel_color_white = 0 if color == 0 else 1
        white_idx.append(square + (piece_type * 2 + rel_color_white + white_king_sq * 10) * 64)
        mirrored_square = square ^ 56
        mirrored_king = black_king_sq ^ 56
        rel_color_black = 0 if color == 1 else 1
        black_idx.append(
            mirrored_square + (piece_type * 2 + rel_color_black + mirrored_king * 10) * 64
        )
    return white_idx, black_idx


def _white_to_move(fen: str) -> bool:
    fields = fen.split(" ")
    return len(fields) < 2 or fields[1] == "w"


def _cp_to_stm_target(fen: str, cp: float) -> float:
    """Lichess eval database cp is White-relative (verified against the database's own
    documentation, not assumed). Flip to side-to-move-relative before the sigmoid conversion."""
    stm_cp = cp if _white_to_move(fen) else -cp
    return float(1.0 / (1.0 + np.exp(-stm_cp / 400.0)))


def _wdl_to_stm_target(fen: str, white_result: float) -> float:
    """tools/generate_training_data.py's `result` is White's outcome (1.0/0.5/0.0); flip to
    side-to-move-relative (a probability complement, not a sign flip, since it's already a
    [0, 1] win probability rather than a centipawn score)."""
    return white_result if _white_to_move(fen) else 1.0 - white_result


def load_dataset(pattern: str, limit: int | None = None) -> tuple[list[str], np.ndarray]:
    """Same lazy loading discipline as tools/train_nnue.py's load_dataset -- FENs and targets
    only, no feature extraction here. Targets are already side-to-move-relative on return."""
    paths: list[str] = []
    for part in pattern.split(","):
        paths.extend(sorted(glob.glob(part.strip())))
    if not paths:
        raise SystemExit(f"no files matched {pattern!r}")

    fens: list[str] = []
    targets: list[float] = []
    for path in paths:
        with open(path, newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            is_cp_format = header == ["fen", "depth", "cp"]
            n_before = len(fens)
            for row in reader:
                if limit is not None and len(fens) - n_before >= limit:
                    break
                if len(row) != 3:
                    continue
                fen, _second, third = row
                fens.append(fen)
                if is_cp_format:
                    targets.append(_cp_to_stm_target(fen, float(third)))
                else:
                    targets.append(_wdl_to_stm_target(fen, float(third)))
            print(f"  {path}: {len(fens) - n_before} positions ({'cp' if is_cp_format else 'wdl'})")
    print(f"loaded {len(fens)} positions from {len(paths)} file(s)")
    return fens, np.array(targets, dtype=np.float32)


# ---- jitted feature extraction (same fix tools/train_nnue.py needed) ----

@njit(cache=False, nogil=True)
def _fen_codes_to_halfkp(
    codes: np.ndarray,
    piece_type_lookup: np.ndarray,
    white_out: np.ndarray,
    black_out: np.ndarray,
) -> int:
    """Fills white_out/black_out (each must have room for MAX_ACTIVE_PER_PERSPECTIVE) with this
    position's active HalfKP feature indices per perspective, returns how many were written
    (both perspectives always have the same count -- one entry per non-king piece)."""
    # First pass: locate the kings (needed before any feature index can be computed).
    rank = 7
    file = 0
    white_king_sq = -1
    black_king_sq = -1
    for i in range(codes.shape[0]):
        code = codes[i]
        if code == 32:
            break
        if code == 47:
            rank -= 1
            file = 0
        elif 48 <= code <= 57:
            file += code - 48
        else:
            if code == 75:  # 'K'
                white_king_sq = rank * 8 + file
            elif code == 107:  # 'k'
                black_king_sq = rank * 8 + file
            file += 1

    if white_king_sq < 0 or black_king_sq < 0:
        raise ValueError("FEN missing a king")
    black_king_mirrored = black_king_sq ^ 56

    # Second pass: emit one feature pair per non-king piece.
    rank = 7
    file = 0
    n = 0
    for i in range(codes.shape[0]):
        code = codes[i]
        if code == 32:
            break
        if code == 47:
            rank -= 1
            file = 0
        elif 48 <= code <= 57:
            file += code - 48
        else:
            if code != 75 and code != 107:
                if n >= len(white_out) or n >= len(black_out):
                    raise ValueError("more than 30 non-king pieces")
                square = rank * 8 + file
                piece_type = piece_type_lookup[code]
                is_white = 1 if code < 97 else 0  # uppercase ASCII < lowercase ASCII
                rel_color_white = 0 if is_white == 1 else 1
                white_out[n] = square + (piece_type * 2 + rel_color_white + white_king_sq * 10) * 64
                mirrored_square = square ^ 56
                rel_color_black = 0 if is_white == 0 else 1
                bp_idx = piece_type * 2 + rel_color_black
                black_out[n] = mirrored_square + (bp_idx + black_king_mirrored * 10) * 64
                n += 1
            file += 1
    return n


_PIECE_TYPE_LOOKUP = np.full(128, -1, dtype=np.int8)
for _ch, _idx in _PIECE_TYPE_FOR_LETTER.items():
    _PIECE_TYPE_LOOKUP[ord(_ch)] = _idx
    _PIECE_TYPE_LOOKUP[ord(_ch.upper())] = _idx


def halfkp_indices_for_fen_fast(fen: str) -> tuple[list[int], list[int]]:
    """Jitted version of halfkp_indices_for_fen -- verified bit-for-bit identical to it before
    use (see tools/nnue_halfkp_bench.py)."""
    codes = np.frombuffer(fen.encode("ascii"), dtype=np.uint8)
    white_out = np.empty(MAX_ACTIVE_PER_PERSPECTIVE, dtype=np.int64)
    black_out = np.empty(MAX_ACTIVE_PER_PERSPECTIVE, dtype=np.int64)
    n = _fen_codes_to_halfkp(codes, _PIECE_TYPE_LOOKUP, white_out, black_out)
    return white_out[:n].tolist(), black_out[:n].tolist()


# Compile once, single-threaded. The 150M CSV also contains impossible boards with
# more than 30 non-king pieces; the explicit bound above prevents an unchecked write.
# Full-dataset training uses tools.halfkp_data and tools.train_halfkp_stream, which
# reject malformed boards, retain rejection counts, and never build full FEN lists.
halfkp_indices_for_fen_fast("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")

def build_embeddingbag_batch(
    fens: list[str], indices: np.ndarray
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Builds the (flat_indices, offsets) pair nn.EmbeddingBag needs, for both perspectives, plus
    a per-row white-to-move mask.

    Deliberately single-threaded, not chunked across a ThreadPoolExecutor the way
    tools/train_nnue.py's features_for_indices is: that exact pattern (nogil=True numba calls
    dispatched to worker threads) reliably crashed the whole process here -- silently, with no
    Python-catchable exception, reproducible at 100k+ rows, so not the compilation-race issue a
    warm-up call would fix (tried that first; didn't help). Isolated by testing the identical
    per-row work single-threaded, which processed all 100k rows correctly in ~0.5s (~180k
    positions/sec) -- fast enough on its own that the extra complexity and crash risk of
    threading buys nothing worth having here. If this ever needs to be faster, revisit with
    multiprocessing (real process isolation) rather than threads."""
    white_lists: list[list[int]] = []
    black_lists: list[list[int]] = []
    for idx in indices:
        w, b = halfkp_indices_for_fen_fast(fens[idx])
        white_lists.append(w)
        black_lists.append(b)

    white_flat: list[int] = []
    white_offsets: list[int] = [0]
    for lst in white_lists:
        white_flat.extend(lst)
        white_offsets.append(len(white_flat))
    black_flat: list[int] = []
    black_offsets: list[int] = [0]
    for lst in black_lists:
        black_flat.extend(lst)
        black_offsets.append(len(black_flat))

    stm_white = np.array([fens[idx].split(" ")[1] == "w" for idx in indices], dtype=bool)

    return (
        torch.tensor(white_flat, dtype=torch.long),
        torch.tensor(white_offsets[:-1], dtype=torch.long),
        torch.tensor(black_flat, dtype=torch.long),
        torch.tensor(black_offsets[:-1], dtype=torch.long),
        torch.from_numpy(stm_white),
    )


def iter_batches(
    fens: list[str], y: np.ndarray, batch_size: int, rng: np.random.Generator | None
) -> Iterator[tuple[tuple[torch.Tensor, ...], np.ndarray]]:
    n = len(fens)
    order = rng.permutation(n) if rng is not None else np.arange(n)
    for start in range(0, n, batch_size):
        idx = order[start : start + batch_size]
        yield build_embeddingbag_batch(fens, idx), y[idx]


class HalfKPNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.feature_transformer = nn.EmbeddingBag(INPUT_SIZE, L1, mode="sum")
        self.ft_bias = nn.Parameter(torch.zeros(L1))
        self.fc1 = nn.Linear(L1 * 2, L2)
        self.fc2 = nn.Linear(L2, L3)
        self.fc3 = nn.Linear(L3, 1)

    def forward(
        self,
        white_flat: torch.Tensor,
        white_offsets: torch.Tensor,
        black_flat: torch.Tensor,
        black_offsets: torch.Tensor,
        stm_white: torch.Tensor,
    ) -> torch.Tensor:
        acc_white = torch.relu(self.feature_transformer(white_flat, white_offsets) + self.ft_bias)
        acc_black = torch.relu(self.feature_transformer(black_flat, black_offsets) + self.ft_bias)
        stm_mask = stm_white.unsqueeze(1)
        acc_stm = torch.where(stm_mask, acc_white, acc_black)
        acc_other = torch.where(stm_mask, acc_black, acc_white)
        combined = torch.cat([acc_stm, acc_other], dim=1)
        h2 = torch.relu(self.fc1(combined))
        h3 = torch.relu(self.fc2(h2))
        output: torch.Tensor = self.fc3(h3).squeeze(-1)
        return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a HalfKP-style value network.")
    parser.add_argument(
        "--data", type=str, required=True,
        help="Glob pattern (or comma-separated patterns) matching CSV files.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=Path, default=Path("data/nnue_halfkp_weights.npz"))
    parser.add_argument(
        "--device", type=str, default=None, choices=["cuda", "mps", "cpu"],
        help="Override auto-detected device (for diagnostics).",
    )
    arguments = parser.parse_args()

    if arguments.device is not None:
        device = torch.device(arguments.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"device: {device}")

    fens, y = load_dataset(arguments.data, arguments.limit)
    n = len(fens)
    split_rng = np.random.default_rng(42)
    perm = split_rng.permutation(n)
    fens = [fens[i] for i in perm]
    y = y[perm]
    n_val = max(1, int(n * arguments.val_fraction))
    val_fens, val_y = fens[:n_val], y[:n_val]
    train_fens, train_y = fens[n_val:], y[n_val:]
    print(f"train: {len(train_fens)}, val: {len(val_fens)}")

    model = HalfKPNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=arguments.lr)

    n_train = len(train_fens)
    epoch_rng = np.random.default_rng(43)
    for epoch in range(1, arguments.epochs + 1):
        model.train()
        total_loss = 0.0
        train_batches = iter_batches(train_fens, train_y, arguments.batch_size, epoch_rng)
        for batch_tensors, batch_y in train_batches:
            batch_tensors = tuple(t.to(device) for t in batch_tensors)
            batch_y_t = torch.from_numpy(batch_y).to(device)
            optimizer.zero_grad()
            raw_output = model(*batch_tensors)
            predicted = torch.sigmoid(raw_output / 400.0)
            loss = torch.mean((predicted - batch_y_t) ** 2)
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
            total_loss += loss.item() * len(batch_y)
        train_loss = total_loss / n_train

        model.eval()
        val_loss_total = 0.0
        with torch.no_grad():
            for batch_tensors, batch_y in iter_batches(val_fens, val_y, arguments.batch_size, None):
                batch_tensors = tuple(t.to(device) for t in batch_tensors)
                batch_y_t = torch.from_numpy(batch_y).to(device)
                val_predicted = torch.sigmoid(model(*batch_tensors) / 400.0)
                val_loss_total += torch.mean((val_predicted - batch_y_t) ** 2).item() * len(batch_y)
        val_loss = val_loss_total / len(val_fens)

        print(
            f"epoch {epoch}/{arguments.epochs}: "
            f"train_loss={train_loss:.5f} val_loss={val_loss:.5f}"
        )

    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        arguments.out,
        W1=model.feature_transformer.weight.detach().cpu().numpy(),
        b1=model.ft_bias.detach().cpu().numpy(),
        W2=model.fc1.weight.detach().cpu().numpy(), b2=model.fc1.bias.detach().cpu().numpy(),
        W3=model.fc2.weight.detach().cpu().numpy(), b3=model.fc2.bias.detach().cpu().numpy(),
        W4=model.fc3.weight.detach().cpu().numpy(), b4=model.fc3.bias.detach().cpu().numpy(),
    )
    print(f"\nsaved weights to {arguments.out}")


if __name__ == "__main__":
    main()
