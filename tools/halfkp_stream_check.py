"""Independent reference checks for streaming features, splits, targets and factor folding."""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
from pathlib import Path

import chess
import numpy as np
import torch

from tools.halfkp_data import RECORD, encode_row, prepare, split_bucket
from tools.nnue_halfkp_bench import _random_positions
from tools.train_halfkp_stream import HalfKPTrainNet, tensors
from tools.train_nnue_halfkp import (
    _cp_to_stm_target,
    _wdl_to_stm_target,
    halfkp_indices_for_fen,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--positions", type=int, default=1000)
    args = parser.parse_args()
    torch.set_num_threads(1)
    positions = [board.fen() for board in _random_positions(args.positions, seed=9731)]
    positions.extend([
        "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
        "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
        "4k3/P7/8/8/8/8/7p/4K3 b - - 0 1",
    ])
    if args.data:
        with args.data.open(newline="", encoding="utf8") as stream:
            reader = csv.reader(stream)
            next(reader)
            for _, row in zip(range(args.positions), reader, strict=False):
                positions.append(row[0])
    records = np.zeros(len(positions), dtype=RECORD)
    for i, fen in enumerate(positions):
        cp = (i % 101 - 50) * 71
        encode_row(fen, cp, True, records[i])
        n = int(records[i]["count"])
        white, black = halfkp_indices_for_fen(fen)
        np.testing.assert_array_equal(records[i]["white"][:n], white)
        np.testing.assert_array_equal(records[i]["black"][:n], black)
        np.testing.assert_allclose(records[i]["target"], _cp_to_stm_target(fen, cp), atol=1e-7)
        mirrored = chess.Board(fen).mirror().fen()
        mw, mb = halfkp_indices_for_fen(mirrored)
        assert sorted(mw) == sorted(black) and sorted(mb) == sorted(white)
        assert split_bucket(fen) == split_bucket(" ".join(fen.split()[:4]) + " 99 200")
        wdl = (i % 3) / 2
        trial = np.zeros(1, dtype=RECORD)
        encode_row(fen, wdl, False, trial[0])
        assert trial[0]["target"] == _wdl_to_stm_target(fen, wdl)
    for fen in ["8/8/8/8/8/8/8/8 w - -", "K7/8/8/8/8/8/8/K6k w - -",
                "PPPPPPPP/PPPPPPPP/PPPPPPPP/PPPPPPPP/8/8/8/K6k w - -",
                "4k3/8/8/8/8/8/8/4K4 w - -", "4k3/8/8/8/8/8/8/4K2x w - -"]:
        try:
            encode_row(fen, 0, True, np.zeros(1, dtype=RECORD)[0])
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid FEN accepted: {fen}")
    torch.manual_seed(17)
    model = HalfKPTrainNet(width=128, factorized=True)
    white_t, black_t, offsets, stm, _ = tensors(records, torch.device("cpu"))
    with torch.no_grad():
        expected = model(white_t, black_t, offsets, stm).numpy() * 400
    weights = model.export_arrays()
    errors = []
    for i, record in enumerate(records):
        count = record["count"]
        aw = (weights["W1"][record["white"][:count]].sum(0) + weights["b1"]).clip(0, 1)
        ab = (weights["W1"][record["black"][:count]].sum(0) + weights["b1"]).clip(0, 1)
        joined = np.concatenate((aw, ab) if record["stm"] else (ab, aw))
        h1 = (weights["W2"] @ joined + weights["b2"]).clip(0, 1)
        h2 = (weights["W3"] @ h1 + weights["b3"]).clip(0, 1)
        value = float((weights["W4"] @ h2 + weights["b4"]).item())
        errors.append(abs(value - expected[i]))
    assert max(errors) < 0.001, max(errors)
    with tempfile.TemporaryDirectory(prefix="halfkp-check-") as temp:
        root = Path(temp)
        source = root / "source.csv"
        with source.open("w", newline="", encoding="utf8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["fen", "depth", "cp"])
            for fen in positions[:137]:
                writer.writerow([fen, 12, 123])
        manifest = prepare([source], root / "cache", shard_rows=31, limit=None)
        assert manifest["rows"] == 137
        assert manifest["train_rows"] + manifest["validation_rows"] == 137
        for group in ("train", "validation"):
            for shard in manifest[group]:
                rows = np.load(root / "cache" / shard["path"], allow_pickle=False)
                assert len(rows) == shard["rows"]
                assert np.all((rows["bucket"] < 10) == (group == "validation"))
    print(json.dumps({"feature_and_target_positions": len(positions),
                      "mirror_checks": len(positions), "counter_independent_splits": True,
                      "malformed_fens_rejected": 5, "sharded_rows_roundtrip": 137,
                      "max_factorization_fold_error_cp": float(max(errors))}, indent=2))


if __name__ == "__main__":
    main()
