"""Continue our own HalfKP lineage with a declared phase-weighted training objective.

Every training row is still visited. The held-out split and inference architecture stay
unchanged. Select by weighted validation MSE with a 2% overall regression ceiling.
This criterion only selects a checkpoint to test; it is not playing-strength evidence.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tools.halfkp_data import file_hash
from tools.train_halfkp_stream import (
    HalfKPTrainNet,
    batches,
    resource_snapshot,
    tensors,
)

PHASE_WEIGHTS = np.array([1.0, 4.0, 2.0, 1.0], dtype=np.float32)
PHASE_NAMES = ["2-7", "8-12", "13-20", "21-32"]


def phase_indices(records: np.ndarray) -> np.ndarray:
    # The sparse transformer contains every non-king piece, with two kings omitted.
    return np.searchsorted(np.array([7, 12, 20]), records["count"] + 2, side="left")


def evaluate(
    model: HalfKPTrainNet, root: Path, shards: list[dict[str, Any]],
    batch_size: int, device: torch.device,
) -> dict[str, Any]:
    totals = np.zeros(4, dtype=np.float64)
    counts = np.zeros(4, dtype=np.int64)
    model.eval()
    with torch.no_grad():
        for records in batches(root, shards, batch_size, None):
            if np.any(records["bucket"] >= 10):
                raise ValueError("Training rows found in held-out shards")
            white, black, offsets, stm, target = tensors(records, device)
            errors = ((model(white, black, offsets, stm).sigmoid() - target) ** 2).cpu().numpy()
            phase = phase_indices(records)
            totals += np.bincount(phase, weights=errors, minlength=4)
            counts += np.bincount(phase, minlength=4)
    return {
        "mse": float(totals.sum() / counts.sum()),
        "weighted_mse": float((totals * PHASE_WEIGHTS).sum()
                              / (counts * PHASE_WEIGHTS).sum()),
        "validation_rows": int(counts.sum()),
        "phases": {name: {"rows": int(counts[i]),
                           "mse": float(totals[i] / counts[i]) if counts[i] else None}
                   for i, name in enumerate(PHASE_NAMES)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--resume", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--max-rows", type=int, default=0, help="Bound a throughput pilot")
    parser.add_argument("--validate-every", type=int, default=10_000_000)
    parser.add_argument("--seed", type=int, default=20260910)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit("Refusing to overwrite a training run")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA must work before training")
    resources = resource_snapshot()
    if resources["ram_available_gb"] < 3:
        raise SystemExit("Less than 3 GB of available RAM")
    manifest_path = args.prepared / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    parent = json.loads((args.resume.parent / "run.json").read_text())
    if (not manifest["complete"] or not parent.get("lineage_random_initialization")
            or parent["published_network_used"] is not False
            or file_hash(args.resume) != parent["checkpoint_sha256"]
            or file_hash(manifest_path) != parent["source_manifest_sha256"]
            or parent["seed"] != args.seed):
        raise SystemExit("Dataset or checkpoint lineage mismatch")
    torch.set_num_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    model = HalfKPTrainNet(parent["width"], parent["factorization"]).to(device)
    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer.load_state_dict(checkpoint["optimizer"])
    for group in optimizer.param_groups:
        group["lr"] = args.lr
        group["initial_lr"] = args.lr
    del checkpoint
    total_steps = args.epochs * sum(math.ceil(s["rows"] / args.batch_size)
                                   for s in manifest["train"])
    if args.max_rows:
        total_steps = min(total_steps, math.ceil(args.max_rows / args.batch_size))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=args.lr * 0.1,
    )
    args.out.mkdir(parents=True)
    note: dict[str, Any] = {
        "command": [sys.executable, *sys.argv], "start_unix": time.time(),
        "seed": args.seed, "random_initialization": False,
        "lineage_random_initialization": True, "published_network_used": False,
        "source_manifest": str(manifest_path.resolve()),
        "source_manifest_sha256": file_hash(manifest_path),
        "parent_run": str((args.resume.parent / "run.json").resolve()),
        "parent_checkpoint_sha256": parent["checkpoint_sha256"],
        "width": parent["width"], "factorization": parent["factorization"],
        "phase_weights": dict(zip(PHASE_NAMES, PHASE_WEIGHTS.tolist(), strict=True)),
        "selection": "weighted held-out MSE, overall MSE <= 1.02 times parent",
        "training_code_sha256": file_hash(Path(__file__)), "resources": resources,
        "torch": torch.__version__, "numpy": np.__version__,
    }
    (args.out / "run.json").write_text(json.dumps(note, indent=2))
    log = (args.out / "metrics.jsonl").open("w", buffering=1)

    def report(event: str, **values: Any) -> None:
        line = json.dumps({"event": event, "unix": time.time(), **values})
        print(line, flush=True)
        log.write(line + "\n")

    def save(rows: int) -> None:
        np.savez(args.out / "best_float.npz", allow_pickle=False, **model.export_arrays())
        torch.save({"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "rows": rows, "seed": args.seed},
                   args.out / "best_checkpoint.pt")

    report("start", **note)
    baseline = evaluate(model, args.prepared, manifest["validation"], args.batch_size, device)
    report("baseline_validation", **baseline)
    best = baseline["weighted_mse"]
    selected = baseline
    selected_rows = 0
    save(0)
    seen, stale = 0, 0
    started = last_report = time.perf_counter()
    next_validation = args.validate_every
    stop = False
    for epoch in range(1, args.epochs + 1):
        model.train()
        for records in batches(args.prepared, manifest["train"], args.batch_size,
                               np.random.default_rng(args.seed + epoch)):
            if args.max_rows:
                records = records[:args.max_rows - seen]
            if np.any(records["bucket"] < 10):
                raise ValueError("Held-out rows found in training shards")
            white, black, offsets, stm, target = tensors(records, device)
            weights = torch.from_numpy(PHASE_WEIGHTS[phase_indices(records)]).to(device)
            optimizer.zero_grad(set_to_none=True)
            errors = (model(white, black, offsets, stm).sigmoid() - target) ** 2
            loss = (errors * weights).sum() / weights.sum()
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            optimizer.step()
            model.constrain()
            scheduler.step()
            seen += len(records)
            now = time.perf_counter()
            if now - last_report >= 30:
                resources = resource_snapshot()
                if resources["ram_available_gb"] < 1.5:
                    raise MemoryError("Stopping before reserve falls below 1.5 GB")
                report("progress", rows=seen, epoch=epoch, loss=float(loss.item()),
                       rows_per_second=seen / (now - started), **resources)
                last_report = now
            if seen >= next_validation or (args.max_rows and seen >= args.max_rows):
                result = evaluate(model, args.prepared, manifest["validation"],
                                  args.batch_size, device)
                eligible = result["mse"] <= baseline["mse"] * 1.02
                improved = eligible and result["weighted_mse"] < best
                report("validation", rows=seen, epoch=epoch, eligible=eligible,
                       selected=improved, **result)
                if improved:
                    best, selected, selected_rows, stale = result["weighted_mse"], result, seen, 0
                    save(seen)
                else:
                    stale += 1
                next_validation = seen + args.validate_every
                model.train()
                if stale >= 5 or (args.max_rows and seen >= args.max_rows):
                    stop = True
                    break
        if stop:
            break
    result = evaluate(model, args.prepared, manifest["validation"], args.batch_size, device)
    if result["mse"] <= baseline["mse"] * 1.02 and result["weighted_mse"] < best:
        best, selected, selected_rows = result["weighted_mse"], result, seen
        save(seen)
    note.update({"rows_trained": seen, "selected_rows": selected_rows,
                 "baseline": baseline, "selected": selected, "final": result,
                 "training_seconds": time.perf_counter() - started,
                 "checkpoint_sha256": file_hash(args.out / "best_checkpoint.pt"),
                 "float_weights_sha256": file_hash(args.out / "best_float.npz"),
                 "resources_final": resource_snapshot(), "playing_strength_validated": False})
    (args.out / "run.json").write_text(json.dumps(note, indent=2))
    report("complete", **note)
    log.close()


if __name__ == "__main__":
    main()
