"""Train a clipped HalfKP network from bounded shards, from random initialization.

The optional 640-feature factor is training-only: export adds it into each of the
64 king buckets exactly. Output is in logits; exported output-layer values are
scaled to centipawns. Resume accepts only a checkpoint whose adjacent provenance
record verifies this team's random-initialization lineage and exact source hash.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from tools.halfkp_data import RECORD, file_hash


class HalfKPTrainNet(nn.Module):
    def __init__(self, width: int = 128, factorized: bool = True) -> None:
        super().__init__()
        self.width = width
        self.feature_transformer = nn.EmbeddingBag(40_960, width, mode="sum")
        self.factor = nn.EmbeddingBag(640, width, mode="sum") if factorized else None
        self.ft_bias = nn.Parameter(torch.full((width,), 0.1))
        self.fc1 = nn.Linear(width * 2, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 1)
        nn.init.uniform_(self.feature_transformer.weight, -0.025, 0.025)
        if self.factor is not None:
            nn.init.uniform_(self.factor.weight, -0.025, 0.025)
        nn.init.normal_(self.fc3.weight, std=0.1)
        nn.init.zeros_(self.fc3.bias)

    def transform(self, flat: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        result = self.feature_transformer(flat, offsets) + self.ft_bias
        if self.factor is not None:
            result = result + self.factor(flat % 640, offsets)
        clipped: torch.Tensor = result.clamp(0, 1)
        return clipped

    def forward(
        self, white: torch.Tensor, black: torch.Tensor, offsets: torch.Tensor,
        stm: torch.Tensor,
    ) -> torch.Tensor:
        aw = self.transform(white, offsets)
        ab = self.transform(black, offsets)
        combined = torch.cat((torch.where(stm[:, None], aw, ab),
                              torch.where(stm[:, None], ab, aw)), dim=1)
        h1 = self.fc1(combined).clamp(0, 1)
        h2 = self.fc2(h1).clamp(0, 1)
        result: torch.Tensor = self.fc3(h2).squeeze(-1)
        return result

    def constrain(self) -> None:
        # Each coalesced FT value stays in [-2,2]. 30 active rows at scale 255
        # plus bias fit int16 even at this conservative, unattainable worst case.
        with torch.no_grad():
            self.feature_transformer.weight.clamp_(-1, 1)
            if self.factor is not None:
                self.factor.weight.clamp_(-1, 1)
            self.ft_bias.clamp_(-1, 1)

    def export_arrays(self) -> dict[str, np.ndarray]:
        w1 = self.feature_transformer.weight.detach().cpu().numpy().copy()
        if self.factor is not None:
            factor = self.factor.weight.detach().cpu().numpy()
            w1.reshape(64, 640, self.width)[:] += factor[None, :, :]
        return {
            "W1": w1, "b1": self.ft_bias.detach().cpu().numpy(),
            "W2": self.fc1.weight.detach().cpu().numpy(),
            "b2": self.fc1.bias.detach().cpu().numpy(),
            "W3": self.fc2.weight.detach().cpu().numpy(),
            "b3": self.fc2.bias.detach().cpu().numpy(),
            "W4": self.fc3.weight.detach().cpu().numpy() * 400,
            "b4": self.fc3.bias.detach().cpu().numpy() * 400,
            "activation_clip": np.array(1, dtype=np.float32),
            "width": np.array(self.width, dtype=np.int32),
        }


def tensors(records: np.ndarray, device: torch.device) -> tuple[torch.Tensor, ...]:
    counts = records["count"].astype(np.int64)
    valid = np.arange(30)[None, :] < counts[:, None]
    offsets = np.empty(len(counts), dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts[:-1], out=offsets[1:])
    return tuple(torch.from_numpy(array).to(device) for array in (
        records["white"][valid].astype(np.int64),
        records["black"][valid].astype(np.int64), offsets,
        records["stm"].copy(), records["target"].copy(),
    ))


def batches(
    root: Path, shards: list[dict[str, Any]], batch_size: int,
    rng: np.random.Generator | None,
) -> Iterator[np.ndarray]:
    order = rng.permutation(len(shards)) if rng is not None else np.arange(len(shards))
    for shard_index in order:
        shard = np.load(root / shards[shard_index]["path"], mmap_mode="r", allow_pickle=False)
        if shard.dtype != RECORD or len(shard) != shards[shard_index]["rows"]:
            raise ValueError("prepared shard does not match its manifest")
        row_order = rng.permutation(len(shard)) if rng is not None else None
        for start in range(0, len(shard), batch_size):
            if row_order is None:
                yield shard[start:start + batch_size].copy()
            else:
                yield shard[row_order[start:start + batch_size]]
        del shard


def resource_snapshot() -> dict[str, Any]:
    import psutil  # type: ignore[import-untyped]

    ram = psutil.virtual_memory()
    result: dict[str, Any] = {
        "ram_available_gb": ram.available / 1e9,
        "system_ram_used_gb": ram.used / 1e9,
        "process_rss_gb": psutil.Process().memory_info().rss / 1e9,
    }
    if torch.cuda.is_available():
        result["cuda_allocated_gb"] = torch.cuda.memory_allocated() / 1e9
        result["cuda_reserved_gb"] = torch.cuda.memory_reserved() / 1e9
        info = subprocess.run([
            "nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ], capture_output=True, text=True, check=False)
        result["nvidia_smi_util_used_total"] = info.stdout.strip()
    return result


def validate(
    model: HalfKPTrainNet, root: Path, shards: list[dict[str, Any]],
    batch_size: int, device: torch.device,
) -> tuple[float, int]:
    model.eval()
    total = 0.0
    rows = 0
    with torch.no_grad():
        for records in batches(root, shards, batch_size, None):
            white, black, offsets, stm, target = tensors(records, device)
            predicted = model(white, black, offsets, stm).sigmoid()
            total += float(((predicted - target) ** 2).sum().item())
            rows += len(records)
    if rows == 0:
        raise ValueError("empty validation split")
    return total / rows, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--no-factorization", action="store_true")
    parser.add_argument("--validate-every", type=int, default=10_000_000)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--resume", type=Path,
                        help="Resume this team's checkpoint with matching run.json and SHA-256")
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"Refusing to overwrite run: {args.out}")
    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA must work before training")
    torch.set_num_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    manifest = json.loads((args.prepared / "manifest.json").read_text())
    if not manifest["complete"]:
        raise SystemExit("Data preparation has not completed")
    args.out.mkdir(parents=True)
    model = HalfKPTrainNet(args.width, not args.no_factorization).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    parent_note: dict[str, Any] | None = None
    if args.resume is not None:
        parent_note = json.loads((args.resume.parent / "run.json").read_text())
        lineage = parent_note.get("lineage_random_initialization",
                                  parent_note.get("random_initialization", False))
        if (not lineage or parent_note.get("published_network_used") is not False
                or parent_note["checkpoint_sha256"] != file_hash(args.resume)
                or parent_note["source_manifest_sha256"]
                != file_hash(args.prepared / "manifest.json")
                or parent_note["seed"] != args.seed):
            raise SystemExit("Checkpoint does not match this team's recorded training lineage")
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = args.lr
            group["initial_lr"] = args.lr
        del checkpoint
    total_steps = args.epochs * sum(math.ceil(s["rows"] / args.batch_size)
                                   for s in manifest["train"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=args.lr * 0.1,
    )
    note = {
        "command": [sys.executable, *sys.argv], "seed": args.seed,
        "random_initialization": args.resume is None, "lineage_random_initialization": True,
        "published_network_used": False,
        "factorization": not args.no_factorization, "width": args.width,
        "architecture": "40960 shared FT, clipped dual perspective -> 32 -> 32 -> 1",
        "source_manifest": str((args.prepared / "manifest.json").resolve()),
        "source_manifest_sha256": file_hash(args.prepared / "manifest.json"),
        "torch": torch.__version__, "numpy": np.__version__,
        "device": str(device), "batch_size": args.batch_size,
        "start_unix": time.time(), "resources": resource_snapshot(),
    }
    if parent_note is not None:
        note["parent_run"] = str((args.resume.parent / "run.json").resolve())
        note["parent_checkpoint_sha256"] = parent_note["checkpoint_sha256"]
    (args.out / "run.json").write_text(json.dumps(note, indent=2), encoding="utf8")
    log = (args.out / "metrics.jsonl").open("a", encoding="utf8", buffering=1)

    def report(entry: dict[str, Any]) -> None:
        entry["unix"] = time.time()
        text = json.dumps(entry)
        print(text, flush=True)
        log.write(text + "\n")

    best = math.inf
    stale = 0
    seen = 0
    step = 0
    next_validation = args.validate_every
    started = time.perf_counter()
    interval_start = started
    interval_rows = 0
    last_report = started
    report({"event": "start", **note})
    if args.resume is not None:
        best, initial_validation_rows = validate(model, args.prepared, manifest["validation"],
                                                 args.batch_size, device)
        report({"event": "resume_validation", "validation_mse": best,
                "validation_rows": initial_validation_rows})
        np.savez(args.out / "best_float.npz", allow_pickle=False, **model.export_arrays())
        torch.save({"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "rows": 0, "seed": args.seed},
                   args.out / "best_checkpoint.pt")
    for epoch in range(1, args.epochs + 1):
        rng = np.random.default_rng(args.seed + epoch)
        epoch_rows = 0
        loss_sum = 0.0
        model.train()
        for records in batches(args.prepared, manifest["train"], args.batch_size, rng):
            white, black, offsets, stm, target = tensors(records, device)
            optimizer.zero_grad(set_to_none=True)
            predicted = model(white, black, offsets, stm).sigmoid()
            loss = ((predicted - target) ** 2).mean()
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite training loss")
            loss.backward()
            optimizer.step()
            model.constrain()
            scheduler.step()
            step += 1
            seen += len(records)
            interval_rows += len(records)
            epoch_rows += len(records)
            loss_sum += float(loss.item()) * len(records)
            now = time.perf_counter()
            if now - last_report >= 30:
                resources = resource_snapshot()
                if resources["ram_available_gb"] < 1.5:
                    raise MemoryError("Stopping before RAM reserve falls below 1.5 GB")
                report({"event": "progress", "epoch": epoch, "rows": seen,
                        "train_mse": loss_sum / epoch_rows,
                        "rows_per_second": interval_rows / (now - interval_start),
                        **resources})
                last_report = now
            if seen >= next_validation:
                val, n_val = validate(model, args.prepared, manifest["validation"],
                                      args.batch_size, device)
                report({"event": "validation", "epoch": epoch, "rows": seen,
                        "validation_mse": val, "validation_rows": n_val})
                if val < best:
                    best, stale = val, 0
                    np.savez(args.out / "best_float.npz", allow_pickle=False,
                             **model.export_arrays())
                    torch.save({"state_dict": model.state_dict(),
                                "optimizer": optimizer.state_dict(),
                                "scheduler": scheduler.state_dict(), "rows": seen,
                                "seed": args.seed}, args.out / "best_checkpoint.pt")
                else:
                    stale += 1
                next_validation = seen + args.validate_every
                model.train()
                if stale >= args.patience:
                    break
        val, n_val = validate(model, args.prepared, manifest["validation"], args.batch_size, device)
        if val < best:
            best, stale = val, 0
            np.savez(args.out / "best_float.npz", allow_pickle=False, **model.export_arrays())
            torch.save({"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(), "rows": seen, "seed": args.seed},
                       args.out / "best_checkpoint.pt")
        np.savez(args.out / f"epoch-{epoch:02d}_float.npz", allow_pickle=False,
                 **model.export_arrays())
        duration = time.perf_counter() - started
        report({"event": "epoch", "epoch": epoch, "rows": seen,
                "epoch_rows": epoch_rows, "train_mse": loss_sum / epoch_rows,
                "validation_mse": val, "validation_rows": n_val,
                "best_validation_mse": best, "elapsed_seconds": duration,
                "end_to_end_rows_per_second": seen / duration,
                "estimated_epoch_seconds": manifest["train_rows"] * duration / seen,
                **resource_snapshot()})
        if stale >= args.patience:
            report({"event": "early_stop", "reason": "held-out validation stopped improving"})
            break
    note.update({"rows_trained": seen, "steps": step, "best_validation_mse": best,
                 "training_seconds": time.perf_counter() - started,
                 "checkpoint_sha256": file_hash(args.out / "best_checkpoint.pt"),
                 "float_weights_sha256": file_hash(args.out / "best_float.npz")})
    (args.out / "run.json").write_text(json.dumps(note, indent=2), encoding="utf8")
    report({"event": "complete", **note})
    log.close()


if __name__ == "__main__":
    main()
