"""Check the actual zip's budget, contents, integer assets and one-core import/move smoke."""

from __future__ import annotations

import argparse
import gzip
import io
import json
import tempfile
import time
import zipfile
from pathlib import Path, PurePosixPath

import chess
import numpy as np

from tools.halfkp_data import file_hash
from tools.platform_agent import SuspendedAgent, local


def expanded_bytes(data: bytes, depth: int = 0) -> int:
    """Count leaf bytes, including ZIP/NPZ and gzip nested inside a submission."""
    if depth > 8:
        raise ValueError("excessive archive nesting")
    if data[:2] == b"\x1f\x8b":
        return expanded_bytes(gzip.decompress(data), depth + 1)
    if zipfile.is_zipfile(io.BytesIO(data)):
        with zipfile.ZipFile(io.BytesIO(data)) as nested:
            return sum(expanded_bytes(nested.read(item), depth + 1)
                       for item in nested.infolist() if not item.is_dir())
    return len(data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", dest="archive", type=Path, required=True)
    parser.add_argument("--engine-python", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cpu", type=int, default=2)
    args = parser.parse_args()
    with zipfile.ZipFile(args.archive) as archive:
        entries = archive.infolist()
        total = sum(entry.file_size for entry in entries)
        if total > 50_000_000:
            raise ValueError(f"submission exceeds 50 MB: {total}")
        recursive_total = sum(expanded_bytes(archive.read(entry)) for entry in entries
                              if not entry.is_dir())
        if recursive_total > 50_000_000:
            raise ValueError(f"submission exceeds recursively expanded 50 MB: {recursive_total}")
        names = [entry.filename for entry in entries]
        assert "agent.py" in names
        assert len(names) == len(set(names)), "duplicate zip member"
        for name in names:
            path = PurePosixPath(name.replace("\\", "/"))
            if path.is_absolute() or ".." in path.parts or ":" in name:
                raise ValueError("unsafe archive path")
            if len(path.parts) == 1:
                assert path.suffix == ".py", name
            else:
                assert path.parts[0] in ("book", "syzygy", "weights"), name
            assert path.suffix.lower() not in (
                ".exe", ".dll", ".pyd", ".so", ".dylib", ".nbc", ".nbi", ".pyc",
                ".pt", ".csv", ".zst", ".parquet", ".cubin",
            ), name
        integer_network = "weights/halfkp.npz" in names
        if integer_network:
            content = io.BytesIO(archive.read("weights/halfkp.npz"))
            with np.load(content, allow_pickle=False) as data:
                assert data["W1"].dtype == np.int16
                assert all(data[key].dtype == np.int8 for key in ("W2", "W3", "W4"))
        with tempfile.TemporaryDirectory(prefix="chess-zip-audit-") as temp:
            root = Path(temp)
            archive.extractall(root)
            process = local(root, cpu=args.cpu, python=args.engine_python)
            assert isinstance(process, SuspendedAgent)
            try:
                started = time.perf_counter()
                process.start(90)
                init_seconds = time.perf_counter() - started
                memory = process.control.memory_info()
                moves = []
                for fen in (
                    chess.STARTING_FEN,
                    "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 4 21",
                    "4k3/8/8/8/8/8/8/3QK3 w - - 0 40",
                ):
                    started = time.perf_counter()
                    move = process.move(fen, 5000)
                    elapsed = time.perf_counter() - started
                    assert chess.Move.from_uci(move) in chess.Board(fen).legal_moves
                    assert elapsed < 5
                    memory = process.control.memory_info()
                    assert memory.rss < 2_000_000_000
                    moves.append({"fen": fen, "move": move, "seconds": elapsed})
                peak_rss = int(getattr(memory, "peak_wset", memory.rss))
            finally:
                process.stop()
            assert "HalfKP load failed" not in process.stderr_tail
            assert "Traceback" not in process.stderr_tail
    result = {"zip": str(args.archive.resolve()), "sha256": file_hash(args.archive),
              "unzipped_bytes": total, "fully_recursive_bytes": recursive_total,
              "file_count": len(names),
              "integer_network": integer_network, "init_seconds": init_seconds,
              "peak_rss_bytes": peak_rss, "cpu_affinity": [args.cpu], "smoke_moves": moves,
              "source_and_permitted_assets_only": True, "playing_strength_validated": False}
    args.out.write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
