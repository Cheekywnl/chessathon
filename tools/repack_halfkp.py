"""Losslessly repack integer NPZ tensors using standard ZIP compression.

NumPy's NPZ reader supports these standard ZIP methods. Array values, shapes,
dtypes and the existing inference format stay unchanged; no decoder ships.
"""
from __future__ import annotations

import argparse
import io
import json
import time
import zipfile
from pathlib import Path

import numpy as np

import chess_halfkp_int as qi
from tools.halfkp_data import file_hash


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--compression", choices=["bzip2", "lzma"], default="bzip2")
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit("Refusing to overwrite a weight archive")
    qi.load_weights(args.source)
    with np.load(args.source, allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}
    method = zipfile.ZIP_BZIP2 if args.compression == "bzip2" else zipfile.ZIP_LZMA
    args.out.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with zipfile.ZipFile(args.out, "w", compression=method) as archive:
        for name, array in arrays.items():
            stream = io.BytesIO()
            np.save(stream, array, allow_pickle=False)
            archive.writestr(name + ".npy", stream.getvalue())
    packed_seconds = time.perf_counter() - started
    started = time.perf_counter()
    with np.load(args.out, allow_pickle=False) as result:
        assert set(result.files) == set(arrays)
        for name, expected in arrays.items():
            actual = result[name]
            assert actual.dtype == expected.dtype and actual.shape == expected.shape
            assert actual.tobytes(order="C") == expected.tobytes(order="C"), name
    qi.load_weights(args.out)
    note = {"source": str(args.source.resolve()), "source_sha256": file_hash(args.source),
            "out": str(args.out.resolve()), "sha256": file_hash(args.out),
            "source_bytes": args.source.stat().st_size, "bytes": args.out.stat().st_size,
            "compression": args.compression, "all_tensor_bytes_identical": True,
            "pack_seconds": packed_seconds, "verify_seconds": time.perf_counter() - started}
    args.out.with_suffix(".repack.json").write_text(json.dumps(note, indent=2))
    print(json.dumps(note, indent=2))


if __name__ == "__main__":
    main()
