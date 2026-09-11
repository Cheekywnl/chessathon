"""Regress the nested-archive size check that the first final upload missed."""

import gzip
import io
import zipfile

import numpy as np

from tools.submission_audit import expanded_bytes


def main() -> None:
    plain = b"hello"
    assert expanded_bytes(plain) == 5
    assert expanded_bytes(gzip.compress(plain)) == 5
    model = io.BytesIO()
    np.savez_compressed(model, numbers=np.ones(3, dtype=np.int16))
    with zipfile.ZipFile(model) as archive:
        model_leaf_bytes = len(archive.read("numbers.npy"))
    assert expanded_bytes(model.getvalue()) == model_leaf_bytes
    outer = io.BytesIO()
    with zipfile.ZipFile(outer, "w") as archive:
        archive.writestr("empty/", b"")
        archive.writestr("plain.bin", plain)
        archive.writestr("nested.gz", gzip.compress(b"0123456789"))
        archive.writestr("model.npz", model.getvalue())
    assert expanded_bytes(outer.getvalue()) == 15 + model_leaf_bytes
    assert expanded_bytes(gzip.compress(outer.getvalue())) == 15 + model_leaf_bytes
    nested = plain
    for _ in range(10):
        nested = gzip.compress(nested)
    try:
        expanded_bytes(nested)
    except ValueError as exc:
        assert "nesting" in str(exc)
    else:
        raise AssertionError("unbounded archive nesting")
    print("PASS: plain, gzip, NPZ, nested ZIP, directory and nesting-limit cases")


if __name__ == "__main__":
    main()
