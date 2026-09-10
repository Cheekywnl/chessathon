"""Quantize team-trained clipped HalfKP weights and prove the int16 accumulator bound."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from tools.halfkp_data import file_hash


def quantize(source: Path, output: Path, dense_scaling: str = "power2") -> dict[str, object]:
    if output.exists():
        raise ValueError(f"Refusing to overwrite weights: {output}")
    with np.load(source, allow_pickle=False) as data:
        if float(data["activation_clip"]) != 1:
            raise ValueError("quantizer requires the clipped architecture")
        arrays = {key: data[key] for key in ("W1", "b1", "W2", "b2", "W3", "b3", "W4", "b4")}
    if not all(np.all(np.isfinite(array)) for array in arrays.values()):
        raise ValueError("non-finite float weights")
    w1 = np.rint(arrays["W1"] * 255).astype(np.int32)
    b1 = np.rint(arrays["b1"] * 255).astype(np.int32)
    bound = int((30 * np.abs(w1).max(axis=0) + np.abs(b1)).max())
    if bound > 32767:
        raise ValueError(f"int16 accumulator unsafe: {bound}")
    result: dict[str, np.ndarray] = {"W1": w1.astype(np.int16), "b1": b1.astype(np.int16)}
    scales = []
    for layer in (2, 3, 4):
        weights = arrays[f"W{layer}"]
        maximum = max(float(np.abs(weights).max()), 1e-9)
        scale = 2.0 ** min(15, math.floor(math.log2(127 / maximum)))
        if dense_scaling == "fullrange":
            scale = min(32768.0, 127 / maximum)
            if layer != 4:
                scale = float(math.floor(scale))
        if layer != 4 and scale < 1:
            raise ValueError("hidden weight outside supported range")
        quantized = np.rint(weights * scale)
        bias = np.rint(arrays[f"b{layer}"] * 255 * scale)
        if np.abs(quantized).max() > 127 or np.abs(bias).max() > 2**30:
            raise ValueError("quantization overflow")
        result[f"W{layer}"] = quantized.astype(np.int8)
        result[f"b{layer}"] = bias.astype(np.int32)
        scales.append(scale)
    result["W4"] = result["W4"].reshape(-1)
    result["b4"] = result["b4"].reshape(())
    result.update({"scale2": np.array(scales[0], dtype=np.int32),
                   "scale3": np.array(scales[1], dtype=np.int32),
                   "output_divisor": np.array(255 * scales[2], dtype=np.float64),
                   "format_version": np.array(1, dtype=np.int32),
                   "activation_scale": np.array(255, dtype=np.int32)})
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, allow_pickle=False, **result)
    note: dict[str, object] = {
        "float_source": str(source.resolve()), "source_sha256": file_hash(source),
        "output": str(output.resolve()), "sha256": file_hash(output),
        "file_bytes": output.stat().st_size, "int16_accumulator_absolute_bound": bound,
        "weight_scales": scales, "width": int(w1.shape[1]),
        "dense_scaling": dense_scaling,
        "playing_strength_validated": False,
    }
    output.with_suffix(".json").write_text(json.dumps(note, indent=2), encoding="utf8")
    return note


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--float", dest="source", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--dense-scaling", choices=("power2", "fullrange"), default="power2")
    args = parser.parse_args()
    print(json.dumps(quantize(args.source, args.out, args.dense_scaling), indent=2))


if __name__ == "__main__":
    main()
