"""CPU-only HalfKP inference: int16 accumulators, int8 dense weights, clipped activations.

The network is trained by tools/train_halfkp_stream.py. Training-only factors are
already coalesced before quantization. Both accumulators are refreshed on each call;
there is no persistent accumulator state to become stale after a king move.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numba import njit

from chess_nnue_halfkp import active_features_halfkp


@dataclass(frozen=True)
class QuantizedWeights:
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray
    w3: np.ndarray
    b3: np.ndarray
    w4: np.ndarray
    b4: int
    scale2: int
    scale3: int
    output_divisor: float


def load_weights(path: str | Path) -> QuantizedWeights:
    with np.load(path, allow_pickle=False) as data:
        if int(data["format_version"]) != 1 or int(data["activation_scale"]) != 255:
            raise ValueError("unsupported HalfKP quantization format")
        w1 = np.ascontiguousarray(data["W1"])
        width = w1.shape[1]
        required = {
            "W1": ((40_960, width), np.dtype("int16")),
            "b1": ((width,), np.dtype("int16")),
            "W2": ((32, width * 2), np.dtype("int8")),
            "b2": ((32,), np.dtype("int32")),
            "W3": ((32, 32), np.dtype("int8")),
            "b3": ((32,), np.dtype("int32")),
            "W4": ((32,), np.dtype("int8")),
        }
        for key, (shape, dtype) in required.items():
            if data[key].shape != shape or data[key].dtype != dtype:
                raise ValueError(f"invalid HalfKP tensor: {key}")
        scale2, scale3 = int(data["scale2"]), int(data["scale3"])
        divisor = float(data["output_divisor"])
        if (
            min(scale2, scale3) < 1
            or max(scale2, scale3) > 32768
            or not np.isfinite(divisor)
            or divisor <= 0
        ):
            raise ValueError("invalid HalfKP quantization scales")
        bound = 30 * np.abs(w1.astype(np.int32)).max(axis=0)
        bound += np.abs(data["b1"].astype(np.int32))
        if bound.max() > 32767:
            raise ValueError("HalfKP accumulator can overflow int16")
        for layer in (2, 3):
            dense_bound = np.abs(data[f"W{layer}"].astype(np.int64)).sum(axis=1) * 255
            dense_bound += np.abs(data[f"b{layer}"].astype(np.int64))
            if dense_bound.max() > np.iinfo(np.int32).max:
                raise ValueError("HalfKP dense accumulator can overflow int32")
        return QuantizedWeights(
            w1,
            np.ascontiguousarray(data["b1"]),
            np.ascontiguousarray(data["W2"]),
            np.ascontiguousarray(data["b2"]),
            np.ascontiguousarray(data["W3"]),
            np.ascontiguousarray(data["b3"]),
            np.ascontiguousarray(data["W4"]),
            int(data["b4"]),
            scale2,
            scale3,
            divisor,
        )


@njit(cache=False)
def accumulate(indices: np.ndarray, count: int, w1: np.ndarray, b1: np.ndarray) -> np.ndarray:
    acc = b1.copy()
    for k in range(count):
        for j in range(len(acc)):
            acc[j] += w1[indices[k], j]
    for j in range(len(acc)):
        acc[j] = min(255, max(0, acc[j]))
    return acc


@njit(cache=False)
def forward(
    white: np.ndarray,
    black: np.ndarray,
    count: int,
    stm: bool,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    w3: np.ndarray,
    b3: np.ndarray,
    w4: np.ndarray,
    b4: int,
    scale2: int,
    scale3: int,
    output_divisor: float,
) -> float:
    own = accumulate(white if stm else black, count, w1, b1)
    other = accumulate(black if stm else white, count, w1, b1)
    width = w1.shape[1]
    h1 = np.empty(32, dtype=np.int16)
    for j in range(32):
        total = np.int32(b2[j])
        for k in range(width):
            # Numba otherwise widens a scalar reduction to int64. The loader
            # proves each dense sum fits int32, allowing a narrower SIMD sum.
            total = np.int32(
                total
                + np.int32(w2[j, k]) * np.int32(own[k])
                + np.int32(w2[j, width + k]) * np.int32(other[k])
            )
        h1[j] = min(255, max(0, int((np.int64(total) + scale2 // 2) // scale2)))
    h2 = np.empty(32, dtype=np.int16)
    for j in range(32):
        total = np.int32(b3[j])
        for k in range(32):
            total = np.int32(total + np.int32(w3[j, k]) * np.int32(h1[k]))
        h2[j] = min(255, max(0, int((np.int64(total) + scale3 // 2) // scale3)))
    output = b4
    for k in range(32):
        output += int(w4[k]) * int(h2[k])
    return output / output_divisor


@njit(cache=False)
def forward_accumulators(
    own: np.ndarray,
    other: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    w3: np.ndarray,
    b3: np.ndarray,
    w4: np.ndarray,
    b4: int,
    scale2: int,
    scale3: int,
    output_divisor: float,
) -> float:
    """Dense network for cached sums; full-refresh keeps its measured fused body.

    Factoring forward through this helper slowed sparse-position search on the
    target local runtime. Both bodies are checked against the same integer oracle.
    """
    width = len(own)
    h1 = np.empty(32, dtype=np.int16)
    for j in range(32):
        total = np.int32(b2[j])
        for k in range(width):
            # Numba otherwise widens a scalar reduction to int64. The loader
            # proves each dense sum fits int32, allowing a narrower SIMD sum.
            total = np.int32(
                total
                + np.int32(w2[j, k]) * np.int32(own[k])
                + np.int32(w2[j, width + k]) * np.int32(other[k])
            )
        h1[j] = min(255, max(0, int((np.int64(total) + scale2 // 2) // scale2)))
    h2 = np.empty(32, dtype=np.int16)
    for j in range(32):
        total = np.int32(b3[j])
        for k in range(32):
            total = np.int32(total + np.int32(w3[j, k]) * np.int32(h1[k]))
        h2[j] = min(255, max(0, int((np.int64(total) + scale3 // 2) // scale3)))
    output = b4
    for k in range(32):
        output += int(w4[k]) * int(h2[k])
    return output / output_divisor


@njit(cache=False)
def evaluate(
    pawns: np.uint64,
    knights: np.uint64,
    bishops: np.uint64,
    rooks: np.uint64,
    queens: np.uint64,
    kings: np.uint64,
    white: np.uint64,
    black: np.uint64,
    stm: bool,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    w3: np.ndarray,
    b3: np.ndarray,
    w4: np.ndarray,
    b4: int,
    scale2: int,
    scale3: int,
    output_divisor: float,
) -> int:
    white_indices = np.empty(30, dtype=np.int64)
    black_indices = np.empty(30, dtype=np.int64)
    count = active_features_halfkp(
        pawns, knights, bishops, rooks, queens, kings, white, black, white_indices, black_indices
    )
    return round(
        forward(
            white_indices,
            black_indices,
            count,
            stm,
            w1,
            b1,
            w2,
            b2,
            w3,
            b3,
            w4,
            b4,
            scale2,
            scale3,
            output_divisor,
        )
    )


@njit(cache=False, inline="always")
def clip_scaled(total: int | np.int32, scale: int, reciprocal: float) -> int:
    """Exact clipped floor division, with a corrected reciprocal estimate.

    The only interior numerators lie in (0, 255*scale), below 2**24 for every
    supported scale. The double-precision estimate is at most one integer off;
    multiplication checks correct either direction, including exact multiples.
    """
    numerator = np.int64(total) + scale // 2
    if numerator <= 0:
        return 0
    if numerator >= 255 * scale:
        return 255
    quotient = int(numerator * reciprocal)
    if quotient * scale > numerator:
        quotient -= 1
    elif (quotient + 1) * scale <= numerator:
        quotient += 1
    return quotient


@njit(cache=False, inline="always")
def binary_scale_shift(scale: int) -> int:
    shift = 0
    while scale > 1 and scale % 2 == 0:
        scale //= 2
        shift += 1
    return shift if scale == 1 else -1


@njit(cache=False, inline="always")
def clip_dense(total: int | np.int32, scale: int, reciprocal: float, shift: int) -> int:
    if shift >= 0:
        return min(255, max(0, int((np.int64(total) + scale // 2) >> shift)))
    return clip_scaled(total, scale, reciprocal)


@njit(cache=False)
def forward_scratch(
    own: np.ndarray,
    other: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    w3: np.ndarray,
    b3: np.ndarray,
    w4: np.ndarray,
    b4: int,
    scale2: int,
    scale3: int,
    output_divisor: float,
    inverse2: float,
    inverse3: float,
    shift2: int,
    shift3: int,
    h1: np.ndarray,
    h2: np.ndarray,
) -> float:
    """Skip zero activations using contiguous transposed dense-layer weights."""
    width = len(own)
    # The int32 second-layer output buffer also holds intermediate sums.
    sums = h2
    for j in range(32):
        sums[j] = b2[j]
    for k in range(width):
        value = np.int32(own[k])
        if value:
            for j in range(32):
                sums[j] = np.int32(sums[j] + np.int32(w2[k, j]) * value)
        value = np.int32(other[k])
        if value:
            for j in range(32):
                sums[j] = np.int32(sums[j] + np.int32(w2[width + k, j]) * value)
    for j in range(32):
        h1[j] = clip_dense(sums[j], scale2, inverse2, shift2)
        sums[j] = b3[j]
    for k in range(32):
        value = np.int32(h1[k])
        if value:
            for j in range(32):
                sums[j] = np.int32(sums[j] + np.int32(w3[k, j]) * value)
    for j in range(32):
        h2[j] = clip_dense(sums[j], scale3, inverse3, shift3)
    output = b4
    for k in range(32):
        output += int(w4[k]) * int(h2[k])
    return output / output_divisor


def warm_up(w: QuantizedWeights) -> None:
    evaluate(
        np.uint64(0x00FF00000000FF00),
        np.uint64(0x4200000000000042),
        np.uint64(0x2400000000000024),
        np.uint64(0x8100000000000081),
        np.uint64(0x0800000000000008),
        np.uint64(0x1000000000000010),
        np.uint64(0xFFFF),
        np.uint64(0xFFFF000000000000),
        True,
        w.w1,
        w.b1,
        w.w2,
        w.b2,
        w.w3,
        w.b3,
        w.w4,
        w.b4,
        w.scale2,
        w.scale3,
        w.output_divisor,
    )
