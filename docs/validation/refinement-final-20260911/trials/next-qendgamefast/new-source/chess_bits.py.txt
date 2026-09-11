"""Portable bit operations lowered by Numba to the host CPU's instructions.

Only readable Python source ships. Numba's normal compiler selects instructions
for the host during import; no architecture-specific machine code is embedded.
"""
from typing import Any

import numpy as np
from numba import njit, types
from numba.extending import intrinsic


@intrinsic
def _trailing_zeros(_typing: Any, value: Any) -> Any:
    if value != types.uint64:
        raise TypeError("trailing zeros requires uint64")
    signature = types.int64(types.uint64)

    def codegen(context: Any, builder: Any, _signature: Any, args: Any) -> Any:
        return builder.cttz(args[0], context.get_constant(types.boolean, False))

    return signature, codegen


@intrinsic
def _population(_typing: Any, value: Any) -> Any:
    if value != types.uint64:
        raise TypeError("population count requires uint64")
    signature = types.int64(types.uint64)

    def codegen(_context: Any, builder: Any, _signature: Any, args: Any) -> Any:
        return builder.ctpop(args[0])

    return signature, codegen


@njit(cache=False, inline="always")
def lsb_index(value: np.uint64) -> int:
    """Lowest set-bit index, or 64 for an empty board."""
    return int(_trailing_zeros(value))  # type: ignore[call-arg]  # Numba supplies typing context.


@njit(cache=False, inline="always")
def popcount(value: np.uint64) -> int:
    return int(_population(value))  # type: ignore[call-arg]  # Numba supplies typing context.
