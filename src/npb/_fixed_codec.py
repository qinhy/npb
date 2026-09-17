"""Small helpers for using FixedStruct payloads with existing NPB transports."""

from __future__ import annotations

from typing import TypeVar

import numpy as np

from ._fixed import FixedStruct

T = TypeVar("T", bound=FixedStruct)


def encode_fixed(value: FixedStruct, *, out: np.ndarray | None = None) -> np.ndarray:
    """Encode one FixedStruct to a compact 1-D uint8 array with no frame padding."""
    size = value.byte_size()
    if out is None:
        out = np.empty(size, dtype=np.uint8)
    else:
        if not isinstance(out, np.ndarray) or out.dtype != np.uint8 or out.ndim != 1:
            raise TypeError("out must be a 1-D np.uint8 array")
        if not out.flags.c_contiguous or not out.flags.writeable:
            raise ValueError("out must be writable and C-contiguous")
        if out.nbytes < size:
            raise ValueError(f"output buffer too small: need {size}, got {out.nbytes}")
        out = out[:size]

    value.pack_into(out)
    return out


def decode_fixed(model_type: type[T], binary: bytes | bytearray | memoryview | np.ndarray) -> T:
    """Decode one exact fixed payload."""
    if isinstance(binary, np.ndarray):
        if binary.dtype != np.uint8 or binary.ndim != 1 or not binary.flags.c_contiguous:
            raise TypeError("binary must be a contiguous 1-D np.uint8 array")
        binary = memoryview(binary)
    return model_type.from_bytes(binary)
