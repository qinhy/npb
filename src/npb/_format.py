"""Physical NPB container format."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import TypeAlias
from uuid import UUID

import numpy as np

from ._errors import FormatError

MAGIC = b"NPB1"
FORMAT_VERSION = 1
FRAME_SIZE = 64 * 1024
ARRAY_ALIGNMENT = 64

# Frame sizes are powers of two so that every frame boundary is also an
# ARRAY_ALIGNMENT boundary. The smallest frame holds exactly the fixed header.
MIN_FRAME_SIZE = 64
MAX_FRAME_SIZE = 1 << 30

# Header flag bits.
#
# FLAG_EXTENDED_MANIFEST is set only when a manifest actually uses a feature
# that NPB <= 0.3 cannot interpret: escaped ``$``-keys or ``bytes`` leaves.
# Containers that do not need those features are written with flags == 0 and
# stay byte-for-byte readable by older decoders, while older decoders reject
# the flagged ones cleanly instead of misreading them.
FLAG_EXTENDED_MANIFEST = 0x0001
KNOWN_FLAGS = FLAG_EXTENDED_MANIFEST

# 64-byte fixed header:
#   4  magic
#   2  format version
#   2  flags
#   4  frame size
#  16  schema UUID
#   4  schema version
#   4  JSON metadata length
#   8  data start
#   8  meaningful data bytes
#   8  encoded container bytes
#   4  reserved
HEADER = struct.Struct("<4sHHI16sIIQQQ4x")
HEADER_SIZE = HEADER.size

assert HEADER_SIZE == 64

#: Anything NPB can read a container from. ``bytes``-like inputs are wrapped
#: without copying; decoded arrays keep the wrapped object alive.
BinaryLike: TypeAlias = np.ndarray | bytes | bytearray | memoryview


@dataclass(frozen=True, slots=True)
class BinaryInfo:
    """Header information returned by :func:`npb.peek`."""

    format_version: int
    frame_size: int
    schema_id: UUID
    schema_version: int
    metadata_bytes: int
    data_start: int
    data_bytes: int
    total_bytes: int
    flags: int = 0

    @property
    def frame_count(self) -> int:
        """Number of fixed-size frames in the container."""
        return self.total_bytes // self.frame_size


def align_up(value: int, alignment: int) -> int:
    """Round ``value`` up to the next multiple of ``alignment``."""
    return (value + alignment - 1) // alignment * alignment


def is_valid_frame_size(frame_size: int) -> bool:
    """Return whether ``frame_size`` is a supported power-of-two frame size."""
    return (
        MIN_FRAME_SIZE <= frame_size <= MAX_FRAME_SIZE
        and frame_size & (frame_size - 1) == 0
    )


def require_frame_size(frame_size: int) -> int:
    """Validate a caller-supplied ``frame_size`` argument."""
    if isinstance(frame_size, bool) or not isinstance(frame_size, int):
        raise TypeError("frame_size must be an integer")
    if not is_valid_frame_size(frame_size):
        raise ValueError(
            f"frame_size must be a power of two between {MIN_FRAME_SIZE} and "
            f"{MAX_FRAME_SIZE}, got {frame_size}"
        )
    return frame_size


def require_binary_array(binary: np.ndarray) -> None:
    """Validate the public binary-container contract."""
    if not isinstance(binary, np.ndarray):
        raise TypeError("binary must be a numpy.ndarray")
    if binary.dtype != np.uint8:
        raise TypeError("binary dtype must be np.uint8")
    if binary.ndim != 1:
        raise ValueError("binary must be a 1-D array")
    if not binary.flags.c_contiguous:
        raise ValueError("binary must be C-contiguous")


def as_binary_array(binary: BinaryLike) -> np.ndarray:
    """Return ``binary`` as a 1-D ``np.uint8`` array without copying.

    NumPy arrays must already satisfy the container contract. ``bytes``,
    ``bytearray`` and ``memoryview`` objects are wrapped with
    :func:`numpy.frombuffer`; the wrapper is read-only when the source is.
    """
    if isinstance(binary, np.ndarray):
        require_binary_array(binary)
        return binary

    if isinstance(binary, (bytes, bytearray, memoryview)):
        try:
            return np.frombuffer(binary, dtype=np.uint8)
        except (ValueError, BufferError) as exc:
            raise ValueError("binary buffer must be contiguous") from exc

    raise TypeError(
        "binary must be a numpy.ndarray, bytes, bytearray, or memoryview"
    )


def parse_header(binary: np.ndarray) -> BinaryInfo:
    """Parse and validate the fixed NPB header."""
    require_binary_array(binary)

    if binary.nbytes < HEADER_SIZE:
        raise FormatError("encoded array is too small")

    (
        magic,
        format_version,
        flags,
        frame_size,
        schema_id_bytes,
        schema_version,
        metadata_bytes,
        data_start,
        data_bytes,
        encoded_bytes,
    ) = HEADER.unpack_from(binary, 0)

    if magic != MAGIC:
        raise FormatError("invalid NPB magic")
    if format_version != FORMAT_VERSION:
        raise FormatError(f"unsupported NPB format version: {format_version}")
    if flags & ~KNOWN_FLAGS:
        raise FormatError(f"unsupported NPB flags: {flags}")
    if not is_valid_frame_size(frame_size):
        raise FormatError(f"unsupported frame size: {frame_size}")
    if encoded_bytes < frame_size:
        raise FormatError("encoded container is smaller than one frame")
    if encoded_bytes > binary.nbytes:
        raise FormatError("encoded container exceeds the supplied buffer")
    if encoded_bytes % frame_size != 0:
        raise FormatError("encoded container size is not frame-aligned")

    metadata_end = HEADER_SIZE + metadata_bytes

    if metadata_end > data_start:
        raise FormatError("metadata overlaps the DATA section")
    if data_start % frame_size != 0:
        raise FormatError("DATA section is not frame-aligned")
    if data_start + data_bytes > encoded_bytes:
        raise FormatError("DATA section exceeds the encoded container")

    return BinaryInfo(
        format_version=format_version,
        frame_size=frame_size,
        schema_id=UUID(bytes=schema_id_bytes),
        schema_version=schema_version,
        metadata_bytes=metadata_bytes,
        data_start=data_start,
        data_bytes=data_bytes,
        total_bytes=encoded_bytes,
        flags=flags,
    )
