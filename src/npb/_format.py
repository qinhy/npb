"""Physical NPB container format."""

from __future__ import annotations

from dataclasses import dataclass
import struct
from uuid import UUID

import numpy as np

from ._errors import FormatError

MAGIC = b"NPB1"
FORMAT_VERSION = 1
FRAME_SIZE = 64 * 1024
ARRAY_ALIGNMENT = 64

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

    @property
    def frame_count(self) -> int:
        """Number of fixed-size frames in the container."""
        return self.total_bytes // self.frame_size


def align_up(value: int, alignment: int) -> int:
    """Round ``value`` up to the next multiple of ``alignment``."""
    return (value + alignment - 1) // alignment * alignment


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
    if flags != 0:
        raise FormatError(f"unsupported NPB flags: {flags}")
    if frame_size != FRAME_SIZE:
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
    )
