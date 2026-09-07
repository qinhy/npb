"""NPB: fast Pydantic + NumPy binary serialization."""

from ._codec import decode, decode_auto, encode, encoded_size, peek
from ._errors import BufferTooSmallError, FormatError, NPBError, SchemaMismatchError
from ._format import FRAME_SIZE, BinaryInfo
from ._schema import BinaryModel, binary_schema

__all__ = [
    "FRAME_SIZE",
    "BinaryInfo",
    "BinaryModel",
    "BufferTooSmallError",
    "FormatError",
    "NPBError",
    "SchemaMismatchError",
    "binary_schema",
    "decode",
    "decode_auto",
    "encode",
    "encoded_size",
    "peek",
]
