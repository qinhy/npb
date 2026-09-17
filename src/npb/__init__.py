"""NPB: fast Pydantic + NumPy binary serialization."""

from ._blob import BlobRef, BlobStore
from ._codec import decode, decode_auto, encode, encoded_size, peek
from ._errors import (
    BlobStoreError,
    BlobStoreRequiredError,
    BufferTooSmallError,
    FormatError,
    NPBError,
    SchemaMismatchError,
)
from ._format import FRAME_SIZE, BinaryInfo
from ._fixed import (
    Bit, Bool8, Enum8, FixedArray, FixedPointU16, FixedStr, FixedStruct, NpbStruct,
    Float32, Float64, Int8, Int16, Int32, Int64, IPv4, Nested, OptionalNested,
    UInt8, UInt16, UInt32, UInt64,
)
from ._fixed_codec import decode_fixed, encode_fixed
from ._schema import BinaryModel, binary_schema
from ._vineyard import VineyardStore
from ._zmq import ZmqPull, ZmqPush

__all__ = [
    "FRAME_SIZE",
    "Bit",
    "Bool8",
    "Enum8",
    "FixedArray",
    "FixedPointU16",
    "FixedStr",
    "FixedStruct",
    "NpbStruct",
    "Float32",
    "Float64",
    "Int8",
    "Int16",
    "Int32",
    "Int64",
    "IPv4",
    "Nested",
    "OptionalNested",
    "UInt8",
    "UInt16",
    "UInt32",
    "UInt64",
    "decode_fixed",
    "encode_fixed",
    "BinaryInfo",
    "BinaryModel",
    "BlobRef",
    "BlobStore",
    "BlobStoreError",
    "BlobStoreRequiredError",
    "BufferTooSmallError",
    "FormatError",
    "NPBError",
    "SchemaMismatchError",
    "VineyardStore",
    "ZmqPull",
    "ZmqPush",
    "binary_schema",
    "decode",
    "decode_auto",
    "encode",
    "encoded_size",
    "peek",
]
