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
from ._schema import BinaryModel, binary_schema
from ._vineyard import VineyardStore
from ._zmq import ZmqPull, ZmqPush

__all__ = [
    "FRAME_SIZE",
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
