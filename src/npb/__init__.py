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
    FixedField,UInt8,UInt16,UInt32,UInt64,Int8,Int16,Int32,Int64,Float32,Float64,
    Bool8,FixedStr,IPv4,Enum8,FixedPointU16,FixedNDArray,Bit,Nested,OptionalNested,
    FixedArray,FixedStructMeta,FixedStruct,NpbStruct
)
from ._fixed_codec import decode_fixed, encode_fixed
from ._schema import BinaryModel, binary_schema
from ._vineyard import VineyardStore
from ._zmq import ZmqPull, ZmqPush
