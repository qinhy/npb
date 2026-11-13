from __future__ import annotations

from datetime import UTC, datetime
from typing import Generic, TypeVar

import numpy as np
import pytest
from pydantic import ConfigDict, ValidationError

from npb import (
    FRAME_SIZE,
    MIN_FRAME_SIZE,
    BinaryModel,
    BufferTooSmallError,
    SchemaMismatchError,
    binary_schema,
    decode,
    decode_auto,
    encode,
    encoded_size,
    peek,
)


class Image(BinaryModel):
    name: str
    pixels: np.ndarray


@binary_schema("tests.capture", version=3)
class Capture(BinaryModel):
    capture_id: str
    created_at: datetime
    meta: dict[str, object]
    images: list[Image]
    calibration: dict[str, np.ndarray]


@binary_schema("tests.other", version=1)
class Other(BinaryModel):
    name: str


@binary_schema("tests.pair", version=1)
class Pair(BinaryModel):
    a: np.ndarray
    b: np.ndarray


@binary_schema("tests.blob", version=1)
class Blob(BinaryModel):
    name: str
    payload: bytes


@binary_schema("tests.meta", version=1)
class Meta(BinaryModel):
    meta: dict[str, object]


@binary_schema("tests.strict-event", version=1)
class StrictEvent(BinaryModel):
    model_config = ConfigDict(strict=True, arbitrary_types_allowed=True)

    at: datetime
    pair: tuple[int, int]


ItemT = TypeVar("ItemT")


@binary_schema("tests.box", version=1)
class Box(BinaryModel, Generic[ItemT]):
    item: ItemT
    data: np.ndarray


def make_capture() -> Capture:
    return Capture(
        capture_id="cap-001",
        created_at=datetime(2026, 9, 7, tzinfo=UTC),
        meta={
            "nested": {
                "enabled": True,
                "threshold": 0.75,
                "labels": ["a", "b"],
            }
        },
        images=[
            Image(
                name="front",
                pixels=np.arange(24, dtype=np.uint8).reshape(2, 4, 3),
            ),
            Image(
                name="depth",
                pixels=np.arange(12, dtype=np.float32).reshape(3, 4),
            ),
        ],
        calibration={
            "matrix": np.eye(3, dtype=np.float64),
        },
    )


def test_typed_round_trip_and_zero_copy() -> None:
    source = make_capture()

    binary = encode(source)
    restored = decode(Capture, binary)

    assert restored.capture_id == source.capture_id
    assert restored.created_at == source.created_at
    assert np.array_equal(restored.images[0].pixels, source.images[0].pixels)
    assert np.array_equal(restored.calibration["matrix"], source.calibration["matrix"])

    assert np.shares_memory(binary, restored.images[0].pixels)
    assert np.shares_memory(binary, restored.images[1].pixels)
    assert np.shares_memory(binary, restored.calibration["matrix"])


def test_decode_auto_returns_plain_tree_with_ndarray_leaves() -> None:
    binary = encode(make_capture())
    data = decode_auto(binary)

    assert type(data) is dict
    assert type(data["images"]) is list
    assert type(data["images"][0]) is dict
    assert isinstance(data["images"][0]["pixels"], np.ndarray)
    assert np.shares_memory(binary, data["images"][0]["pixels"])

    # Generic JSON decode does not know the Pydantic field type.
    assert isinstance(data["created_at"], str)


def test_peek_reads_schema_and_framing() -> None:
    binary = encode(make_capture())
    info = peek(binary)

    assert info.schema_version == 3
    assert info.frame_size == FRAME_SIZE
    assert info.total_bytes == binary.nbytes
    assert info.frame_count == binary.nbytes // FRAME_SIZE
    assert info.data_start % FRAME_SIZE == 0


def test_output_is_frame_aligned() -> None:
    binary = encode(make_capture())

    assert binary.dtype == np.uint8
    assert binary.ndim == 1
    assert binary.flags.c_contiguous
    assert binary.nbytes % FRAME_SIZE == 0

    frames = binary.reshape(-1, FRAME_SIZE)
    assert frames.shape[1] == FRAME_SIZE


def test_preallocated_out_buffer_is_reused() -> None:
    source = make_capture()
    first = encode(source)

    storage = np.empty(first.nbytes + 3 * FRAME_SIZE, dtype=np.uint8)
    binary = encode(source, out=storage)

    assert binary.nbytes == first.nbytes
    assert np.shares_memory(binary, storage)
    assert binary.ctypes.data == storage.ctypes.data

    restored = decode(Capture, binary)
    assert np.array_equal(restored.images[0].pixels, source.images[0].pixels)



def test_full_oversized_pool_buffer_can_be_decoded() -> None:
    source = make_capture()
    required = encode(source).nbytes

    storage = np.empty(required + 12345, dtype=np.uint8)
    encoded_view = encode(source, out=storage)

    # The returned view remains the canonical representation.
    assert encoded_view.nbytes == required

    # But the embedded encoded length makes the whole oversized backing
    # allocation safe to pass to the decoder too.
    restored = decode(Capture, storage)
    info = peek(storage)

    assert info.total_bytes == required
    assert np.array_equal(restored.images[0].pixels, source.images[0].pixels)
    assert np.shares_memory(storage, restored.images[0].pixels)

def test_out_buffer_too_small() -> None:
    source = make_capture()
    required = encode(source).nbytes

    with pytest.raises(BufferTooSmallError):
        encode(source, out=np.empty(required - 1, dtype=np.uint8))


def test_schema_mismatch_is_rejected() -> None:
    binary = encode(make_capture())

    with pytest.raises(SchemaMismatchError):
        decode(Other, binary)


def test_large_metadata_can_span_multiple_frames() -> None:
    @binary_schema("tests.large-meta", version=1)
    class LargeMeta(BinaryModel):
        text: str
        array: np.ndarray

    source = LargeMeta(
        text="x" * (FRAME_SIZE + 1000),
        array=np.arange(100, dtype=np.int64),
    )

    binary = encode(source)
    info = peek(binary)
    restored = decode(LargeMeta, binary)

    assert info.data_start >= 2 * FRAME_SIZE
    assert restored.text == source.text
    assert np.array_equal(restored.array, source.array)


def test_non_contiguous_input_array_is_supported() -> None:
    @binary_schema("tests.noncontiguous", version=1)
    class Matrix(BinaryModel):
        value: np.ndarray

    base = np.arange(100, dtype=np.int32).reshape(10, 10)
    source = Matrix(value=base[:, ::2])

    assert not source.value.flags.c_contiguous

    binary = encode(source)
    restored = decode(Matrix, binary)

    assert restored.value.flags.c_contiguous
    assert np.array_equal(restored.value, source.value)


def test_object_dtype_is_rejected() -> None:
    @binary_schema("tests.object-array", version=1)
    class ObjectArray(BinaryModel):
        value: np.ndarray

    source = ObjectArray(value=np.array([object()], dtype=object))

    with pytest.raises(TypeError, match="object dtypes"):
        encode(source)


def test_encoded_size_matches_encode_without_payload_copy_requirement() -> None:
    source = make_capture()

    size = encoded_size(source)
    binary = encode(source)

    assert size == binary.nbytes


def test_encoded_size_handles_non_contiguous_array() -> None:
    @binary_schema("tests.encoded-size-noncontiguous", version=1)
    class Matrix(BinaryModel):
        value: np.ndarray

    base = np.arange(100, dtype=np.int32).reshape(10, 10)
    source = Matrix(value=base[:, ::2])

    assert not source.value.flags.c_contiguous
    assert encoded_size(source) == encode(source).nbytes


# --- schema identity -------------------------------------------------------


def test_subclass_does_not_inherit_schema_identity() -> None:
    class ExtendedOther(Other):
        extra: int = 0

    with pytest.raises(TypeError, match="no binary schema identity of its own"):
        encode(ExtendedOther(name="x", extra=7))

    with pytest.raises(TypeError, match="no binary schema identity of its own"):
        decode(ExtendedOther, encode(Other(name="x")))


def test_subclass_can_redeclare_its_own_schema_identity() -> None:
    @binary_schema("tests.other-extended", version=1)
    class ExtendedOther(Other):
        extra: int = 0

    binary = encode(ExtendedOther(name="x", extra=7))

    assert decode(ExtendedOther, binary).extra == 7
    with pytest.raises(SchemaMismatchError):
        decode(Other, binary)


def test_parametrized_generic_model_uses_the_identity_of_its_origin() -> None:
    restored = decode(Box[int], encode(Box[int](item=3, data=np.arange(4))))

    assert restored.item == 3
    assert np.array_equal(restored.data, np.arange(4))
    assert peek(encode(Box[str](item="a", data=np.arange(1)))).schema_id == (
        Box.__binary_schema_id__
    )


# --- values that previously failed or were stored inefficiently ------------


def test_non_utf8_bytes_round_trip_as_binary_leaf() -> None:
    payload = bytes(range(256)) * 3
    binary = encode(Blob(name="jpeg", payload=payload))

    assert peek(binary).data_bytes == len(payload)
    assert decode(Blob, binary).payload == payload
    assert decode_auto(binary)["payload"] == payload


def test_utf8_bytes_keep_their_json_representation() -> None:
    binary = encode(Blob(name="text", payload=b"plain ascii"))
    info = peek(binary)

    # Unchanged from NPB 0.3: no DATA section, no extended-manifest flag.
    assert info.data_bytes == 0
    assert info.flags == 0
    assert decode(Blob, binary).payload == b"plain ascii"


def test_strict_model_decodes_when_strict_is_overridden() -> None:
    source = StrictEvent(at=datetime(2026, 1, 1, tzinfo=UTC), pair=(1, 2))
    binary = encode(source)

    with pytest.raises(ValidationError):
        decode(StrictEvent, binary)

    assert decode(StrictEvent, binary, strict=False) == source


@pytest.mark.parametrize(
    "meta",
    [
        {"$blob": "hello"},
        {"$bin": {"kind": "ndarray", "offset": 0, "nbytes": 8, "dtype": "<f8", "shape": [1]}},
        {"$bin": {"kind": "bytes", "offset": 0, "nbytes": 4}},
        {"$$already": 1},
        {"outer": {"$blob": {"$bin": {"$$x": [{"$blob": None}]}}}},
    ],
)
def test_user_dicts_that_look_like_references_round_trip(meta: dict[str, object]) -> None:
    binary = encode(Meta(meta=meta))

    assert peek(binary).flags != 0
    assert decode_auto(binary)["meta"] == meta
    assert decode(Meta, binary).meta == meta


def test_ordinary_dollar_keys_do_not_set_the_extended_flag() -> None:
    meta = {"$ref": "#/defs/a", "nested": {"$id": 1}, "$bin": 1, "other": 2}
    binary = encode(Meta(meta=meta))

    # Multi-key dicts and unreserved "$" keys were never ambiguous.
    assert peek(binary).flags == 0
    assert decode(Meta, binary).meta == meta


def test_same_array_referenced_twice_is_stored_once() -> None:
    shared = np.arange(100_000, dtype=np.uint8)
    binary = encode(Pair(a=shared, b=shared))

    assert peek(binary).data_bytes == shared.nbytes

    restored = decode(Pair, binary)
    assert np.array_equal(restored.a, shared)
    assert np.shares_memory(restored.a, restored.b)

    # Equal contents in distinct objects are still stored separately.
    assert peek(encode(Pair(a=shared, b=shared.copy()))).data_bytes > shared.nbytes


@pytest.mark.parametrize(
    "array",
    [
        np.array(3.5),
        np.empty((0, 3), dtype=np.float32),
        np.arange(6, dtype=">i4").reshape(2, 3),
        np.asfortranarray(np.arange(12, dtype=np.float64).reshape(3, 4)),
        np.arange(60, dtype=np.int16).reshape(3, 4, 5)[::-1, ::2, 1:4],
        np.arange(40, dtype=np.complex64).reshape(4, 10).T,
        np.array(["2026-01-01", "2026-06-30"], dtype="datetime64[s]"),
        np.array(["ab", "c"]),
        np.array([True, False, True]),
    ],
    ids=[
        "0-d", "empty", "big-endian", "fortran", "strided",
        "transposed", "datetime64", "unicode", "bool",
    ],
)
def test_array_layouts_round_trip(array: np.ndarray) -> None:
    out = np.full(encoded_size(Pair(a=array, b=array[...])), 0xAB, dtype=np.uint8)
    restored = decode(Pair, encode(Pair(a=array, b=array[...]), out=out))

    for value in (restored.a, restored.b):
        assert value.dtype == array.dtype
        assert value.shape == array.shape
        assert np.array_equal(value, array)
        assert value.flags.c_contiguous


def test_reused_buffer_produces_identical_bytes_to_a_fresh_one() -> None:
    source = Pair(
        a=np.arange(7, dtype=np.uint8),
        b=np.arange(30, dtype=np.float64).reshape(5, 6)[:, ::2],
    )
    dirty = np.full(encoded_size(source) + 1000, 0xFF, dtype=np.uint8)

    assert np.array_equal(encode(source, out=dirty), encode(source))


# --- inputs ----------------------------------------------------------------


@pytest.mark.parametrize("wrap", [bytes, bytearray, memoryview])
def test_bytes_like_containers_are_accepted_without_copying(wrap) -> None:
    source = make_capture()
    raw = wrap(encode(source).tobytes())

    assert peek(raw).schema_version == 3
    assert decode_auto(raw)["capture_id"] == "cap-001"

    restored = decode(Capture, raw)
    pixels = restored.images[0].pixels
    assert np.array_equal(pixels, source.images[0].pixels)
    assert np.shares_memory(pixels, np.frombuffer(raw, dtype=np.uint8))
    assert pixels.flags.writeable is (wrap is bytearray)


def test_unsupported_container_types_are_rejected() -> None:
    with pytest.raises(TypeError):
        peek("not bytes")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        decode_auto(np.zeros(FRAME_SIZE, dtype=np.uint16))


# --- frame size ------------------------------------------------------------


@pytest.mark.parametrize("frame_size", [MIN_FRAME_SIZE, 256, 4096, FRAME_SIZE, 1 << 20])
def test_custom_frame_size_round_trip(frame_size: int) -> None:
    source = make_capture()
    binary = encode(source, frame_size=frame_size)
    info = peek(binary)

    assert info.frame_size == frame_size
    assert binary.nbytes % frame_size == 0
    assert info.data_start % frame_size == 0
    assert binary.nbytes == encoded_size(source, frame_size=frame_size)
    assert binary.reshape(info.frame_count, frame_size).shape[0] == info.frame_count

    restored = decode(Capture, binary)
    assert restored.capture_id == source.capture_id
    for got, want in zip(restored.images, source.images, strict=True):
        assert np.array_equal(got.pixels, want.pixels)
        assert np.shares_memory(got.pixels, binary)


def test_small_frames_keep_tiny_messages_tiny() -> None:
    message = Other(name="x")

    assert encode(message).nbytes == FRAME_SIZE
    assert encode(message, frame_size=MIN_FRAME_SIZE).nbytes == 2 * MIN_FRAME_SIZE


@pytest.mark.parametrize("frame_size", [0, 32, 100, 65537, 1 << 31, -64])
def test_invalid_frame_size_is_rejected(frame_size: int) -> None:
    with pytest.raises(ValueError, match="frame_size"):
        encode(Other(name="x"), frame_size=frame_size)
    with pytest.raises(ValueError, match="frame_size"):
        encoded_size(Other(name="x"), frame_size=frame_size)


def test_frame_size_must_be_an_integer() -> None:
    with pytest.raises(TypeError):
        encode(Other(name="x"), frame_size=4096.0)  # type: ignore[arg-type]
