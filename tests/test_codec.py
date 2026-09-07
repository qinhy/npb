from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from npb import (
    FRAME_SIZE,
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


def make_capture() -> Capture:
    return Capture(
        capture_id="cap-001",
        created_at=datetime(2026, 9, 7, tzinfo=timezone.utc),
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
