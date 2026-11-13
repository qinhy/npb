"""Corrupted or hostile containers must fail with FormatError, nothing else."""

from __future__ import annotations

import contextlib
import json
import random

import numpy as np
import pytest

from npb import (
    FRAME_SIZE,
    BinaryModel,
    FormatError,
    binary_schema,
    decode,
    decode_auto,
    encode,
    peek,
)
from npb._format import HEADER, HEADER_SIZE

FLAGS, FRAME, METADATA_BYTES, DATA_START, DATA_BYTES, TOTAL = 2, 3, 6, 7, 8, 9


@binary_schema("tests.malformed", version=1)
class Sample(BinaryModel):
    name: str
    values: np.ndarray
    raw: bytes


def good() -> np.ndarray:
    return encode(
        Sample(
            name="ok",
            values=np.arange(16, dtype=np.float32),
            raw=b"\xff\x00\xfe",  # forces a bytes leaf and the extended flag
        )
    )


def with_header(index: int, value: int) -> np.ndarray:
    binary = good()
    fields = list(HEADER.unpack_from(binary, 0))
    fields[index] = value
    HEADER.pack_into(binary, 0, *fields)
    return binary


def with_manifest(manifest: object, *, flags: int | None = None) -> np.ndarray:
    binary = good()
    payload = manifest if isinstance(manifest, bytes) else json.dumps(manifest).encode()
    fields = list(HEADER.unpack_from(binary, 0))
    assert HEADER_SIZE + len(payload) <= fields[DATA_START]
    fields[METADATA_BYTES] = len(payload)
    if flags is not None:
        fields[FLAGS] = flags
    HEADER.pack_into(binary, 0, *fields)
    binary[HEADER_SIZE : HEADER_SIZE + len(payload)] = np.frombuffer(payload, dtype=np.uint8)
    return binary


def ndarray_ref(**overrides: object) -> dict[str, object]:
    ref = {"kind": "ndarray", "offset": 0, "nbytes": 8, "dtype": "<f4", "shape": [2]}
    ref.update(overrides)
    return {"values": {"$bin": ref}}


def test_fixture_is_valid() -> None:
    restored = decode(Sample, good())
    assert restored.raw == b"\xff\x00\xfe"
    assert peek(good()).flags == 1


@pytest.mark.parametrize(
    ("index", "value"),
    [
        (0, b"NPB2"),
        (1, 2),
        (FLAGS, 0x8000),
        (FLAGS, 0x0003),
        (FRAME, 0),
        (FRAME, 63),
        (FRAME, 3 * 1024),
        (FRAME, 1 << 31),
        (METADATA_BYTES, 2**32 - 1),
        (DATA_START, 2**63),
        (DATA_START, FRAME_SIZE + 64),
        (DATA_BYTES, 2**64 - 1),
        (TOTAL, 0),
        (TOTAL, FRAME_SIZE * 3),
        (TOTAL, FRAME_SIZE + 1),
    ],
)
def test_corrupt_header_is_rejected(index: int, value: object) -> None:
    binary = with_header(index, value)  # type: ignore[arg-type]

    with pytest.raises(FormatError):
        peek(binary)
    with pytest.raises(FormatError):
        decode_auto(binary)


def test_truncated_container_is_rejected() -> None:
    binary = good()

    for size in (0, 1, HEADER_SIZE - 1, HEADER_SIZE, FRAME_SIZE, binary.nbytes - 1):
        with pytest.raises(FormatError):
            decode_auto(binary[:size])


@pytest.mark.parametrize(
    "manifest",
    [
        b"",
        b"{",
        b"\xff\xfe",
        b"[" * 5000 + b"]" * 5000,
        b'{"a":' * 5000 + b"1" + b"}" * 5000,
        [],
        "text",
        ndarray_ref(offset=1e400),
        ndarray_ref(offset=1.0),
        ndarray_ref(offset="0"),
        ndarray_ref(offset=True),
        ndarray_ref(offset=-4),
        ndarray_ref(offset=10**30),
        ndarray_ref(nbytes=-8),
        ndarray_ref(nbytes=7),
        ndarray_ref(nbytes=10**30, shape=[10**30 // 4]),
        ndarray_ref(shape=[3]),
        ndarray_ref(shape=[-2]),
        ndarray_ref(shape=2),
        ndarray_ref(shape=[2.0]),
        ndarray_ref(shape=[[2]]),
        ndarray_ref(nbytes=0, shape=[0, 10**23]),
        ndarray_ref(dtype="O"),
        ndarray_ref(dtype="<f4,<f4"),
        ndarray_ref(dtype="(2,)<f4", shape=[1]),
        ndarray_ref(dtype="S0", nbytes=0),
        ndarray_ref(dtype="S99999999999999999999"),
        ndarray_ref(dtype="nope"),
        ndarray_ref(dtype="(,"),
        ndarray_ref(dtype=7),
        ndarray_ref(dtype=None),
        ndarray_ref(dtype=[["x", "<f4"]]),
        ndarray_ref(dtype={"names": ["x"], "formats": ["<f4"]}),
        {"values": {"$bin": {"kind": "ndarray"}}},
        {"values": {"$bin": {"kind": "bytes", "offset": 0, "nbytes": 10**9}}},
        {"values": {"$bin": {"kind": "bytes", "offset": 0}}},
        {"values": {"$bin": {"kind": "mystery", "offset": 0, "nbytes": 1}}},
        {"values": {"$bin": "oops"}},
        {"values": {"$blob": "oops"}},
        {"values": {"$blob": {"store": "s", "id": "x", "offset": 1e400, "nbytes": 0,
                              "dtype": "<f4", "shape": []}}},
        {"values": {"$blob": {"store": "s", "id": "x", "offset": 0, "nbytes": 4,
                              "dtype": "(,", "shape": [1]}}},
        {"values": {"$blob": {"store": "", "id": "x", "offset": 0, "nbytes": 4,
                              "dtype": "<f4", "shape": [1]}}},
    ],
)
def test_corrupt_manifest_is_rejected(manifest: object) -> None:
    with pytest.raises(FormatError):
        decode_auto(with_manifest(manifest))


def test_unflagged_manifests_keep_the_legacy_lenient_reading() -> None:
    # NPB <= 0.3 wrote user dictionaries verbatim, so in a container without
    # the extended flag an unknown "$bin" kind or a "$$" key is plain data.
    manifest = {"a": {"$bin": {"kind": "mystery"}}, "b": {"$$x": 1}, "c": {"$bin": 5}}

    assert decode_auto(with_manifest(manifest, flags=0)) == manifest


def _only_format_errors(binary: np.ndarray) -> None:
    # Any other exception type propagates and fails the test.
    with contextlib.suppress(FormatError):
        decode_auto(binary)


def test_fuzzed_headers_only_raise_format_error() -> None:
    rng = random.Random(0xB10B)
    template = good()

    for _ in range(3000):
        binary = template.copy()
        for _ in range(rng.randint(1, 4)):
            position = rng.randrange(HEADER_SIZE)
            binary[position] = rng.choice([0, 1, 0x7F, 0x80, 0xFF, rng.randrange(256)])
        _only_format_errors(binary)


def test_fuzzed_manifests_only_raise_format_error() -> None:
    rng = random.Random(0x5EED)
    template = good()
    end = HEADER_SIZE + peek(template).metadata_bytes
    interesting = b'0123456789-.eE"{}[]:,$ntfO<>|()'

    for _ in range(6000):
        binary = template.copy()
        for _ in range(rng.randint(1, 3)):
            position = rng.randrange(HEADER_SIZE, end)
            binary[position] = rng.choice(interesting)
        _only_format_errors(binary)


def test_fuzzed_reference_fields_only_raise_format_error() -> None:
    rng = random.Random(0xF00D)
    junk: list[object] = [
        None, True, False, 0, -1, 1, 7, 2**31, 2**63, 2**64, 10**40, 0.5, -0.0, 1e308,
        "", "0", "<f4", "|u1", ">i8", "<M8[ns]", "<U3", "|V4", "O", "i4,i4", "(3,)u1",
        [], [0], [1, 2], [-1], [2**40, 2**40], [[1]], {}, {"kind": "ndarray"},
    ]
    kinds = ["ndarray", "bytes", "blob", 3, None]

    for _ in range(6000):
        ref = {
            key: rng.choice(junk)
            for key in ("offset", "nbytes", "dtype", "shape", "store", "id")
            if rng.random() < 0.85
        }
        ref["kind"] = rng.choice(kinds)
        marker = rng.choice(["$bin", "$blob", "$$bin", "$other"])
        manifest = {"values": {marker: ref}, "list": [{marker: ref}]}
        _only_format_errors(with_manifest(manifest, flags=rng.choice([0, 1])))
