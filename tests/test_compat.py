"""Wire compatibility with containers written by NPB 0.3.x."""

from __future__ import annotations

import numpy as np

from npb import (
    FRAME_SIZE,
    BinaryModel,
    binary_schema,
    decode,
    decode_auto,
    encode,
    peek,
)

# Header + manifest, and DATA section, of a container produced by NPB 0.3.1 for
# Frame(name="legacy", samples=np.arange(6, dtype="<i2").reshape(2, 3)).
# Everything else in the two 64 KiB frames is zero padding.
LEGACY_HEAD = bytes.fromhex(
    "4e5042310100000000000100e847f0bb66a4509a9144f9f744b9498e020000006a000000"
    "00000100000000000c00000000000000000002000000000000000000"
    "7b226e616d65223a226c6567616379222c2273616d706c6573223a7b222462696e223a7b"
    "226b696e64223a226e646172726179222c226f6666736574223a302c226e627974657322"
    "3a31322c226474797065223a223c6932222c227368617065223a5b322c335d7d7d7d"
)
LEGACY_DATA = bytes.fromhex("000001000200030004000500")


@binary_schema("compat.frame", version=2)
class Frame(BinaryModel):
    name: str
    samples: np.ndarray


def legacy_container() -> bytes:
    raw = bytearray(2 * FRAME_SIZE)
    raw[: len(LEGACY_HEAD)] = LEGACY_HEAD
    raw[FRAME_SIZE : FRAME_SIZE + len(LEGACY_DATA)] = LEGACY_DATA
    return bytes(raw)


def test_container_written_by_0_3_1_still_decodes() -> None:
    raw = legacy_container()
    info = peek(raw)

    assert info.format_version == 1
    assert info.flags == 0
    assert info.frame_size == FRAME_SIZE
    assert info.schema_version == 2

    restored = decode(Frame, raw)
    assert restored.name == "legacy"
    assert restored.samples.dtype == np.dtype("<i2")
    assert np.array_equal(restored.samples, np.arange(6).reshape(2, 3))
    assert decode_auto(raw)["name"] == "legacy"


def test_default_encoding_is_byte_identical_to_0_3_1() -> None:
    source = Frame(name="legacy", samples=np.arange(6, dtype="<i2").reshape(2, 3))

    assert encode(source).tobytes() == legacy_container()
