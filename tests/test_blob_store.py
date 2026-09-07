from __future__ import annotations

from dataclasses import replace
import uuid

import numpy as np
import pytest

from npb import (
    BlobRef,
    BlobStoreRequiredError,
    BinaryModel,
    binary_schema,
    decode,
    decode_auto,
    encode,
    peek,
)


class FakeBlobStore:
    kind = "fake"

    def __init__(self) -> None:
        self.objects: dict[str, np.ndarray] = {}
        self.put_calls = 0

    def put_array(self, array: np.ndarray) -> BlobRef:
        self.put_calls += 1

        # Reuse an existing object when this array already shares its storage.
        for object_id, stored in self.objects.items():
            if np.shares_memory(array, stored):
                offset = (
                    int(array.__array_interface__["data"][0])
                    - int(stored.__array_interface__["data"][0])
                )
                return BlobRef(
                    store=self.kind,
                    object_id=object_id,
                    offset=offset,
                    nbytes=array.nbytes,
                    dtype=array.dtype.str,
                    shape=tuple(array.shape),
                )

        object_id = uuid.uuid4().hex
        stored = np.ascontiguousarray(array).copy()
        stored.setflags(write=False)
        self.objects[object_id] = stored

        return BlobRef(
            store=self.kind,
            object_id=object_id,
            offset=0,
            nbytes=stored.nbytes,
            dtype=stored.dtype.str,
            shape=tuple(stored.shape),
        )

    def get_array(self, ref: BlobRef) -> np.ndarray:
        base = self.objects[ref.object_id]
        return np.frombuffer(
            base,
            dtype=np.dtype(ref.dtype),
            count=ref.nbytes // np.dtype(ref.dtype).itemsize,
            offset=ref.offset,
        ).reshape(ref.shape)

    def exists(self, ref: BlobRef) -> bool:
        return ref.object_id in self.objects

    def persist(self, ref: BlobRef) -> None:
        return None

    def release(self, ref: BlobRef) -> None:
        return None

    def delete(self, ref: BlobRef) -> None:
        self.objects.pop(ref.object_id, None)


@binary_schema("tests.external-blob", version=1)
class Payload(BinaryModel):
    name: str
    small: np.ndarray
    huge: np.ndarray


def make_payload() -> Payload:
    return Payload(
        name="demo",
        small=np.arange(16, dtype=np.int32),
        huge=np.arange(1024 * 1024, dtype=np.float32),
    )


def test_large_array_can_be_externalized() -> None:
    store = FakeBlobStore()
    source = make_payload()

    binary = encode(
        source,
        blob_store=store,
        externalize_min_bytes=1024,
    )

    # The huge 4 MiB ndarray is no longer inside the NPB container.
    assert binary.nbytes < 256 * 1024
    assert len(store.objects) == 1

    restored = decode(Payload, binary, blob_store=store)

    assert np.array_equal(restored.small, source.small)
    assert np.array_equal(restored.huge, source.huge)
    assert restored.huge.flags.writeable is False


def test_decode_auto_without_store_returns_blob_ref() -> None:
    store = FakeBlobStore()
    binary = encode(
        make_payload(),
        blob_store=store,
        externalize_min_bytes=1024,
    )

    generic = decode_auto(binary)

    assert isinstance(generic["huge"], BlobRef)
    assert generic["huge"].store == "fake"


def test_typed_decode_requires_store_for_external_refs() -> None:
    store = FakeBlobStore()
    binary = encode(
        make_payload(),
        blob_store=store,
        externalize_min_bytes=1024,
    )

    with pytest.raises(BlobStoreRequiredError):
        decode(Payload, binary)


def test_republish_reuses_store_backing_instead_of_copying_payload_again() -> None:
    store = FakeBlobStore()
    source = make_payload()

    first = encode(
        source,
        blob_store=store,
        externalize_min_bytes=1024,
    )

    first_decoded = decode(Payload, first, blob_store=store)

    object_count_before = len(store.objects)

    second = encode(
        first_decoded,
        blob_store=store,
        externalize_min_bytes=1024,
    )

    # put_array() is asked again, but recognizes the ndarray as already backed
    # by the store, so no second huge object is allocated.
    assert len(store.objects) == object_count_before

    second_decoded = decode(Payload, second, blob_store=store)
    assert np.shares_memory(first_decoded.huge, second_decoded.huge)


def test_externalize_zero_moves_all_arrays_out_of_npb() -> None:
    store = FakeBlobStore()
    source = make_payload()

    binary = encode(
        source,
        blob_store=store,
        externalize_min_bytes=0,
    )

    generic = decode_auto(binary)

    assert isinstance(generic["small"], BlobRef)
    assert isinstance(generic["huge"], BlobRef)
    assert len(store.objects) == 2
