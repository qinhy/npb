from __future__ import annotations

import uuid

import numpy as np
import pytest

from npb import (
    MIN_FRAME_SIZE,
    BinaryModel,
    BlobRef,
    BlobStore,
    BlobStoreRequiredError,
    BufferTooSmallError,
    binary_schema,
    decode,
    decode_auto,
    encode,
    encoded_size,
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


class PreviewingBlobStore(FakeBlobStore):
    """A store that implements the optional ``preview_ref`` hook."""

    kind = "previewing"

    def __init__(self) -> None:
        super().__init__()
        self.fail_on_put: int | None = None
        self.deleted: list[str] = []

    def put_array(self, array: np.ndarray) -> BlobRef:
        if self.fail_on_put is not None and self.put_calls + 1 >= self.fail_on_put:
            self.put_calls += 1
            raise RuntimeError("store is full")
        return super().put_array(array)

    def preview_ref(self, array: np.ndarray) -> BlobRef:
        object_id, offset = "0" * 32, 0
        for existing_id, stored in self.objects.items():
            if np.shares_memory(array, stored):
                object_id = existing_id
                offset = (
                    int(array.__array_interface__["data"][0])
                    - int(stored.__array_interface__["data"][0])
                )
                break

        return BlobRef(
            store=self.kind,
            object_id=object_id,
            offset=offset,
            nbytes=array.nbytes,
            dtype=array.dtype.str,
            shape=tuple(array.shape),
        )

    def delete(self, ref: BlobRef) -> None:
        self.deleted.append(ref.object_id)
        super().delete(ref)


@binary_schema("tests.external-blob", version=1)
class Payload(BinaryModel):
    name: str
    small: np.ndarray
    huge: np.ndarray


@binary_schema("tests.republished", version=1)
class Republished(BinaryModel):
    name: str
    huge: np.ndarray
    extra: np.ndarray


@binary_schema("tests.annotated", version=1)
class Annotated(BinaryModel):
    meta: dict[str, object]


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


# --- sizing and failure handling -------------------------------------------


@pytest.mark.parametrize("store_type", [FakeBlobStore, PreviewingBlobStore])
def test_encoded_size_does_not_write_to_the_store(store_type: type[FakeBlobStore]) -> None:
    store = store_type()
    source = make_payload()

    size = encoded_size(source, blob_store=store, externalize_min_bytes=1024)

    assert store.put_calls == 0
    assert store.objects == {}

    binary = encode(source, blob_store=store, externalize_min_bytes=1024)

    assert binary.nbytes == size
    assert store.put_calls == 1
    assert len(store.objects) == 1


def test_encoded_size_is_exact_for_already_resolved_arrays() -> None:
    store = PreviewingBlobStore()
    first = encode(make_payload(), blob_store=store, externalize_min_bytes=1024)
    resolved = decode(Payload, first, blob_store=store)
    puts_before = store.put_calls

    size = encoded_size(
        resolved,
        blob_store=store,
        externalize_min_bytes=1024,
        frame_size=MIN_FRAME_SIZE,
    )

    assert store.put_calls == puts_before
    assert size == encode(
        resolved,
        blob_store=store,
        externalize_min_bytes=1024,
        frame_size=MIN_FRAME_SIZE,
    ).nbytes


def test_old_style_store_without_preview_hook_still_satisfies_the_protocol() -> None:
    assert isinstance(FakeBlobStore(), BlobStore)
    assert not hasattr(FakeBlobStore(), "preview_ref")


@pytest.mark.parametrize("store_type", [FakeBlobStore, PreviewingBlobStore])
def test_bad_output_buffer_is_rejected_before_the_store_is_touched(
    store_type: type[FakeBlobStore],
) -> None:
    store = store_type()

    with pytest.raises(TypeError):
        encode(
            make_payload(),
            out=np.empty(1 << 20, dtype=np.uint16),
            blob_store=store,
            externalize_min_bytes=1024,
        )

    assert store.put_calls == 0


def test_too_small_output_buffer_is_rejected_before_the_store_is_touched() -> None:
    store = PreviewingBlobStore()

    with pytest.raises(BufferTooSmallError):
        encode(
            make_payload(),
            out=np.empty(16, dtype=np.uint8),
            blob_store=store,
            externalize_min_bytes=1024,
        )

    assert store.put_calls == 0
    assert store.objects == {}


@pytest.mark.parametrize("store_type", [FakeBlobStore, PreviewingBlobStore])
def test_unencodable_model_is_rejected_before_the_store_is_touched(
    store_type: type[FakeBlobStore],
) -> None:
    store = store_type()

    # The unsupported array comes *after* a perfectly good large one.
    broken = Payload(
        name="demo",
        small=np.arange(2048, dtype=np.int32),
        huge=np.array([object()], dtype=object),
    )

    with pytest.raises(TypeError, match="object dtypes"):
        encode(broken, blob_store=store, externalize_min_bytes=0)

    assert store.put_calls == 0
    assert store.objects == {}


def test_objects_created_before_a_store_failure_are_rolled_back() -> None:
    store = PreviewingBlobStore()
    store.fail_on_put = 2  # "small" is stored, then "huge" fails

    with pytest.raises(RuntimeError, match="store is full"):
        encode(make_payload(), blob_store=store, externalize_min_bytes=0)

    assert len(store.deleted) == 1
    assert store.objects == {}


def test_rollback_never_deletes_reused_objects() -> None:
    store = PreviewingBlobStore()
    first = encode(make_payload(), blob_store=store, externalize_min_bytes=1024)
    resolved = decode(Payload, first, blob_store=store)
    upstream_objects = set(store.objects)

    # Re-publishing reuses "huge"; the new "extra" array then fails to store.
    republished = Republished(name="next", huge=resolved.huge, extra=np.zeros(4096))
    store.fail_on_put = store.put_calls + 2

    with pytest.raises(RuntimeError, match="store is full"):
        encode(republished, blob_store=store, externalize_min_bytes=1024)

    assert store.deleted == []
    assert set(store.objects) == upstream_objects


def test_same_array_referenced_twice_is_externalized_once() -> None:
    store = FakeBlobStore()
    shared = np.arange(4096, dtype=np.float64)

    binary = encode(
        Republished(name="twice", huge=shared, extra=shared),
        blob_store=store,
        externalize_min_bytes=1024,
    )

    assert store.put_calls == 1
    generic = decode_auto(binary)
    assert generic["huge"] == generic["extra"]


def test_user_dict_shaped_like_a_blob_reference_is_not_resolved() -> None:
    store = FakeBlobStore()
    real = store.put_array(np.arange(4, dtype=np.int64))
    forged = {"$blob": real.to_manifest()}

    binary = encode(Annotated(meta=forged))
    generic = decode_auto(binary, blob_store=store)

    # Plain user data stays plain data, even when a matching store is given.
    assert generic["meta"] == forged
    assert decode(Annotated, binary, blob_store=store).meta == forged

