from __future__ import annotations

import gc
from dataclasses import replace

import numpy as np
import pytest

from npb import (
    FRAME_SIZE,
    MIN_FRAME_SIZE,
    BinaryModel,
    VineyardStore,
    binary_schema,
    decode,
    encode,
    encoded_size,
)


class FakeObjectID:
    def __init__(self, value: str) -> None:
        self.value = value

    def __repr__(self) -> str:
        return self.value


class FakeVineyardClient:
    is_ipc = True

    def __init__(self) -> None:
        self.objects: dict[str, np.ndarray] = {}
        self.put_calls = 0
        self.put_persist_values: list[bool] = []
        self.persist_calls: list[str] = []

    def put(self, value: np.ndarray, *, persist: bool = False):
        self.put_calls += 1
        self.put_persist_values.append(persist)
        object_id = f"o{self.put_calls:016x}"
        stored = np.ascontiguousarray(value).copy()
        stored.setflags(write=False)
        self.objects[object_id] = stored
        return FakeObjectID(object_id)

    def get(self, object_id: str):
        return self.objects[object_id]

    def exists(self, object_id: str) -> bool:
        return object_id in self.objects

    def persist(self, object_id: str) -> None:
        self.persist_calls.append(object_id)

    def release_object(self, object_id: str) -> None:
        return None

    def delete(self, object_id: str) -> None:
        self.objects.pop(object_id, None)


class FakeVineyardStore(VineyardStore):
    def _object_id(self, object_id: str):
        return object_id


@binary_schema("tests.vineyard-frame", version=1)
class Frame(BinaryModel):
    name: str
    pixels: np.ndarray


def test_put_persists_high_level_ndarray_object() -> None:
    client = FakeVineyardClient()
    store = FakeVineyardStore(client, persist_on_put=True)

    ref = store.put_array(np.arange(1024, dtype=np.float32))

    assert ref.object_id == "o0000000000000001"
    assert client.put_calls == 1
    assert client.put_persist_values == [True]
    # NPB must not separately call persist() on an internal raw Blob.
    assert client.persist_calls == []


def test_resolved_array_republishes_without_second_put() -> None:
    client = FakeVineyardClient()
    store = FakeVineyardStore(client, persist_on_put=True)

    first = store.put_array(np.arange(1024, dtype=np.float32))
    resolved = store.get_array(first)
    second = store.put_array(resolved)

    assert second == first
    assert client.put_calls == 1
    assert not resolved.flags.writeable


def test_explicit_persist_targets_tensor_object_id() -> None:
    client = FakeVineyardClient()
    store = FakeVineyardStore(client)

    ref = store.put_array(np.arange(32, dtype=np.uint8))
    store.persist(ref)

    assert client.persist_calls == [ref.object_id]


# --- reuse cache lifetime --------------------------------------------------


def test_cache_entry_is_evicted_when_the_resolved_array_is_collected() -> None:
    client = FakeVineyardClient()
    store = FakeVineyardStore(client)

    for size in range(64, 264):
        ref = store.put_array(np.zeros(size, dtype=np.uint8))
        resolved = store.get_array(ref)
        assert store.put_array(resolved) == ref
        del resolved
        client.delete(ref.object_id)  # deleted behind the store's back

    gc.collect()
    assert store._resolved_refs == {}


def test_new_array_at_a_recycled_address_is_never_mistaken_for_an_old_object() -> None:
    client = FakeVineyardClient()
    store = FakeVineyardStore(client)

    for index in range(300):
        frame = np.full(4096, index % 251, dtype=np.uint8)  # brand new each time
        ref = store.put_array(frame)

        # The reference must name a live object holding *this* frame's data.
        assert ref.object_id in client.objects
        assert np.array_equal(client.objects[ref.object_id], frame)

        resolved = store.get_array(ref)
        del resolved
        client.delete(ref.object_id)

    assert client.put_calls == 300


def test_stale_entry_is_ignored_even_before_it_is_evicted() -> None:
    client = FakeVineyardClient()
    store = FakeVineyardStore(client)

    ref = store.put_array(np.arange(1024, dtype=np.float32))
    resolved = store.get_array(ref)

    # A different array object whose address, size, dtype and shape all match
    # the registered key, as happens when freed memory is handed out again.
    impostor = np.arange(1024, dtype=np.float32)
    key = store._key(resolved)
    store._resolved_refs[store._key(impostor)] = store._resolved_refs.pop(key)

    assert store._key(impostor) != key
    assert store.put_array(impostor) != ref
    assert client.put_calls == 2


def test_views_of_a_resolved_array_keep_reusing_its_object() -> None:
    client = FakeVineyardClient()
    store = FakeVineyardStore(client)

    ref = store.put_array(np.arange(1024, dtype=np.float32))
    resolved = store.get_array(ref)
    view = resolved[:]
    del resolved
    gc.collect()

    assert store.put_array(view) == ref
    assert client.put_calls == 1


def test_partial_view_of_a_resolved_array_is_stored_as_a_new_object() -> None:
    client = FakeVineyardClient()
    store = FakeVineyardStore(client)

    ref = store.put_array(np.arange(1024, dtype=np.float32))
    resolved = store.get_array(ref)

    assert store.put_array(resolved[:512]) != ref
    assert client.put_calls == 2


def test_sub_range_reference_is_registered_for_reuse() -> None:
    client = FakeVineyardClient()
    store = FakeVineyardStore(client)

    whole = store.put_array(np.arange(1024, dtype=np.float32))
    half = replace(whole, offset=2048, nbytes=2048, shape=(512,))

    resolved = store.get_array(half)

    assert np.array_equal(resolved, np.arange(512, 1024, dtype=np.float32))
    assert store.put_array(resolved) == half
    assert client.put_calls == 1


@pytest.mark.parametrize("operation", ["delete", "release"])
def test_delete_and_release_forget_the_object(operation: str) -> None:
    client = FakeVineyardClient()
    store = FakeVineyardStore(client)

    ref = store.put_array(np.arange(1024, dtype=np.float32))
    resolved = store.get_array(ref)

    getattr(store, operation)(ref)

    assert store._resolved_refs == {}
    assert store.put_array(resolved) != ref


def test_preview_ref_stores_nothing_and_matches_put() -> None:
    client = FakeVineyardClient()
    store = FakeVineyardStore(client)
    array = np.arange(1024, dtype=np.float32)

    preview = store.preview_ref(array)
    assert client.put_calls == 0

    ref = store.put_array(array)
    assert len(preview.object_id) == len(ref.object_id)
    assert replace(preview, object_id=ref.object_id) == ref

    # Once resolved, the preview is the real reference.
    resolved = store.get_array(ref)
    assert store.preview_ref(resolved) == ref
    assert client.put_calls == 1


def test_encoded_size_with_vineyard_is_exact_and_side_effect_free() -> None:
    client = FakeVineyardClient()
    store = FakeVineyardStore(client)
    frame = Frame(name="cam", pixels=np.zeros((256, 256), dtype=np.uint8))

    for frame_size in (MIN_FRAME_SIZE, FRAME_SIZE):
        size = encoded_size(
            frame, blob_store=store, externalize_min_bytes=0, frame_size=frame_size
        )
        assert client.put_calls == 0

        binary = encode(
            frame, blob_store=store, externalize_min_bytes=0, frame_size=frame_size
        )
        assert binary.nbytes == size
        client.put_calls = 0


def test_long_chain_republish_copies_the_payload_exactly_once() -> None:
    client = FakeVineyardClient()
    store = FakeVineyardStore(client)

    message = encode(
        Frame(name="cam", pixels=np.arange(65536, dtype=np.uint8)),
        blob_store=store,
        externalize_min_bytes=0,
        frame_size=MIN_FRAME_SIZE,
    )
    assert message.nbytes < 512

    for _hop in range(20):
        frame = decode(Frame, message.tobytes(), blob_store=store)
        message = encode(
            frame, blob_store=store, externalize_min_bytes=0, frame_size=MIN_FRAME_SIZE
        )

    assert client.put_calls == 1
    assert len(store._resolved_refs) == 1
