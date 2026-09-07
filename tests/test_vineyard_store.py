from __future__ import annotations

import numpy as np

from npb import VineyardStore


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
