"""Optional Vineyard blob-store backend."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._blob import BlobRef
from ._errors import BlobStoreError


def _load_vineyard():
    try:
        import vineyard
    except ImportError as exc:
        raise ImportError(
            "Vineyard support is optional. Install it with "
            "`uv add 'npb[vineyard]'` or `pip install 'npb[vineyard]'`."
        ) from exc
    return vineyard


class VineyardStore:
    """NPB external-array store backed by a local Vineyard IPC client.

    NPB references a Vineyard *Tensor/ndarray object ID*, not the raw Blob ID
    that physically backs the tensor. This distinction is important because
    Vineyard Tensor objects can be persisted while raw Blob objects cannot.

    Arrays returned by :meth:`get_array` are registered locally. Re-encoding
    the same resolved ndarray therefore reuses its Vineyard object ID and does
    not copy its multi-GiB payload again.
    """

    kind = "vineyard"

    def __init__(
        self,
        client: Any,
        *,
        persist_on_put: bool = False,
        copy_concurrency: int = 6,
    ) -> None:
        if not getattr(client, "is_ipc", False):
            raise ValueError(
                "VineyardStore requires a local IPC client; RPC clients cannot "
                "provide same-host zero-copy ndarray access"
            )

        if copy_concurrency <= 0:
            raise ValueError("copy_concurrency must be > 0")

        self.client = client
        self.persist_on_put = persist_on_put

        # Kept in the API for compatibility with NPB 0.2/0.3. Vineyard's
        # high-level ndarray builder owns its copy strategy; NPB no longer
        # creates and persists raw Blob objects itself.
        self.copy_concurrency = copy_concurrency

        # Exact ndarray views resolved by this client -> Vineyard object ref.
        # This is deliberately process-local. Each pipeline stage decodes the
        # object first, registering the shared-memory view, then its next
        # encode() can reuse the same object ID without another huge copy.
        self._resolved_refs: dict[
            tuple[int, int, str, tuple[int, ...]], BlobRef
        ] = {}

    @classmethod
    def connect(
        cls,
        socket: str | None = None,
        *,
        persist_on_put: bool = False,
        copy_concurrency: int = 6,
    ) -> "VineyardStore":
        """Connect to a local Vineyard daemon."""
        vineyard = _load_vineyard()

        client = vineyard.connect(socket) if socket is not None else vineyard.connect()

        return cls(
            client,
            persist_on_put=persist_on_put,
            copy_concurrency=copy_concurrency,
        )

    def _object_id(self, object_id: str):
        vineyard = _load_vineyard()
        return vineyard.ObjectID(object_id)

    @staticmethod
    def _validate_array(array: np.ndarray) -> None:
        if not isinstance(array, np.ndarray):
            raise TypeError("VineyardStore only stores numpy.ndarray values")
        if array.dtype.hasobject:
            raise TypeError("NumPy object dtypes are not supported")
        if array.dtype.fields is not None:
            raise TypeError("NumPy structured dtypes are not supported")

    @staticmethod
    def _key(array: np.ndarray) -> tuple[int, int, str, tuple[int, ...]]:
        return (
            int(array.__array_interface__["data"][0]),
            int(array.nbytes),
            array.dtype.str,
            tuple(array.shape),
        )

    @staticmethod
    def _ref_for_object(object_id: object, array: np.ndarray, *, offset: int = 0) -> BlobRef:
        return BlobRef(
            store="vineyard",
            object_id=repr(object_id),
            offset=offset,
            nbytes=array.nbytes,
            dtype=array.dtype.str,
            shape=tuple(array.shape),
        )

    def _register_resolved(self, array: np.ndarray, ref: BlobRef) -> None:
        self._resolved_refs[self._key(array)] = ref

    def _try_reuse_resolved(self, array: np.ndarray) -> BlobRef | None:
        """Reuse an ndarray previously resolved by this VineyardStore."""
        return self._resolved_refs.get(self._key(array))

    def put_array(self, array: np.ndarray) -> BlobRef:
        """Store an ndarray or reuse an already-resolved Vineyard object.

        The first put performs the unavoidable one-time copy into Vineyard.
        Subsequent encode() calls on arrays returned by get_array() reuse the
        same Vineyard Tensor object ID without copying the payload again.
        """
        self._validate_array(array)

        existing = self._try_reuse_resolved(array)
        if existing is not None:
            return existing

        # Vineyard's ndarray builder expects a contiguous value for a compact
        # tensor payload. This copy only occurs here when the user's source is
        # non-contiguous; normal contiguous camera/tensor arrays are unchanged.
        arr = np.ascontiguousarray(array)

        # IMPORTANT: client.put(np.ndarray) creates a high-level Vineyard
        # Tensor/ndarray object whose metadata references an internal Blob.
        # Persist the Tensor object, never the raw Blob. Vineyard explicitly
        # rejects persist(blob_id).
        object_id = self.client.put(arr, persist=self.persist_on_put)

        return self._ref_for_object(object_id, arr)

    def get_array(self, ref: BlobRef) -> np.ndarray:
        """Resolve a Vineyard Tensor object as a zero-copy ndarray view."""
        if ref.store != self.kind:
            raise BlobStoreError(
                f"reference requires store {ref.store!r}, not {self.kind!r}"
            )

        value = self.client.get(self._object_id(ref.object_id))

        if not isinstance(value, np.ndarray):
            raise BlobStoreError(
                "Vineyard object referenced by NPB did not resolve to numpy.ndarray"
            )

        if not value.flags.c_contiguous:
            raise BlobStoreError(
                "Vineyard ndarray backing object is unexpectedly non-contiguous"
            )

        # The object ID normally denotes the complete ndarray (offset == 0).
        # Keeping offset in BlobRef preserves the generic store contract and
        # permits future sub-view support without changing the wire format.
        raw_nbytes = int(value.nbytes)
        if ref.offset < 0 or ref.offset + ref.nbytes > raw_nbytes:
            raise BlobStoreError("blob reference exceeds Vineyard ndarray size")

        dtype = np.dtype(ref.dtype)
        count = ref.nbytes // dtype.itemsize

        if ref.offset == 0 and ref.nbytes == raw_nbytes and value.dtype == dtype:
            # reshape() is view-only when the element count is unchanged.
            array = value.reshape(ref.shape)
        else:
            byte_view = value.view(np.uint8).reshape(-1)
            array = np.frombuffer(
                byte_view,
                dtype=dtype,
                count=count,
                offset=ref.offset,
            ).reshape(ref.shape)

        # Vineyard objects are immutable. Mark the Python view read-only as an
        # explicit guard even if the underlying resolver already does so.
        try:
            array.setflags(write=False)
        except ValueError:
            pass

        self._register_resolved(array, ref)
        return array

    def exists(self, ref: BlobRef) -> bool:
        return bool(self.client.exists(self._object_id(ref.object_id)))

    def persist(self, ref: BlobRef) -> None:
        # ref.object_id is a Tensor/ndarray object ID, not a raw Blob ID.
        self.client.persist(self._object_id(ref.object_id))

    def release(self, ref: BlobRef) -> None:
        self.client.release_object(self._object_id(ref.object_id))

    def delete(self, ref: BlobRef) -> None:
        self.client.delete(self._object_id(ref.object_id))
