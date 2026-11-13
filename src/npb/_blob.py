"""External immutable blob references and storage protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

#: Object-ID length assumed by ``encoded_size`` for stores that do not
#: implement the optional ``preview_ref`` hook (a UUID in hex).
DEFAULT_OBJECT_ID_LENGTH = 32


@dataclass(frozen=True, slots=True)
class BlobRef:
    """Reference to an ndarray payload stored outside the NPB container."""

    store: str
    object_id: str
    offset: int
    nbytes: int
    dtype: str
    shape: tuple[int, ...]

    def to_manifest(self) -> dict[str, object]:
        return {
            "store": self.store,
            "id": self.object_id,
            "offset": self.offset,
            "nbytes": self.nbytes,
            "dtype": self.dtype,
            "shape": list(self.shape),
        }

    @classmethod
    def from_manifest(cls, value: object) -> BlobRef:
        if not isinstance(value, dict):
            raise TypeError("blob reference must be an object")

        try:
            store = str(value["store"])
            object_id = str(value["id"])
            offset = int(value["offset"])
            nbytes = int(value["nbytes"])
            dtype = str(value["dtype"])
            shape = tuple(int(dim) for dim in value["shape"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise TypeError("invalid blob reference") from exc

        if not store:
            raise ValueError("blob store name cannot be empty")
        if not object_id:
            raise ValueError("blob object id cannot be empty")
        if offset < 0 or nbytes < 0:
            raise ValueError("blob offset and size must be non-negative")
        if any(dim < 0 for dim in shape):
            raise ValueError("blob ndarray shape cannot contain negative dimensions")

        try:
            np_dtype = np.dtype(dtype)
        except SyntaxError as exc:  # raised by NumPy for some malformed strings
            raise ValueError("invalid dtype in blob reference") from exc

        if (
            np_dtype.hasobject
            or np_dtype.fields is not None
            or np_dtype.subdtype is not None
        ):
            raise ValueError("unsupported dtype in blob reference")

        expected = np_dtype.itemsize
        for dim in shape:
            expected *= dim

        if expected != nbytes:
            raise ValueError("blob ndarray shape does not match nbytes")

        return cls(
            store=store,
            object_id=object_id,
            offset=offset,
            nbytes=nbytes,
            dtype=dtype,
            shape=shape,
        )


@runtime_checkable
class BlobStore(Protocol):
    """Storage backend used by NPB for immutable ndarray payloads.

    Optional hook
    -------------
    A store may additionally implement::

        def preview_ref(self, array: np.ndarray) -> BlobRef: ...

    ``preview_ref`` must not store anything. It returns the existing
    reference when ``put_array(array)`` would reuse an object that is already
    in the store, and otherwise a placeholder whose ``object_id`` has the same
    length as a real one. NPB uses it to

    * keep :func:`npb.encoded_size` exact and free of side effects, and
    * tell newly created objects from reused ones, so that objects created by
      an ``encode()`` call that subsequently fails can be deleted again.

    The hook is deliberately not part of this protocol's required surface, so
    stores written for NPB <= 0.3 keep satisfying ``isinstance(store,
    BlobStore)``. Without it, ``encoded_size`` assumes
    :data:`DEFAULT_OBJECT_ID_LENGTH`-character object IDs and failed encodes
    are not rolled back.
    """

    @property
    def kind(self) -> str:
        """Stable backend identifier written into the NPB manifest."""
        ...

    def put_array(self, array: np.ndarray) -> BlobRef:
        """Store or reuse an ndarray and return a stable reference."""
        ...

    def get_array(self, ref: BlobRef) -> np.ndarray:
        """Resolve a reference to an ndarray, ideally without copying."""
        ...

    def exists(self, ref: BlobRef) -> bool:
        """Return whether the referenced object still exists."""
        ...

    def persist(self, ref: BlobRef) -> None:
        """Make the referenced object durable/cluster-visible when supported."""
        ...

    def release(self, ref: BlobRef) -> None:
        """Release this client's hold on the object when supported."""
        ...

    def delete(self, ref: BlobRef) -> None:
        """Delete the referenced object when supported."""
        ...
