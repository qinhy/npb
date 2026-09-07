"""External immutable blob references and storage protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


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
    def from_manifest(cls, value: object) -> "BlobRef":
        if not isinstance(value, dict):
            raise TypeError("blob reference must be an object")

        try:
            store = str(value["store"])
            object_id = str(value["id"])
            offset = int(value["offset"])
            nbytes = int(value["nbytes"])
            dtype = str(value["dtype"])
            shape = tuple(int(dim) for dim in value["shape"])
        except (KeyError, TypeError, ValueError) as exc:
            raise TypeError("invalid blob reference") from exc

        if not store:
            raise ValueError("blob store name cannot be empty")
        if not object_id:
            raise ValueError("blob object id cannot be empty")
        if offset < 0 or nbytes < 0:
            raise ValueError("blob offset and size must be non-negative")
        if any(dim < 0 for dim in shape):
            raise ValueError("blob ndarray shape cannot contain negative dimensions")

        np_dtype = np.dtype(dtype)

        if np_dtype.hasobject or np_dtype.fields is not None:
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
    """Storage backend used by NPB for immutable ndarray payloads."""

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
