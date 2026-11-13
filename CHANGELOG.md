# Changelog

## 0.4.0

The default wire format is unchanged: for everything 0.3 could encode, 0.4
writes byte-identical containers, and it reads every 0.3 container.

### Fixed

- `encoded_size(..., blob_store=...)` no longer writes arrays to the blob
  store. Sizing a message and then encoding it used to store every
  externalized array twice and orphan the first copy.
- `encode()` now validates dtypes, JSON serializability and the `out` buffer
  before the first array is stored, so a rejected encode leaves nothing behind
  in the store. If the store fails part-way, objects created by that call are
  deleted again (stores with `preview_ref`; reused objects are never deleted).
- `VineyardStore` could hand back the reference of a deleted object for a
  brand-new array that happened to occupy a recycled address, and its reuse
  cache grew without bound. Registrations are now tied to the lifetime of the
  resolved view, checked against the owning mapping, and dropped on
  `delete()` and `release()`.
- User dictionaries whose only key is `$bin`, `$blob` or starts with `$$`
  encoded successfully but then failed to decode, or were resolved as real
  blob-store objects. They are now escaped and round-trip as plain data.
- Malformed containers now always raise `FormatError`. Overflowing offsets,
  impossible shapes and some malformed dtype strings previously leaked
  `OverflowError`, `ValueError` or `SyntaxError`; non-string and sub-array
  dtypes are now rejected outright.

### Added

- `frame_size=` for `encode()`, `encoded_size()` and `ZmqPush.send_model()`,
  plus `MIN_FRAME_SIZE`. Reference-only control messages can shrink from
  64 KiB to a few hundred bytes.
- Non-UTF-8 `bytes` fields are stored as raw binary leaves instead of failing.
- `decode(..., strict=...)` and `recv_model(..., strict=...)` for models
  configured with `strict=True`.
- `peek()`, `decode()` and `decode_auto()` accept `bytes`, `bytearray` and
  `memoryview` without copying. `BinaryLike` names the accepted types.
- `BinaryInfo.flags`.
- Optional `BlobStore.preview_ref()` hook; implemented by `VineyardStore`.
- `py.typed`, so the type hints advertised by the classifiers are shipped.

### Changed

- A subclass no longer inherits its parent's `@binary_schema` identity; it
  was silently decoded as the parent, dropping its extra fields. Decorate the
  subclass, reusing the parent's name and version only if both serialize
  identically. Parametrized generics (`Model[int]`) keep using the identity
  of their origin class.
- An array object referenced from several fields is stored (or externalized)
  once instead of once per field.
- Manifest integers are no longer coerced from floats or strings.
- Strided and Fortran-ordered arrays are copied into the container once,
  without an intermediate contiguous copy. Encoding and decoding
  metadata-heavy models is faster; the tiniest messages pay under a
  microsecond more per encode for the pre-flight bookkeeping.

### Housekeeping

- `ruff check .` passes again (it reported 63 findings), `ruff` is pinned to
  the 0.17 series for reproducible CI, and the examples are exempt from two
  purely stylistic rules.
- 131 new tests, including a 0.3.1 fixture, a malformed-container corpus and
  seeded fuzzing of headers and manifests.

## 0.3.1

- Fixed Vineyard persistence: NPB now stores/persists Vineyard Tensor/ndarray
  object IDs rather than raw Blob IDs. Vineyard raw blobs cannot be persisted.
- `VineyardStore.get_array()` now resolves the high-level ndarray object with
  `client.get()` and registers its zero-copy view for transparent ID reuse on
  downstream re-encode.
- Long ZeroMQ/Vineyard chains therefore keep the huge payload in one Vineyard
  object while forwarding only the small NPB control message.

## 0.3.0

- Added optional `pyzmq` integration with `ZmqPush` and `ZmqPull`.
- Added `send_model()`, `recv_model()`, and `recv_auto()` convenience APIs.
- Added `zeromq` and `zeromq-vineyard` optional dependency groups.
- Added `examples/zeromq_vineyard_chain/`, where multi-GiB arrays remain in
  Vineyard and only 64 KiB NPB control messages traverse ZeroMQ.
- Added a per-hop object-ID verification to catch accidental huge republish
  copies.
- Vineyard extras pin `setuptools<81` for Vineyard 0.24.x compatibility.

## 0.2.0

- Added `BlobRef` and `BlobStore`.
- Added optional `VineyardStore` backend.
- Added `encode(..., blob_store=..., externalize_min_bytes=...)`.
- Added `decode(..., blob_store=...)` and `decode_auto(..., blob_store=...)`.
- Generic decode without a store returns unresolved `BlobRef` values.
- Transparent Vineyard object reuse avoids huge re-copy operations in long
  pub/sub chains.
- Added `examples/vineyard_chain/` showing Vineyard as the data plane and
  iceoryx2 as the control plane.

## 0.1.0

- Initial Pydantic + NumPy self-contained NPB format.
