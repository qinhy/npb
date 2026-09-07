# Changelog

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
