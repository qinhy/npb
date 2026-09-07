# NPB + Vineyard + iceoryx2 long-chain demo

This example fixes the throughput collapse caused by repeatedly copying a huge
payload through a long pub/sub chain.

The data plane and control plane are separated:

```text
                         Vineyard
                immutable shared-memory blobs
                ┌────────────────────────────┐
                │ image: 2 GiB               │
                │ tensor: ...                │
                └────▲────▲────▲────▲────────┘
                     │    │    │    │
                     │ mmap / zero-copy
                     │    │    │    │

iceoryx2 control:
producer ──► stage1 ──► stage2 ──► stage3 ──► sink
             ~64 KiB NPB messages only
```

The first producer copies a large ndarray into Vineyard once:

```python
control = encode(
    message,
    blob_store=store,
    externalize_min_bytes=1 * 1024 * 1024,
)
```

The NPB manifest contains an external reference similar to:

```json
{
  "$blob": {
    "store": "vineyard",
    "id": "o800...",
    "offset": 0,
    "nbytes": 2147483648,
    "dtype": "|u1",
    "shape": [2147483648]
  }
}
```

Each stage resolves that reference as a zero-copy NumPy view:

```python
message = decode(
    ChainMessage,
    control,
    blob_store=store,
)
```

When the stage republishes:

```python
outgoing = encode(
    message,
    blob_store=store,
    externalize_min_bytes=1 * 1024 * 1024,
)
```

`VineyardStore` checks whether the ndarray pointer is already in Vineyard
shared memory. If it is, NPB reuses the existing Vineyard ndarray/Tensor object ID instead of
copying the multi-GiB array again.

## Install

```bash
uv sync --extra pipeline
```

Start Vineyard in another terminal:

```bash
python -m vineyard --socket /tmp/vineyard.sock
```

Then start a 3-stage chain, sink first:

```bash
uv run --extra pipeline \
  python examples/vineyard_chain/sink.py \
  --after-stage 3 \
  --vineyard-socket /tmp/vineyard.sock
```

```bash
uv run --extra pipeline \
  python examples/vineyard_chain/stage.py \
  --stage 3 \
  --vineyard-socket /tmp/vineyard.sock
```

```bash
uv run --extra pipeline \
  python examples/vineyard_chain/stage.py \
  --stage 2 \
  --vineyard-socket /tmp/vineyard.sock
```

```bash
uv run --extra pipeline \
  python examples/vineyard_chain/stage.py \
  --stage 1 \
  --vineyard-socket /tmp/vineyard.sock
```

Finally publish a 2 GiB object:

```bash
uv run --extra pipeline \
  python examples/vineyard_chain/producer.py \
  --gib 2 \
  --vineyard-socket /tmp/vineyard.sock
```

Expected behavior after the initial externalization:

```text
stage=1 image=2.000 GiB control=64.0 KiB decode=<small> re-encode=<small> send=<us>
stage=2 image=2.000 GiB control=64.0 KiB decode=<small> re-encode=<small> send=<us>
stage=3 image=2.000 GiB control=64.0 KiB decode=<small> re-encode=<small> send=<us>
```

The huge array remains immutable in Vineyard; the long chain only republishes
small NPB control messages.
