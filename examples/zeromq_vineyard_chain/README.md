# ZeroMQ + Vineyard long-chain demo

This is the recommended NPB architecture for a long local processing chain:

```text
                         Vineyard
                immutable shared-memory blobs
                ┌────────────────────────────┐
                │ image: 2 GiB               │
                │ tensor: ...                │
                └────▲────▲────▲────▲────────┘
                     │    │    │    │
                 mmap / zero-copy reads
                     │    │    │    │

ZeroMQ control plane:
producer ──► stage1 ──► stage2 ──► stage3 ──► sink
             64 KiB NPB control messages only
```

The producer pays the large copy once when an ordinary NumPy array is placed in
Vineyard. Every downstream stage decodes a shared-memory ndarray, reads it, and
republishes only the small NPB control container. `VineyardStore` detects that
those arrays are already in Vineyard and reuses the same Vineyard object ID.

## Install

```bash
uv sync --extra zeromq-vineyard
```

The extra includes the compatibility Setuptools pin required by Vineyard 0.24.x.

## Start Vineyard

```bash
uv run python -m vineyard --socket /tmp/vineyard.sock
```

Keep that terminal running.

## Start a three-stage pipeline

Start consumers from the end of the chain first.

Terminal 2:

```bash
uv run --extra zeromq-vineyard \
  python examples/zeromq_vineyard_chain/sink.py \
  --after-stage 3
```

Terminal 3:

```bash
uv run --extra zeromq-vineyard \
  python examples/zeromq_vineyard_chain/stage.py \
  --stage 3
```

Terminal 4:

```bash
uv run --extra zeromq-vineyard \
  python examples/zeromq_vineyard_chain/stage.py \
  --stage 2
```

Terminal 5:

```bash
uv run --extra zeromq-vineyard \
  python examples/zeromq_vineyard_chain/stage.py \
  --stage 1
```

Finally publish from Terminal 6:

```bash
uv run --extra zeromq-vineyard \
  python examples/zeromq_vineyard_chain/producer.py \
  --gib 2
```

The sockets use local ZeroMQ IPC endpoints:

```text
ipc:///tmp/npb-zmq-vineyard-0.sock
ipc:///tmp/npb-zmq-vineyard-1.sock
ipc:///tmp/npb-zmq-vineyard-2.sock
ipc:///tmp/npb-zmq-vineyard-3.sock
```

## What to verify

The producer should show one relatively expensive externalization:

```text
source image     : 2.000 GiB
NPB control      : 64.0 KiB
Vineyard object  : o...
first encode     : ... ms
ZeroMQ send      : ... us
```

Each stage should instead look roughly like:

```text
stage=1 image=2.000 GiB control=64.0 KiB ... re-encode=<small> ... same_object=True
stage=2 image=2.000 GiB control=64.0 KiB ... re-encode=<small> ... same_object=True
stage=3 image=2.000 GiB control=64.0 KiB ... re-encode=<small> ... same_object=True
```

`same_object=True` is the critical check: the image's Vineyard object ID did not
change while passing through the chain.

## NPB ZeroMQ API

For ordinary pipeline use, NPB exposes lightweight PUSH/PULL wrappers:

```python
from npb import ZmqPull, ZmqPush

receiver = ZmqPull.bind("ipc:///tmp/input.sock")
sender = ZmqPush.connect("ipc:///tmp/output.sock")

binary = receiver.recv()
sender.send(binary)
```

Or encode/decode models directly:

```python
message = receiver.recv_model(MyModel, blob_store=store)

sender.send_model(
    message,
    blob_store=store,
    externalize_min_bytes=16 << 20,
)
```

The same wrapper accepts TCP endpoints, so the control plane can later move
between machines without changing NPB itself:

```text
tcp://127.0.0.1:5555
tcp://0.0.0.0:5555
```

For multi-host use, remember that local Vineyard shared-memory blobs must also
be made accessible through an appropriate distributed data-plane strategy; only
the ZeroMQ control transport becomes remote automatically.
