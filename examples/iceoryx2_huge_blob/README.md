# iceoryx2: one huge NPB shared-memory object

This example sends a **single multi-GiB Pydantic object** as one dynamic
iceoryx2 `Slice[c_uint8]` sample.

It is intentionally different from chopping the object into many IPC
messages:

```text
publisher process
─────────────────────────────────────────────────────────────

disk-backed 2 GiB np.memmap
            │
            │  NPB encode: one sequential copy
            ▼
┌───────────────────────────────────────────────────────────┐
│ iceoryx2 loaned shared-memory sample                      │
│                                                           │
│ NPB header + JSON metadata + raw ndarray data             │
│                                                           │
│ ~2+ GiB, one sample                                       │
└─────────────────────────────┬─────────────────────────────┘
                              │
                              │ sample.send()
                              │ zero-copy IPC
                              ▼

subscriber process
─────────────────────────────────────────────────────────────

┌───────────────────────────────────────────────────────────┐
│ same iceoryx2 shared-memory sample                        │
└─────────────────────────────┬─────────────────────────────┘
                              │
                              │ np.frombuffer(no copy)
                              ▼
                         NPB decode()
                              │
                              │ ndarray views, no blob copy
                              ▼
                         HugeBlob model
```

The crucial integration is:

```python
required = encoded_size(message)

sample = publisher.loan_slice_uninit(required)

shared_memory = np.frombuffer(
    sample.payload().as_memory_view(),
    dtype=np.uint8,
)

encode(message, out=shared_memory)

sample = sample.assume_init()
sample.send()
```

The subscriber does the reverse:

```python
sample = subscriber.receive()

binary = np.frombuffer(
    sample.payload().as_memory_view(),
    dtype=np.uint8,
)

message = decode(HugeBlob, binary)

assert np.shares_memory(binary, message.blob)
```

`Slice.as_memory_view()` is an iceoryx2 no-copy view of the contiguous sample,
so NPB can operate directly on iceoryx2 memory.

## Requirements

This example targets **iceoryx2 0.9.3**.

```bash
uv sync --extra iceoryx2
```

For a 2 GiB payload, ensure the machine has enough shared-memory capacity.
iceoryx2 preallocates publisher memory for worst-case service requirements, so
this demo deliberately sets:

- one publisher
- one subscriber
- subscriber buffer size = 1
- subscriber borrowed samples = 1
- publisher loaned samples = 1
- history = 0

On Linux, inspect memory/shared-memory availability before starting:

```bash
free -h
df -h /dev/shm
```

If running in Docker, the normal Docker shared-memory limit is far too small
for this demonstration; start the container with an appropriately large
`--shm-size`.

## Run

Start the subscriber first:

```bash
uv run --extra iceoryx2 python examples/iceoryx2_huge_blob/subscriber.py
```

Then publish a real 2 GiB blob:

```bash
uv run --extra iceoryx2 python examples/iceoryx2_huge_blob/publisher.py --gib 2
```

Or 4 GiB on a machine with sufficient resources:

```bash
uv run --extra iceoryx2 python examples/iceoryx2_huge_blob/publisher.py --gib 4
```

For generic decoding:

```bash
uv run --extra iceoryx2 python examples/iceoryx2_huge_blob/subscriber.py --auto
```

## Why the source is an `np.memmap`

The publisher creates the huge source blob as a sparse disk-backed NumPy
memmap. That avoids first allocating another 2–4 GiB Python/NumPy heap buffer.

NPB still performs the real sequential copy of the complete logical blob into
the loaned iceoryx2 shared-memory sample. The transmitted sample therefore
really is multi-GiB.

## Lifetime rule

This is important:

```python
sample = subscriber.receive()
binary = ...
message = decode(HugeBlob, binary)
```

`message.blob` and the other decoded arrays point into memory owned by
`sample`.

Use them while `sample` is alive. Do not retain those ndarray views after the
iceoryx2 sample is released.

## What should be fast?

There are two very different costs:

1. **NPB encoding** copies the source NumPy leaves once into the shared-memory
   sample. For a 2 GiB blob this is memory-bandwidth work and should take a
   measurable amount of time.
2. **iceoryx2 send + NPB decode** do not copy the multi-GiB payload. Sending
   publishes the shared-memory sample, while decoding builds lightweight
   ndarray views plus the Pydantic object tree.

That is the behavior this demo is intended to make visible.
