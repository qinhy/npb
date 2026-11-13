# npb

`npb` is a small binary codec for Pydantic models that contain NumPy arrays.

It keeps ordinary metadata in a compact JSON manifest and stores NumPy array
payloads as raw bytes in the same contiguous `np.uint8` container.

```text
Pydantic object
      |
      v
+-------------------------------+
| fixed NPB header              |
+-------------------------------+
| JSON manifest                 |
| ndarray leaves -> references  |
+-------------------------------+
| frame padding                 |
+-------------------------------+
| raw ndarray bytes             |
| raw ndarray bytes             |
| ...                           |
+-------------------------------+
| final frame padding           |
+-------------------------------+
```

The public API is deliberately small:

```python
binary = encode(obj)
info = peek(binary)
generic = decode_auto(binary)
typed = decode(MyModel, binary)
```

Decoded NumPy arrays are zero-copy views into the encoded binary array.

## Install

Once the package is published to your package index:

```bash
uv add npb
```

For local development:

```bash
git clone <this-repository>
cd npb
uv sync --dev
uv run pytest
```

## Benchmarks

The repository includes a dependency-free benchmark harness for allocated and
preallocated encode, generic and typed decode, encoded-size calculation, and
header inspection:

```bash
uv run python benchmarks/benchmark_codec.py
```

Use `--quick` for a smoke test, or configure payload sizes and array count:

```bash
uv run python benchmarks/benchmark_codec.py --sizes 64KiB 4MiB 64MiB --arrays 4
```

See [`benchmarks/README.md`](benchmarks/README.md) for methodology and JSON
output suitable for tracking results over time.

## Define a model

Use `BinaryModel` so NumPy arrays are accepted naturally, then give the
top-level serialized model a stable schema identity.

```python
import numpy as np
from npb import BinaryModel, binary_schema


class Image(BinaryModel):
    name: str
    pixels: np.ndarray


@binary_schema("com.example.capture-bundle", version=1)
class CaptureBundle(BinaryModel):
    capture_id: str
    meta: dict[str, object]
    images: list[Image]
```

`binary_schema()` converts the stable human-readable name into a UUID stored
in the fixed binary header. Keep the name and version stable for persisted or
networked data.

The identity belongs to the decorated class only. A subclass does not inherit
it, because it may add or change fields and would otherwise be decoded as its
parent without complaint; give every class you encode its own
`@binary_schema(...)`. Parametrized generics such as `Box[int]` use the
identity of `Box`.

## Encode

```python
from npb import encode

obj = CaptureBundle(
    capture_id="capture-001",
    meta={"device": "cam-a", "quality": 0.95},
    images=[
        Image(
            name="front",
            pixels=np.zeros((720, 1280, 3), dtype=np.uint8),
        )
    ],
)

binary = encode(obj)

assert isinstance(binary, np.ndarray)
assert binary.dtype == np.uint8
assert binary.ndim == 1
```

The result is self-contained and frame aligned.

## Encode into a preallocated buffer

Use the NumPy-style `out=` API when you want to reuse memory:

```python
container = np.empty(16 * 1024 * 1024, dtype=np.uint8)

binary = encode(obj, out=container)

assert np.shares_memory(binary, container)
```

`binary` is a view covering exactly the encoded region. The original
`container` may be larger. NPB also records the encoded length in its fixed
header, so passing the full oversized backing buffer to `peek()` or a decoder
is supported.

This is useful for buffer pools, shared memory, `np.memmap`, and high-throughput
services.

## Typed decode

```python
from npb import decode

restored = decode(CaptureBundle, binary)

assert isinstance(restored, CaptureBundle)
assert np.shares_memory(binary, restored.images[0].pixels)
```

Typed decode verifies the embedded schema UUID and schema version before
Pydantic validation.

Metadata travels as JSON, so values such as `datetime` and `tuple` reach
validation as strings and lists. Pydantic's default lax mode converts them
back. A model configured with `strict=True` rejects them unless you relax
validation for the decode:

```python
restored = decode(StrictModel, binary, strict=False)
```

`peek()`, `decode()` and `decode_auto()` accept a 1-D `np.uint8` array or any
`bytes`, `bytearray` or `memoryview`. Bytes-like inputs are wrapped without
copying, and the decoded arrays keep the wrapped object alive.

## Generic decode

When the Pydantic class is unavailable or unnecessary:

```python
from npb import decode_auto

data = decode_auto(binary)

assert isinstance(data, dict)
assert isinstance(data["images"][0]["pixels"], np.ndarray)
assert np.shares_memory(binary, data["images"][0]["pixels"])
```

`decode_auto()` does **not** dynamically import or execute Python classes from
the binary. It returns a plain tree of dictionaries, lists, JSON scalars, and
NumPy arrays.

Values that require the Pydantic schema for reconstruction, such as
`datetime`, remain in their JSON representation during `decode_auto()` and are
restored by typed `decode()`.

## Inspect without decoding

```python
from npb import peek

info = peek(binary)

print(info.schema_id)
print(info.schema_version)
print(info.frame_count)
print(info.data_bytes)
print(info.flags)
```

`peek()` reads only the fixed header. It does not parse JSON or touch NumPy
payloads.

## Fixed-size frames

The complete encoded array is padded to a whole number of frames. The default
frame is `FRAME_SIZE`, 64 KiB:

```python
from npb import FRAME_SIZE

frames = binary.reshape(-1, FRAME_SIZE)

for frame in frames:
    send(frame)
```

Every frame has exactly the same physical size.

The frame size is recorded in the header and can be chosen per message. It
must be a power of two between `MIN_FRAME_SIZE` (64 bytes) and 1 GiB:

```python
from npb import MIN_FRAME_SIZE

compact = encode(obj, frame_size=MIN_FRAME_SIZE)
size = encoded_size(obj, frame_size=MIN_FRAME_SIZE)
```

A small frame matters most for control messages whose arrays live in an
external blob store: a few hundred bytes of header and manifest then occupy a
few hundred bytes instead of a full 64 KiB frame. Decoders read the frame size
from the header, so no decode-side argument is needed. Only the default frame
size can be read by NPB 0.3 and earlier.

## Supported NumPy arrays

NPB currently supports ordinary, non-object, non-structured NumPy dtypes.

Good examples:

```python
np.uint8
np.int16
np.int64
np.float32
np.float64
np.complex64
np.bool_
```

Object arrays and structured dtypes are intentionally rejected because their
raw memory representation is not appropriate for this portable binary format.

Non-C-contiguous arrays are normalized to C-contiguous storage while encoding.
Strided and Fortran-ordered inputs are copied straight into the container, with
no intermediate contiguous copy.

An array object that is referenced from several fields is stored once; after
decoding, those fields are views of the same memory.

## Bytes fields and reserved keys

`bytes` values that are valid UTF-8 are kept in the JSON manifest, as in
earlier versions. Any other `bytes` value (an encoded image, a protobuf) is
stored raw in the DATA section and comes back as `bytes`; this is a copy, not a
view.

NPB marks its own references in the manifest with the single-key objects
`{"$bin": ...}` and `{"$blob": ...}`. Your own dictionaries may use those keys
too: a user dictionary that would be mistaken for a reference is escaped on
encode and restored on decode, and is never resolved against a blob store.

## Buffer lifetime

Decoded arrays point into the encoded `np.uint8` container:

```python
restored = decode(CaptureBundle, binary)
```

Keep `binary` alive while using `restored` arrays.

If the supplied binary is writable, those decoded views are writable too.
For untrusted data, prefer a read-only backing buffer or avoid mutating the
decoded arrays.

## Format stability

The container has two independent versions:

- **format version** — the NPB physical container layout.
- **schema version** — your application's Pydantic model contract.

Changing application fields should normally change the schema version, not the
NPB format version.

NPB is currently alpha (`0.x`), so the binary format may evolve before `1.0`.

### Compatibility with NPB 0.3

NPB 0.4 reads every container written by 0.3. In the other direction, a 0.4
encoder with default arguments writes exactly the bytes 0.3 wrote for anything
0.3 could encode, so mixed-version pipelines keep working. Three opt-in or
previously impossible cases produce containers that 0.3 rejects with a clear
`FormatError` rather than misreading:

| What you encode | Header | NPB 0.3 decoder |
| --- | --- | --- |
| `frame_size=` other than 64 KiB | frame size field | `unsupported frame size` |
| non-UTF-8 `bytes` (failed to encode in 0.3) | flag bit 0 | `unsupported NPB flags` |
| a user dict keyed only by `$bin`, `$blob` or `$$...` | flag bit 0 | `unsupported NPB flags` |


## External immutable blobs with Vineyard

For long pub/sub chains, copying the same multi-GiB ndarray at every hop will
eventually saturate memory bandwidth. NPB 0.2 can move large ndarray leaves to
an external immutable blob store while keeping the Pydantic structure in a
small NPB control message.

```python
from npb import VineyardStore, encode, decode

store = VineyardStore.connect("/tmp/vineyard.sock")

control = encode(
    obj,
    blob_store=store,
    externalize_min_bytes=16 * 1024 * 1024,
)

restored = decode(
    MyModel,
    control,
    blob_store=store,
)
```

Large arrays are stored as Vineyard blobs; the NPB manifest contains stable
`$blob` references with object ID, byte range, dtype, and shape.

The important long-chain optimization is transparent reuse. When a decoded
ndarray already points into Vineyard shared memory, the next
`encode(..., blob_store=store, ...)` detects the existing backing blob and
reuses its object ID instead of copying the payload again.

Without a store, generic decoding remains useful:

```python
generic = decode_auto(control)

assert isinstance(generic["image"], BlobRef)
```

Typed decoding of external references requires the matching store:

```python
obj = decode(MyModel, control, blob_store=store)
```

`encoded_size(obj, blob_store=store, externalize_min_bytes=...)` never writes
to the store, so it is safe to size a loan first and encode afterwards.

`encode()` checks everything it can before the first array is stored:
unsupported dtypes, metadata that is not JSON serializable, and the type and
capacity of `out`. If the store itself then fails part-way, the objects that
call created are deleted again; objects that were reused are left alone.

`VineyardStore` ties each reuse registration to the lifetime of the resolved
shared-memory view. The registration disappears when the last view of the
object is garbage-collected, or when you call `store.delete(ref)` or
`store.release(ref)`.

### Writing your own store

A store implements the `BlobStore` protocol. It may also provide one optional
method:

```python
def preview_ref(self, array: np.ndarray) -> BlobRef:
    """Return what put_array(array) would return, without storing anything.

    The real reference if the array is already backed by the store, otherwise
    a placeholder whose object_id has the same length as a real one.
    """
```

With `preview_ref`, `encoded_size()` is exact and failed encodes are rolled
back. Without it, `encoded_size()` assumes 32-character object IDs and nothing
is rolled back, because NPB cannot tell a new object from a reused one.

See `examples/vineyard_chain/` for an iceoryx2 pipeline where a 2 GiB image is
stored once in Vineyard while only ~64 KiB NPB control messages traverse each
pub/sub hop.


## ZeroMQ control transport

NPB 0.3 includes an optional ZeroMQ PUSH/PULL adapter for transport-agnostic
control messages. This is especially useful with an external blob store: large
arrays stay in Vineyard while only a small NPB container crosses each ZeroMQ
hop.

Install the transport alone:

```bash
uv sync --extra zeromq
```

Or install the recommended ZeroMQ + Vineyard pipeline stack:

```bash
uv sync --extra zeromq-vineyard
```

Basic binary transport:

```python
from npb import ZmqPull, ZmqPush

receiver = ZmqPull.bind("ipc:///tmp/input.sock")
sender = ZmqPush.connect("ipc:///tmp/output.sock")

binary = receiver.recv()
sender.send(binary)
```

Direct typed model transport is also available:

```python
message = receiver.recv_model(MyModel, blob_store=store)

sender.send_model(
    message,
    blob_store=store,
    externalize_min_bytes=16 * 1024 * 1024,
)
```

The wrappers also accept `tcp://` endpoints. NPB deliberately uses ordinary
ZeroMQ copies by default because, after externalization, the control container
is small and its buffer-lifetime management is simpler than forcing transport
zero-copy.

When every receiver runs NPB 0.4 or later, send reference-only control
messages with a small frame so that each hop moves a few hundred bytes rather
than 64 KiB:

```python
from npb import MIN_FRAME_SIZE

sender.send_model(
    message,
    blob_store=store,
    externalize_min_bytes=16 * 1024 * 1024,
    frame_size=MIN_FRAME_SIZE,
)
```

See `examples/zeromq_vineyard_chain/` for a three-stage pipeline that verifies
`same_object=True` at every republish.

## iceoryx2 huge-blob example

The repository includes a real multi-GiB zero-copy IPC integration under
`examples/iceoryx2_huge_blob/`.

It loans one dynamic iceoryx2 byte slice, encodes NPB directly into that
shared-memory sample with `encode(..., out=...)`, then reconstructs the
Pydantic object on the subscriber while the huge NumPy leaves remain views of
the received shared memory.

Use `encoded_size(model)` to determine the exact iceoryx2 loan size without
allocating the NPB output first.

See `examples/iceoryx2_huge_blob/README.md`.

## Development

```bash
uv sync --dev
uv run pytest
uv run ruff check .
uv build
```
