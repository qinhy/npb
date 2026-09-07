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
git clone <your-repository>
cd npb
uv sync --dev
uv run pytest
```

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
```

`peek()` reads only the fixed header. It does not parse JSON or touch NumPy
payloads.

## Fixed-size frames

The complete encoded array is padded to `FRAME_SIZE`, currently 64 KiB:

```python
from npb import FRAME_SIZE

frames = binary.reshape(-1, FRAME_SIZE)

for frame in frames:
    send(frame)
```

Every frame has exactly the same physical size.

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

## Development

```bash
uv sync --dev
uv run pytest
uv run ruff check .
uv build
```
