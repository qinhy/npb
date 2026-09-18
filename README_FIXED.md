# NPB fixed branch prototype

This branch keeps the existing JSON-manifest + ndarray/BlobRef codec unchanged and adds a compact fixed-layout path for small control/RPC messages.

## Design rules

- `FixedStruct`: lightweight Python object; no Pydantic in the binary hot path.
- Fixed nested structs are recursively **inlined**.
- `Bit`: multiple booleans may share one byte.
- `OptionalNested`: presence bit + fixed inline payload.
- `FixedArray`: fixed-count scalar/struct arrays are inline.
- Large NumPy/tensor/point-cloud payloads should continue to use the existing ndarray/`BlobRef` path rather than becoming fixed structs.
- `FixedStruct.pydantic_model()` and `.json_schema()` provide JSON/OpenAPI/debug projection from the same schema.

## Example

```python
class Network(FixedStruct):
    ip = IPv4()
    port = UInt16(5555)

class Camera(FixedStruct):
    camera_id = FixedStr(15)
    network = Nested(Network)
    need_yolo = Bit(0, False)
    need_pcd = Bit(1, False)
    backend = Enum8("cpu", "cuda", default="cpu")
```

Binary nesting is flattened. JSON/OpenAPI stays nested.

## Current scope

Implemented: primitive integers/floats, `Bool8`, `Bit`, `FixedStr`, `IPv4`, `Enum8`, `FixedPointU16`, `Nested`, `OptionalNested`, `FixedArray`, Pydantic/OpenAPI projection, NumPy transport helpers, and ZeroMQ `send_fixed` / `recv_fixed`.

Not implemented yet: variable strings, variable arrays (`offset + count`), schema/version envelope for fixed payloads, generated FastAPI route decorator, and `BlobRef` as a native fixed field.

## Fixed NumPy arrays

Small arrays whose dtype and shape are known at schema-definition time can be stored inline:

```python
class Calibration(FixedStruct):
    k = FixedNDArray(np.float32, (3, 3))
    dist = FixedNDArray(np.float32, (5,))
```

The wire size is exactly `prod(shape) * dtype.itemsize`; there is no per-message dtype or shape metadata. `unpack_from()` returns a NumPy view over the input buffer. JSON/OpenAPI projects the array as nested JSON arrays and includes `x-npb-dtype`, `x-npb-shape`, and `x-npb-size` schema extensions.

Use `FixedNDArray` for small fixed-shape values such as calibration matrices, transforms, colors, embeddings with fixed dimensions, etc. Keep large or variable-shape images/tensors/point clouds on the existing ndarray/`BlobRef` path.
