from __future__ import annotations

import numpy as np

from npb import FixedArray, FixedNDArray, FixedStr, NpbStruct


class Image(NpbStruct):
    # Fixed-width UTF-8 name on the wire.
    name = FixedStr(15)

    # dtype and shape belong to the schema, so there is no ndarray metadata
    # in each message. Wire cost = 20 * 32 * 3 = 1920 bytes exactly.
    pixels:np.ndarray = FixedNDArray(np.uint8, (20, 32, 3))


class Capture(NpbStruct):
    capture_id = FixedStr(15)

    # This example has exactly one image, so it can be fully fixed-layout.
    # If the number of images is dynamic, use VarArray later instead.
    images:list[Image] = FixedArray(Image, 1)


capture = Capture(
    capture_id="capture-001",
    images=[
        Image(
            name="front",
            pixels=np.arange(1920, dtype=np.uint8).reshape(20, 32, 3),
        )
    ],
)

print("Capture byte size:", Capture.byte_size())
print("Image byte size:", Image.byte_size())
print("layout:")
for row in Capture.layout():
    print(row)

# ------------------------------------------------------------
# Normal allocation
# ------------------------------------------------------------

binary = capture.to_bytearray()

print("binary type:", type(binary))
print("binary bytes:", len(binary))


# ------------------------------------------------------------
# Optional preallocated-buffer API
# ------------------------------------------------------------

pool_buffer = np.empty(
    Capture.byte_size() + 1024 * 1024,
    dtype=np.uint8,
)

capture.pack_into(pool_buffer)

# Only this prefix contains the fixed NPB object.
binary_from_pool = pool_buffer[: Capture.byte_size()]


# ------------------------------------------------------------
# Typed decode
# ------------------------------------------------------------

typed:Capture = Capture.unpack_from(binary_from_pool)

print(type(typed), typed.capture_id)
print(type(typed.images[0]), typed.images[0].name)
print(
    "pixels:",
    typed.images[0].pixels.shape,
    typed.images[0].pixels.dtype,
)


# ------------------------------------------------------------
# Zero-copy decode
# ------------------------------------------------------------

print(
    "zero-copy:",
    np.shares_memory(
        binary_from_pool,
        typed.images[0].pixels,
    ),
)


# ------------------------------------------------------------
# JSON / OpenAPI debug projection
# ------------------------------------------------------------

CaptureJson = Capture.pydantic_model()

json_model = CaptureJson.model_validate(capture.to_dict())
print(json_model.model_dump())

print(Capture.json_schema())