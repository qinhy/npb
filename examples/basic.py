from __future__ import annotations

import numpy as np

from npb import BinaryModel, binary_schema, decode, decode_auto, encode, peek


class Image(BinaryModel):
    name: str
    pixels: np.ndarray


@binary_schema("com.example.capture", version=1)
class Capture(BinaryModel):
    capture_id: str
    images: list[Image]


capture = Capture(
    capture_id="capture-001",
    images=[
        Image(
            name="front",
            pixels=np.arange(1920, dtype=np.uint8).reshape(20, 32, 3),
        )
    ],
)

# Normal allocation.
binary = encode(capture)

# Optional preallocated-buffer API.
pool_buffer = np.empty(binary.nbytes + 1024 * 1024, dtype=np.uint8)
binary_from_pool = encode(capture, out=pool_buffer)

print(peek(binary_from_pool))

generic = decode_auto(binary_from_pool)
print(type(generic), generic.keys())

typed = decode(Capture, binary_from_pool)
print(type(typed), typed.capture_id)

print(
    "zero-copy:",
    np.shares_memory(binary_from_pool, typed.images[0].pixels),
)
