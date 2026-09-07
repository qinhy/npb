from __future__ import annotations

import ctypes
import time

import numpy as np

from npb import BinaryModel, VineyardStore, binary_schema


EXTERNALIZE = 1 * 1024 * 1024


class Tensor(BinaryModel):
    name: str
    values: np.ndarray


@binary_schema("org.example.npb.vineyard-chain", version=1)
class ChainMessage(BinaryModel):
    frame_id: int
    stage: int
    meta: dict[str, object]
    image: np.ndarray
    tensors: list[Tensor]


def make_message(gib: float) -> ChainMessage:
    size = int(gib * 1024**3)

    image = np.empty(size, dtype=np.uint8)
    image[:] = 7
    image[:4096] = np.arange(4096, dtype=np.uint32).astype(np.uint8)

    return ChainMessage(
        frame_id=time.time_ns(),
        stage=0,
        meta={"purpose": "NPB + Vineyard long-chain zero-copy demo"},
        image=image,
        tensors=[
            Tensor(
                name="imu",
                values=np.arange(300_000, dtype=np.float32).reshape(100_000, 3),
            )
        ],
    )


def slice_as_numpy(payload) -> np.ndarray:
    ptr = ctypes.cast(
        payload.as_ptr(),
        ctypes.POINTER(ctypes.c_uint8),
    )
    return np.ctypeslib.as_array(ptr, shape=(payload.len(),))


def open_store(socket: str | None) -> VineyardStore:
    return VineyardStore.connect(socket, persist_on_put=True)
