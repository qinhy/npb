from __future__ import annotations

import time

import numpy as np

from npb import BinaryModel, VineyardStore, binary_schema


EXTERNALIZE = 1 * 1024 * 1024
DEFAULT_VINEYARD_SOCKET = "/tmp/vineyard.sock"


class Tensor(BinaryModel):
    name: str
    values: np.ndarray


@binary_schema("org.example.npb.zeromq-vineyard-chain", version=1)
class ChainMessage(BinaryModel):
    frame_id: int
    stage: int
    meta: dict[str, object]
    image: np.ndarray
    tensors: list[Tensor]


def endpoint(index: int) -> str:
    return f"ipc:///tmp/npb-zmq-vineyard-{index}.sock"


def open_store(socket: str) -> VineyardStore:
    # Persist externalized blobs so the one-shot producer can exit while the
    # downstream stages continue resolving the same immutable object IDs.
    return VineyardStore.connect(socket, persist_on_put=True)


def make_message(gib: float) -> ChainMessage:
    size = int(gib * 1024**3)
    if size <= 0:
        raise ValueError("gib must produce a positive image size")

    image = np.empty(size, dtype=np.uint8)
    image.fill(7)
    image[:4096] = np.arange(4096, dtype=np.uint32).astype(np.uint8)

    return ChainMessage(
        frame_id=time.time_ns(),
        stage=0,
        meta={"purpose": "NPB + Vineyard + ZeroMQ chain demo"},
        image=image,
        tensors=[
            Tensor(
                name="imu",
                values=np.arange(300_000, dtype=np.float32).reshape(100_000, 3),
            )
        ],
    )
