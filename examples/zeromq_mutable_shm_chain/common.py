from __future__ import annotations

import os
import time

from npb import BinaryModel, binary_schema
from mstore import default_endpoint


DEFAULT_MSTORE_ENDPOINT = default_endpoint()
ZMQ_BASE_PORT = 55600


def endpoint(index: int) -> str:
    """Local ZeroMQ control endpoint.

    Windows libzmq builds commonly do not support ipc://, so use loopback TCP there.
    Only the tiny NPB control message crosses ZeroMQ; the ndarray stays in mstore.
    """
    if os.name == "nt":
        return f"tcp://127.0.0.1:{ZMQ_BASE_PORT + index}"
    return f"ipc:///tmp/npb-zmq-mstore-{index}.sock"


class ShmArrayRef(BinaryModel):
    """Small control-plane reference to an ndarray owned by mutable-shm-store."""

    object_id: str
    token: str
    generation: int


@binary_schema("org.example.npb.zeromq-mutable-shm-chain", version=2)
class ChainMessage(BinaryModel):
    frame_id: int
    sequence: int
    stage: int
    meta: dict[str, object]
    image: ShmArrayRef


def new_frame_id() -> int:
    return time.time_ns()
