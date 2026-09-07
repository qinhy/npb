"""Optional ZeroMQ transport for NPB control messages."""

from __future__ import annotations

from types import TracebackType
from typing import Any, TypeVar

import numpy as np
from pydantic import BaseModel

from ._blob import BlobStore
from ._codec import decode, decode_auto, encode
from ._format import require_binary_array

T = TypeVar("T", bound=BaseModel)


def _load_zmq():
    try:
        import zmq
    except ImportError as exc:  # pragma: no cover - exercised without optional extra
        raise ImportError(
            "ZeroMQ support is optional. Install it with "
            "`uv sync --extra zeromq` or `pip install 'npb[zeromq]'`."
        ) from exc
    return zmq


class _ZmqSocket:
    """Base class for small NPB control messages over ZeroMQ.

    NPB intentionally defaults to normal ZeroMQ copies here. Once large arrays
    are externalized to a BlobStore such as Vineyard, control messages are small
    enough that a copy is generally simpler and cheaper than managing transport
    buffer lifetimes manually.
    """

    _socket_type_name: str

    def __init__(
        self,
        endpoint: str,
        *,
        bind: bool,
        context: Any | None = None,
        hwm: int | None = 4,
        linger_ms: int = 0,
        send_timeout_ms: int | None = None,
        recv_timeout_ms: int | None = None,
    ) -> None:
        if not endpoint:
            raise ValueError("endpoint cannot be empty")
        if hwm is not None and hwm <= 0:
            raise ValueError("hwm must be > 0 or None")
        if linger_ms < 0:
            raise ValueError("linger_ms must be >= 0")

        zmq = _load_zmq()
        self._zmq = zmq
        self.endpoint = endpoint
        self._context = context if context is not None else zmq.Context.instance()
        self._socket = self._context.socket(getattr(zmq, self._socket_type_name))
        self._closed = False

        self._socket.setsockopt(zmq.LINGER, int(linger_ms))

        if hwm is not None:
            self._socket.setsockopt(zmq.SNDHWM, int(hwm))
            self._socket.setsockopt(zmq.RCVHWM, int(hwm))

        if send_timeout_ms is not None:
            self._socket.setsockopt(zmq.SNDTIMEO, int(send_timeout_ms))

        if recv_timeout_ms is not None:
            self._socket.setsockopt(zmq.RCVTIMEO, int(recv_timeout_ms))

        if bind:
            self._socket.bind(endpoint)
        else:
            self._socket.connect(endpoint)

    @classmethod
    def bind(cls, endpoint: str, **kwargs: Any):
        """Create a socket and bind it to ``endpoint``."""
        return cls(endpoint, bind=True, **kwargs)

    @classmethod
    def connect(cls, endpoint: str, **kwargs: Any):
        """Create a socket and connect it to ``endpoint``."""
        return cls(endpoint, bind=False, **kwargs)

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        if self._closed:
            return
        self._socket.close()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def send(
        self,
        binary: np.ndarray,
        *,
        copy: bool = True,
        flags: int = 0,
    ) -> int:
        """Send one NPB binary container and return its byte count."""
        require_binary_array(binary)
        self._socket.send(binary, flags=flags, copy=copy)
        return int(binary.nbytes)

    def recv(self, *, flags: int = 0) -> np.ndarray:
        """Receive one NPB message as a read-only 1-D ``np.uint8`` array."""
        payload = self._socket.recv(flags=flags, copy=True)
        return np.frombuffer(payload, dtype=np.uint8)

    def send_model(
        self,
        model: BaseModel,
        *,
        blob_store: BlobStore | None = None,
        externalize_min_bytes: int | None = None,
        copy: bool = True,
        flags: int = 0,
    ) -> int:
        """Encode and send a Pydantic model in one call."""
        binary = encode(
            model,
            blob_store=blob_store,
            externalize_min_bytes=externalize_min_bytes,
        )
        return self.send(binary, copy=copy, flags=flags)

    def recv_model(
        self,
        model_type: type[T],
        *,
        blob_store: BlobStore | None = None,
        flags: int = 0,
    ) -> T:
        """Receive and typed-decode one NPB control message."""
        return decode(
            model_type,
            self.recv(flags=flags),
            blob_store=blob_store,
        )

    def recv_auto(
        self,
        *,
        blob_store: BlobStore | None = None,
        flags: int = 0,
    ) -> dict[str, Any]:
        """Receive and generic-decode one NPB control message."""
        return decode_auto(
            self.recv(flags=flags),
            blob_store=blob_store,
        )


class ZmqPush(_ZmqSocket):
    """ZeroMQ PUSH endpoint for pipeline-style NPB control messages."""

    _socket_type_name = "PUSH"


class ZmqPull(_ZmqSocket):
    """ZeroMQ PULL endpoint for pipeline-style NPB control messages."""

    _socket_type_name = "PULL"
