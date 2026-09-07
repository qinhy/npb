from __future__ import annotations

import uuid

import numpy as np
import pytest

pytest.importorskip("zmq")

from npb import BinaryModel, ZmqPull, ZmqPush, binary_schema  # noqa: E402


@binary_schema("tests.zmq-message", version=1)
class Message(BinaryModel):
    name: str
    values: np.ndarray


def unique_endpoint() -> str:
    return f"inproc://npb-{uuid.uuid4().hex}"


def test_zmq_push_pull_binary_roundtrip() -> None:
    endpoint = unique_endpoint()
    source = np.arange(1024, dtype=np.uint8)

    with (
        ZmqPull.bind(endpoint, recv_timeout_ms=1000) as receiver,
        ZmqPush.connect(endpoint, send_timeout_ms=1000) as sender,
    ):
        sent = sender.send(source)
        restored = receiver.recv()

    assert sent == source.nbytes
    assert restored.dtype == np.uint8
    assert restored.ndim == 1
    assert np.array_equal(restored, source)
    assert not restored.flags.writeable


def test_zmq_send_model_recv_model() -> None:
    endpoint = unique_endpoint()
    source = Message(name="hello", values=np.arange(128, dtype=np.float32))

    with (
        ZmqPull.bind(endpoint, recv_timeout_ms=1000) as receiver,
        ZmqPush.connect(endpoint, send_timeout_ms=1000) as sender,
    ):
        sent = sender.send_model(source)
        restored = receiver.recv_model(Message)

    assert sent >= 64 * 1024
    assert restored.name == source.name
    assert np.array_equal(restored.values, source.values)


def test_zmq_recv_auto() -> None:
    endpoint = unique_endpoint()
    source = Message(name="auto", values=np.arange(16, dtype=np.int32))

    with (
        ZmqPull.bind(endpoint, recv_timeout_ms=1000) as receiver,
        ZmqPush.connect(endpoint, send_timeout_ms=1000) as sender,
    ):
        sender.send_model(source)
        restored = receiver.recv_auto()

    assert restored["name"] == "auto"
    assert np.array_equal(restored["values"], source.values)
