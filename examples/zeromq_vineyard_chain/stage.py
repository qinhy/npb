from __future__ import annotations

import argparse
import time

from npb import BlobRef, ZmqPull, ZmqPush, decode, decode_auto, encode

from common import DEFAULT_VINEYARD_SOCKET, EXTERNALIZE, ChainMessage, endpoint, open_store


def image_ref(control):
    value = decode_auto(control)["image"]
    if not isinstance(value, BlobRef):
        raise RuntimeError("expected image to be externalized as BlobRef")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--vineyard-socket", default=DEFAULT_VINEYARD_SOCKET)
    args = parser.parse_args()

    if args.stage <= 0:
        raise SystemExit("--stage must be >= 1")

    store = open_store(args.vineyard_socket)
    input_endpoint = endpoint(args.stage - 1)
    output_endpoint = endpoint(args.stage)

    print(f"stage {args.stage}: waiting on {input_endpoint}")

    with (
        ZmqPull.bind(input_endpoint, hwm=4, recv_timeout_ms=None) as receiver,
        ZmqPush.connect(
            output_endpoint,
            hwm=4,
            linger_ms=1000,
            send_timeout_ms=5000,
        ) as sender,
    ):
        t0 = time.perf_counter()
        receiver._socket.poll()
        wait_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        incoming = receiver.recv()
        recv_ms = (time.perf_counter() - t0) * 1000

        incoming_ref = image_ref(incoming)

        t0 = time.perf_counter()
        message = decode(ChainMessage, incoming, blob_store=store)
        decode_ms = (time.perf_counter() - t0) * 1000.0

        # Simulate a read-only algorithm touching only a small region. The full
        # multi-GiB ndarray remains backed by the immutable Vineyard blob.
        checksum = int(message.image[:4096].sum())
        message.stage = args.stage
        message.meta["last_checksum"] = checksum

        t0 = time.perf_counter()
        outgoing = encode(
            message,
            blob_store=store,
            externalize_min_bytes=EXTERNALIZE,
        )
        encode_ms = (time.perf_counter() - t0) * 1000.0

        outgoing_ref = image_ref(outgoing)
        same_object = incoming_ref.object_id == outgoing_ref.object_id

        if not same_object:
            raise RuntimeError("Vineyard ndarray object was copied/replaced during republish")

        t0 = time.perf_counter()
        sender.send(outgoing)
        send_us = (time.perf_counter() - t0) * 1e6

    print(
        f"stage={args.stage} "
        f"image={message.image.nbytes / 1024**3:.3f} GiB "
        f"control={outgoing.nbytes / 1024:.1f} KiB "
        f"recv={recv_ms:.3f} ms "
        f"decode={decode_ms:.3f} ms "
        f"re-encode={encode_ms:.3f} ms "
        f"send={send_us:.2f} us "
        f"same_object={same_object} "
        f"readonly={not message.image.flags.writeable}"
    )


if __name__ == "__main__":
    main()
