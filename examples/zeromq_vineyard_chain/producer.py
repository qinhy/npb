from __future__ import annotations

import argparse
import time

from npb import ZmqPush, decode_auto, encode

from common import DEFAULT_VINEYARD_SOCKET, EXTERNALIZE, endpoint, make_message, open_store


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gib", type=float, default=2.0)
    parser.add_argument("--vineyard-socket", default=DEFAULT_VINEYARD_SOCKET)
    args = parser.parse_args()

    store = open_store(args.vineyard_socket)
    message = make_message(args.gib)

    print("externalizing huge ndarray into Vineyard once...")
    t0 = time.perf_counter()
    control = encode(
        message,
        blob_store=store,
        externalize_min_bytes=EXTERNALIZE,
    )
    encode_ms = (time.perf_counter() - t0) * 1000.0

    generic = decode_auto(control)
    image_ref = generic["image"]

    print(f"source image     : {message.image.nbytes / 1024**3:.3f} GiB")
    print(f"NPB control      : {control.nbytes / 1024:.1f} KiB")
    print(f"Vineyard object  : {image_ref.object_id}")
    print(f"first encode     : {encode_ms:.3f} ms")

    with ZmqPush.connect(
        endpoint(0),
        hwm=4,
        linger_ms=1000,
        send_timeout_ms=5000,
    ) as sender:
        t0 = time.perf_counter()
        sender.send(control)
        send_us = (time.perf_counter() - t0) * 1e6

    print(f"ZeroMQ send      : {send_us:.2f} us")
    print("sent one tiny NPB control message")


if __name__ == "__main__":
    main()
