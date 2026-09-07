from __future__ import annotations

import argparse
import time

from npb import ZmqPull, decode

from common import DEFAULT_VINEYARD_SOCKET, ChainMessage, endpoint, open_store


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--after-stage", type=int, required=True)
    parser.add_argument("--vineyard-socket", default=DEFAULT_VINEYARD_SOCKET)
    args = parser.parse_args()

    if args.after_stage < 0:
        raise SystemExit("--after-stage must be >= 0")

    store = open_store(args.vineyard_socket)
    input_endpoint = endpoint(args.after_stage)

    print(f"sink waiting on {input_endpoint}")

    with ZmqPull.bind(input_endpoint, hwm=4) as receiver:
        t0 = time.perf_counter()
        receiver._socket.poll()
        wait_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        control = receiver.recv()
        recv_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        message = decode(
            ChainMessage,
            control,
            blob_store=store,
        )
        decode_ms = (time.perf_counter() - t0) * 1000

    print(
        f"received stage={message.stage} "
        f"image={message.image.nbytes / 1024**3:.3f} GiB "
        f"control={control.nbytes / 1024:.1f} KiB "
        f"recv={recv_ms:.3f} ms "
        f"decode={decode_ms:.3f} ms "
        f"first={int(message.image[0])} "
        f"readonly={not message.image.flags.writeable}"
    )


if __name__ == "__main__":
    main()
