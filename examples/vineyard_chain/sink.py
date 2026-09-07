from __future__ import annotations

import argparse
import ctypes
import time

import iceoryx2 as iox2
import numpy as np

from npb import decode

from common import ChainMessage, open_store, slice_as_numpy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--after-stage", type=int, required=True)
    parser.add_argument("--vineyard-socket", default=None)
    args = parser.parse_args()

    service_name = f"npb/vineyard-chain/{args.after_stage}"

    store = open_store(args.vineyard_socket)

    node = iox2.NodeBuilder.new().create(iox2.ServiceType.Ipc)
    service = (
        node.service_builder(iox2.ServiceName.new(service_name))
        .publish_subscribe(iox2.Slice[ctypes.c_uint8])
        .open_or_create()
    )
    subscriber = service.subscriber_builder().create()

    print(f"sink waiting on {service_name}")

    while True:
        sample = subscriber.receive()

        if sample is None:
            node.wait(iox2.Duration.from_millis(10))
            continue

        control = slice_as_numpy(sample.payload())

        t0 = time.perf_counter()
        message = decode(ChainMessage, control, blob_store=store)
        decode_ms = (time.perf_counter() - t0) * 1000

        print(
            f"received stage={message.stage}, "
            f"image={message.image.nbytes / 1024**3:.3f} GiB, "
            f"control={control.nbytes / 1024:.1f} KiB, "
            f"decode={decode_ms:.3f} ms, "
            f"first={message.image[0]}"
        )
        return


if __name__ == "__main__":
    main()
