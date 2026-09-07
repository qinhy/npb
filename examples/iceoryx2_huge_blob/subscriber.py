from __future__ import annotations

import argparse
import ctypes
import time

import iceoryx2 as iox2
import numpy as np

from npb import decode, decode_auto, peek

from common import HugeBlob, SERVICE_NAME, check_markers, slice_as_numpy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Receive and zero-copy decode one huge NPB iceoryx2 sample."
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="use decode_auto() and return a Python dict instead of HugeBlob",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    iox2.set_log_level_from_env_or(iox2.LogLevel.Info)

    node = iox2.NodeBuilder.new().create(iox2.ServiceType.Ipc)

    service = (
        node.service_builder(iox2.ServiceName.new(SERVICE_NAME))
        .publish_subscribe(iox2.Slice[ctypes.c_uint8])
        .max_publishers(1)
        .max_subscribers(1)
        .subscriber_max_buffer_size(1)
        .subscriber_max_borrowed_samples(1)
        .history_size(0)
        .open_or_create()
    )

    subscriber = service.subscriber_builder().create()

    print("subscriber ready; waiting for one huge sample...")

    try:
        while True:
            sample = subscriber.receive()

            if sample is None:
                node.wait(iox2.Duration.from_millis(100))
                continue

            # binary is a zero-copy NumPy view of the iceoryx2 shared-memory
            # sample. Do not let `sample` go out of scope while using decoded
            # ndarray views.
            binary = slice_as_numpy(sample.payload())

            info = peek(binary)
            print(
                f"received {info.total_bytes / (1024**3):.3f} GiB "
                f"({info.frame_count:,} NPB frames)"
            )

            t0 = time.perf_counter()

            if args.auto:
                obj = decode_auto(binary)
                blob = obj["blob"]
            else:
                obj = decode(HugeBlob, binary)
                blob = obj.blob

            decode_s = time.perf_counter() - t0

            check_markers(blob)

            print(f"decode        : {decode_s * 1000:.3f} ms")
            print(f"blob          : {blob.nbytes / (1024**3):.3f} GiB")
            print(f"blob dtype    : {blob.dtype}")
            print(f"zero-copy     : {np.shares_memory(binary, blob)}")
            print(f"schema        : {info.schema_id} v{info.schema_version}")
            print("markers       : OK")

            # IMPORTANT:
            # `blob` and every other ndarray returned by decode/decode_auto
            # points into `sample`'s iceoryx2 shared memory. Finish all work
            # with those arrays before this sample is released.
            return

    except iox2.NodeWaitFailure:
        print("exit")


if __name__ == "__main__":
    main()
