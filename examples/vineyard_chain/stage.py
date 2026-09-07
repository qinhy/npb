from __future__ import annotations

import argparse
import ctypes
import time

import iceoryx2 as iox2
import numpy as np

from npb import decode, encode

from common import EXTERNALIZE, ChainMessage, open_store, slice_as_numpy


def service_name(index: int) -> str:
    return f"npb/vineyard-chain/{index}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--vineyard-socket", default=None)
    args = parser.parse_args()

    if args.stage <= 0:
        raise SystemExit("--stage must be >= 1")

    store = open_store(args.vineyard_socket)

    node = iox2.NodeBuilder.new().create(iox2.ServiceType.Ipc)

    in_service = (
        node.service_builder(iox2.ServiceName.new(service_name(args.stage - 1)))
        .publish_subscribe(iox2.Slice[ctypes.c_uint8])
        .open_or_create()
    )

    out_service = (
        node.service_builder(iox2.ServiceName.new(service_name(args.stage)))
        .publish_subscribe(iox2.Slice[ctypes.c_uint8])
        .open_or_create()
    )

    subscriber = in_service.subscriber_builder().create()

    # Control message remains small, so a modest maximum is enough.
    publisher = (
        out_service.publisher_builder()
        .initial_max_slice_len(256 * 1024)
        .create()
    )

    print(
        f"stage {args.stage}: "
        f"{service_name(args.stage - 1)} -> {service_name(args.stage)}"
    )

    while True:
        sample = subscriber.receive()

        if sample is None:
            node.wait(iox2.Duration.from_millis(10))
            continue

        binary = slice_as_numpy(sample.payload())

        t0 = time.perf_counter()
        message = decode(
            ChainMessage,
            binary,
            blob_store=store,
        )
        decode_ms = (time.perf_counter() - t0) * 1000

        # The image maps the same Vineyard blob. Touch a tiny piece to simulate
        # a read-only stage without scanning/copying the multi-GiB payload.
        checksum = int(message.image[:4096].sum())

        message.stage = args.stage
        message.meta["last_checksum"] = checksum

        t0 = time.perf_counter()
        outgoing = encode(
            message,
            blob_store=store,
            externalize_min_bytes=EXTERNALIZE,
        )
        encode_ms = (time.perf_counter() - t0) * 1000

        # This encode should only rebuild the tiny control message. The
        # VineyardStore recognizes image/tensors as already residing in its
        # shared-memory blobs and reuses their IDs.
        out_sample = publisher.loan_slice_uninit(outgoing.nbytes)
        out = slice_as_numpy(out_sample.payload())
        out[:] = outgoing

        out_sample = out_sample.assume_init()

        t0 = time.perf_counter()
        out_sample.send()
        send_us = (time.perf_counter() - t0) * 1e6

        print(
            f"stage={args.stage} "
            f"image={message.image.nbytes / 1024**3:.3f} GiB "
            f"control={outgoing.nbytes / 1024:.1f} KiB "
            f"decode={decode_ms:.3f} ms "
            f"re-encode={encode_ms:.3f} ms "
            f"send={send_us:.2f} us "
            f"readonly={not message.image.flags.writeable}"
        )


if __name__ == "__main__":
    main()
