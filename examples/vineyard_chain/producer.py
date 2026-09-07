from __future__ import annotations

import argparse
import ctypes
import time

import iceoryx2 as iox2

from npb import encode

from common import EXTERNALIZE, ChainMessage, make_message, open_store, slice_as_numpy


SERVICE = "npb/vineyard-chain/0"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gib", type=float, default=2.0)
    parser.add_argument("--vineyard-socket", default=None)
    args = parser.parse_args()

    store = open_store(args.vineyard_socket)
    message = make_message(args.gib)

    print("first encode: huge ndarray -> Vineyard blob (one-time copy)")
    t0 = time.perf_counter()
    control = encode(
        message,
        blob_store=store,
        externalize_min_bytes=EXTERNALIZE,
    )
    dt = time.perf_counter() - t0

    print(f"source image : {message.image.nbytes / 1024**3:.3f} GiB")
    print(f"control NPB  : {control.nbytes / 1024:.1f} KiB")
    print(f"externalize  : {dt:.3f} s")

    node = iox2.NodeBuilder.new().create(iox2.ServiceType.Ipc)
    service = (
        node.service_builder(iox2.ServiceName.new(SERVICE))
        .publish_subscribe(iox2.Slice[ctypes.c_uint8])
        .open_or_create()
    )
    publisher = (
        service.publisher_builder()
        .initial_max_slice_len(control.nbytes)
        .create()
    )

    sample = publisher.loan_slice_uninit(control.nbytes)
    out = slice_as_numpy(sample.payload())
    out[:] = control

    sample = sample.assume_init()

    t0 = time.perf_counter()
    sample.send()
    print(f"iceoryx2 send: {(time.perf_counter() - t0) * 1e6:.2f} us")


if __name__ == "__main__":
    main()
