from __future__ import annotations

import argparse
import ctypes
from pathlib import Path
import time

import iceoryx2 as iox2
import numpy as np

from npb import encode, encoded_size

from common import DEFAULT_GIB, SERVICE_NAME, make_message, make_sparse_blob, slice_as_numpy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish one genuinely huge NPB object in one iceoryx2 shared-memory sample."
    )
    parser.add_argument(
        "--gib",
        type=float,
        default=DEFAULT_GIB,
        help="size of the main blob in GiB (default: %(default)s)",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="optional path for the sparse disk-backed source blob",
    )
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=5.0,
        help="keep publisher alive after send so the subscriber can attach",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.gib <= 0:
        raise SystemExit("--gib must be > 0")

    blob_bytes = int(args.gib * (1024**3))
    blob, source_path = make_sparse_blob(blob_bytes, args.source)
    message = make_message(blob)

    # Critical point: this computes the exact loan size WITHOUT allocating or
    # copying the multi-GB NPB output buffer.
    required = encoded_size(message)

    print(f"source blob : {blob.nbytes / (1024**3):.3f} GiB")
    print(f"NPB sample  : {required / (1024**3):.3f} GiB")
    print(f"source file : {source_path}")

    iox2.set_log_level_from_env_or(iox2.LogLevel.Info)

    node = iox2.NodeBuilder.new().create(iox2.ServiceType.Ipc)

    # For huge samples, keep the worst-case iceoryx2 resource counts at one.
    # iceoryx2 preallocates for worst-case service requirements.
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

    publisher = (
        service.publisher_builder()
        .initial_max_slice_len(required)
        .max_loaned_samples(1)
        .create()
    )

    print("loaning huge iceoryx2 sample...")
    t_loan = time.perf_counter()
    sample = publisher.loan_slice_uninit(required)
    loan_s = time.perf_counter() - t_loan

    # This NumPy array IS the loaned iceoryx2 shared memory. No intermediate
    # NPB output allocation exists.
    shared_memory = slice_as_numpy(sample.payload())

    print("encoding NPB directly into loaned shared memory...")
    t_encode = time.perf_counter()
    encoded = encode(message, out=shared_memory)
    encode_s = time.perf_counter() - t_encode

    assert encoded.nbytes == required
    assert np.shares_memory(encoded, shared_memory)

    # NPB has initialized every byte of its frame-aligned container.
    sample = sample.assume_init()

    t_send = time.perf_counter()
    sample.send()
    send_s = time.perf_counter() - t_send

    gib = blob.nbytes / (1024**3)
    print(f"loan          : {loan_s * 1000:.3f} ms")
    print(f"encode/copy   : {encode_s:.3f} s ({gib / encode_s:.3f} GiB/s)")
    print(f"iceoryx2 send : {send_s * 1000:.3f} ms")
    print("sent one huge shared-memory sample")

    if args.hold_seconds > 0:
        # Keeping the publisher alive is convenient for the one-shot demo.
        time.sleep(args.hold_seconds)


if __name__ == "__main__":
    main()
