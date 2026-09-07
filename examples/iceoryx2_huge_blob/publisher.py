from __future__ import annotations

import argparse
import ctypes
from pathlib import Path
import time

import iceoryx2 as iox2
import numpy as np

from npb import encode, encoded_size

from common import (
    DEFAULT_GIB,
    SERVICE_NAME,
    make_message,
    make_sparse_blob,
    print_profile_summary,
    slice_as_numpy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Profile repeated publication of one genuinely huge NPB object "
            "through iceoryx2 shared memory."
        )
    )
    parser.add_argument(
        "--gib",
        type=float,
        default=DEFAULT_GIB,
        help="size of the main blob in GiB (default: %(default)s)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="number of measured sends (default: %(default)s)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="number of unmeasured warm-up sends (default: %(default)s)",
    )
    parser.add_argument(
        "--pause-ms",
        type=float,
        default=50.0,
        help=(
            "pause after every send so the subscriber can release the previous "
            "huge sample; pause is not included in timings (default: %(default)s)"
        ),
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
        default=1.0,
        help="keep publisher alive after the final send (default: %(default)s)",
    )
    return parser.parse_args()


def set_iteration_metadata(message, *, phase: str, sequence: int) -> None:
    # Fixed-width values keep metadata size stable across iterations.
    message.meta["benchmark_phase"] = phase
    message.meta["benchmark_sequence"] = f"{sequence:08d}"


def main() -> None:
    args = parse_args()

    if args.gib <= 0:
        raise SystemExit("--gib must be > 0")
    if args.iterations <= 0:
        raise SystemExit("--iterations must be > 0")
    if args.warmup < 0:
        raise SystemExit("--warmup must be >= 0")
    if args.pause_ms < 0:
        raise SystemExit("--pause-ms must be >= 0")

    blob_bytes = int(args.gib * (1024**3))
    blob, source_path = make_sparse_blob(blob_bytes, args.source)
    message = make_message(blob)

    # "W" and "M" have equal length, so the encoded layout remains stable.
    set_iteration_metadata(message, phase="W", sequence=0)
    required = encoded_size(message)

    set_iteration_metadata(message, phase="M", sequence=0)
    measured_required = encoded_size(message)
    if measured_required != required:
        raise RuntimeError("benchmark metadata unexpectedly changed encoded size")

    print(f"source blob       : {blob.nbytes / (1024**3):.3f} GiB")
    print(f"NPB sample        : {required / (1024**3):.3f} GiB")
    print(f"source file       : {source_path}")
    print(f"warmup iterations : {args.warmup}")
    print(f"measured iterations: {args.iterations}")
    print(f"pause             : {args.pause_ms:.1f} ms")

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

    publisher = (
        service.publisher_builder()
        .initial_max_slice_len(required)
        .max_loaned_samples(1)
        .create()
    )

    measured: dict[str, list[float]] = {
        "loan_ms": [],
        "view_ms": [],
        "encode_ms": [],
        "send_ms": [],
        "total_ms": [],
        "throughput_gib_s": [],
    }

    total_runs = args.warmup + args.iterations

    for run_index in range(total_runs):
        warmup = run_index < args.warmup

        if warmup:
            phase = "W"
            sequence = run_index
            label = f"warmup {run_index + 1}/{args.warmup}"
        else:
            phase = "M"
            sequence = run_index - args.warmup
            label = f"measure {sequence + 1}/{args.iterations}"

        set_iteration_metadata(
            message,
            phase=phase,
            sequence=sequence,
        )

        t_total = time.perf_counter()

        t0 = time.perf_counter()
        sample = publisher.loan_slice_uninit(required)
        loan_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        shared_memory = slice_as_numpy(sample.payload())
        view_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        encoded = encode(message, out=shared_memory)
        encode_s = time.perf_counter() - t0
        encode_ms = encode_s * 1000.0

        if encoded.nbytes != required:
            raise RuntimeError("encoded size changed during benchmark")
        if not np.shares_memory(encoded, shared_memory):
            raise RuntimeError("NPB did not encode into the loaned shared memory")

        sample = sample.assume_init()

        t0 = time.perf_counter()
        sample.send()
        send_ms = (time.perf_counter() - t0) * 1000.0

        total_ms = (time.perf_counter() - t_total) * 1000.0
        throughput = (blob.nbytes / (1024**3)) / encode_s

        print(
            f"{label:<16} "
            f"loan={loan_ms:8.3f} ms  "
            f"view={view_ms:8.3f} ms  "
            f"encode={encode_ms:10.3f} ms  "
            f"send={send_ms:8.3f} ms  "
            f"total={total_ms:10.3f} ms  "
            f"{throughput:7.3f} GiB/s"
        )

        if not warmup:
            measured["loan_ms"].append(loan_ms)
            measured["view_ms"].append(view_ms)
            measured["encode_ms"].append(encode_ms)
            measured["send_ms"].append(send_ms)
            measured["total_ms"].append(total_ms)
            measured["throughput_gib_s"].append(throughput)

        # This pause is deliberately outside total_ms.
        if args.pause_ms:
            time.sleep(args.pause_ms / 1000.0)

    # Timings.
    print_profile_summary(
        "publisher timings",
        {
            "loan": measured["loan_ms"],
            "view": measured["view_ms"],
            "encode": measured["encode_ms"],
            "send": measured["send_ms"],
            "total": measured["total_ms"],
        },
        unit="ms",
    )

    print_profile_summary(
        "publisher encode throughput",
        {"encode": measured["throughput_gib_s"]},
        unit="GiB/s",
    )

    if args.hold_seconds > 0:
        time.sleep(args.hold_seconds)


if __name__ == "__main__":
    main()
