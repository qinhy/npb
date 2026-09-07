from __future__ import annotations

import argparse
import ctypes
import time

import iceoryx2 as iox2
import numpy as np

from npb import decode, decode_auto, peek

from common import (
    HugeBlob,
    SERVICE_NAME,
    check_markers,
    print_profile_summary,
    slice_as_numpy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile repeated receive/decode of huge NPB iceoryx2 samples."
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="number of measured samples expected (default: %(default)s)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="number of warm-up samples expected (default: %(default)s)",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="benchmark decode_auto() instead of typed decode()",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.iterations <= 0:
        raise SystemExit("--iterations must be > 0")
    if args.warmup < 0:
        raise SystemExit("--warmup must be >= 0")

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

    print(
        "subscriber ready; expecting "
        f"{args.warmup} warm-up + {args.iterations} measured huge samples..."
    )

    measured: dict[str, list[float]] = {
        "view_ms": [],
        "peek_ms": [],
        "decode_ms": [],
        "verify_ms": [],
        "total_ms": [],
    }

    received_warmup = 0
    received_measured = 0

    try:
        while received_measured < args.iterations:
            sample = subscriber.receive()

            if sample is None:
                node.wait(iox2.Duration.from_millis(100))
                continue

            # IMPORTANT: all ndarray views below depend on `sample`.
            t_total = time.perf_counter()

            t0 = time.perf_counter()
            binary = slice_as_numpy(sample.payload())
            view_ms = (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            info = peek(binary)
            peek_ms = (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            if args.auto:
                obj = decode_auto(binary)
                blob = obj["blob"]
                meta = obj["meta"]
            else:
                obj = decode(HugeBlob, binary)
                blob = obj.blob
                meta = obj.meta
            decode_ms = (time.perf_counter() - t0) * 1000.0

            phase = str(meta.get("benchmark_phase", "?"))
            sequence = str(meta.get("benchmark_sequence", "?"))

            t0 = time.perf_counter()
            check_markers(blob)
            zero_copy = np.shares_memory(binary, blob)
            if not zero_copy:
                raise RuntimeError("decoded blob does not share iceoryx2 memory")
            verify_ms = (time.perf_counter() - t0) * 1000.0

            total_ms = (time.perf_counter() - t_total) * 1000.0

            if phase == "W":
                received_warmup += 1
                label = f"warmup {received_warmup}/{args.warmup}"
            elif phase == "M":
                received_measured += 1
                label = f"measure {received_measured}/{args.iterations}"

                measured["view_ms"].append(view_ms)
                measured["peek_ms"].append(peek_ms)
                measured["decode_ms"].append(decode_ms)
                measured["verify_ms"].append(verify_ms)
                measured["total_ms"].append(total_ms)
            else:
                label = f"unknown phase={phase}"

            print(
                f"{label:<16} "
                f"seq={sequence}  "
                f"sample={info.total_bytes / (1024**3):.3f} GiB  "
                f"view={view_ms:8.3f} ms  "
                f"peek={peek_ms:8.3f} ms  "
                f"decode={decode_ms:8.3f} ms  "
                f"verify={verify_ms:8.3f} ms  "
                f"total={total_ms:8.3f} ms  "
                f"zero_copy={zero_copy}"
            )

            # Do not retain `obj`, `blob`, or `binary` beyond this iteration.
            # They point into the current iceoryx2 sample.
            del blob
            del obj
            del binary
            del sample

    except iox2.NodeWaitFailure:
        print("exit")
        return

    mode = "decode_auto" if args.auto else "typed decode"

    print_profile_summary(
        f"subscriber timings ({mode})",
        {
            "view": measured["view_ms"],
            "peek": measured["peek_ms"],
            "decode": measured["decode_ms"],
            "verify": measured["verify_ms"],
            "total": measured["total_ms"],
        },
        unit="ms",
    )


if __name__ == "__main__":
    main()
