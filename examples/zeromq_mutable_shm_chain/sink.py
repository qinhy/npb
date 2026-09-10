from __future__ import annotations

import argparse
import statistics
import time

from npb import ZmqPull, decode
from mstore import connect

from common import ChainMessage, DEFAULT_MSTORE_ENDPOINT, endpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--after-stage", type=int, required=True)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--mstore-endpoint", default=DEFAULT_MSTORE_ENDPOINT)
    parser.add_argument("--cache-size", type=int, default=64)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    if args.after_stage < 0:
        raise SystemExit("--after-stage must be >= 0")
    if args.count <= 0:
        raise SystemExit("--count must be > 0")

    input_endpoint = endpoint(args.after_stage)
    expected = sum(range(1, args.after_stage + 1)) & 0xFF
    open_times: list[float] = []

    with connect(args.mstore_endpoint, cache_size=args.cache_size) as client:
        t0 = time.perf_counter()
        client.ping()
        warmup_us = (time.perf_counter() - t0) * 1e6

        print(
            f"sink: mstore ready in {warmup_us:.2f} us; "
            f"waiting on {input_endpoint}; count={args.count}"
        )

        with ZmqPull.bind(input_endpoint, hwm=max(4, args.count)) as receiver:
            for _ in range(args.count):
                t0 = time.perf_counter()
                control = receiver.recv()
                recv_us = (time.perf_counter() - t0) * 1e6

                t0 = time.perf_counter()
                message = decode(ChainMessage, control)
                decode_us = (time.perf_counter() - t0) * 1e6

                ref = message.image
                t0 = time.perf_counter()
                shared = client.open(
                    ref.object_id,
                    ref.token,
                    mode="read",
                    cache=not args.no_cache,
                )
                open_us = (time.perf_counter() - t0) * 1e6
                open_times.append(open_us)

                if shared.generation != ref.generation:
                    shared.close()
                    raise RuntimeError(
                        f"object generation changed: expected {ref.generation}, "
                        f"got {shared.generation}"
                    )

                image = shared.numpy()
                slot = message.sequence
                value = int(image[slot])
                readonly = not image.flags.writeable
                cache_hit = shared.cache_hit

                if message.stage != args.after_stage:
                    raise RuntimeError(
                        f"seq {slot}: expected stage={args.after_stage}, got {message.stage}"
                    )
                if value != expected:
                    raise RuntimeError(
                        f"seq {slot}: expected shared value {expected}, got {value}"
                    )
                if not readonly:
                    raise RuntimeError("sink mapping unexpectedly writable")

                nbytes = image.nbytes
                del image
                shared.close()

                print(
                    f"received seq={message.sequence:03d} stage={message.stage} "
                    f"image={nbytes / 1024**3:.3f} GiB "
                    f"control={control.nbytes / 1024:.1f} KiB "
                    f"recv={recv_us:.2f} us decode={decode_us:.2f} us "
                    f"open={open_us:.2f} us cache_hit={cache_hit} "
                    f"value={value} readonly={readonly} object={ref.object_id}"
                )

        info = client.cache_info()
        print(
            f"sink done: open_p50={statistics.median(open_times):.2f} us "
            f"cache_size={info['size']} hits={info['hits']} "
            f"misses={info['misses']} evictions={info['evictions']}"
        )


if __name__ == "__main__":
    main()
