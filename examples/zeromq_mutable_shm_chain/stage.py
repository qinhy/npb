from __future__ import annotations

import argparse
import statistics
import time

from npb import ZmqPull, ZmqPush, decode, encode
from mstore import connect

from common import ChainMessage, DEFAULT_MSTORE_ENDPOINT, endpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--mstore-endpoint", default=DEFAULT_MSTORE_ENDPOINT)
    parser.add_argument("--cache-size", type=int, default=64)
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="force every frame through daemon authorization + OS mapping attach",
    )
    args = parser.parse_args()

    if args.stage <= 0:
        raise SystemExit("--stage must be >= 1")
    if args.count <= 0:
        raise SystemExit("--count must be > 0")

    input_endpoint = endpoint(args.stage - 1)
    output_endpoint = endpoint(args.stage)
    open_times: list[float] = []

    with connect(args.mstore_endpoint, cache_size=args.cache_size) as client:
        # Connect before receiving the first frame so the first open measurement does
        # not include named-pipe/socket establishment.
        t0 = time.perf_counter()
        client.ping()
        warmup_us = (time.perf_counter() - t0) * 1e6

        print(
            f"stage {args.stage}: mstore ready in {warmup_us:.2f} us; "
            f"waiting on {input_endpoint}; count={args.count}"
        )

        with (
            ZmqPull.bind(input_endpoint, hwm=max(4, args.count), recv_timeout_ms=None) as receiver,
            ZmqPush.connect(
                output_endpoint,
                hwm=max(4, args.count),
                linger_ms=2000,
                send_timeout_ms=5000,
            ) as sender,
        ):
            for _ in range(args.count):
                t0 = time.perf_counter()
                incoming = receiver.recv()
                recv_us = (time.perf_counter() - t0) * 1e6

                t0 = time.perf_counter()
                message = decode(ChainMessage, incoming)
                decode_us = (time.perf_counter() - t0) * 1e6

                ref = message.image
                t0 = time.perf_counter()
                shared = client.open(
                    ref.object_id,
                    ref.token,
                    mode="write",
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
                if slot < 0 or slot >= image.size:
                    shared.close()
                    raise RuntimeError(f"sequence slot {slot} is outside shared array")

                # Each frame mutates its own byte, so multiple frames can be in flight
                # without racing one another while all reuse the same 2-GiB mapping.
                before = int(image[slot])
                t0 = time.perf_counter()
                image[slot] = (before + args.stage) & 0xFF
                mutate_us = (time.perf_counter() - t0) * 1e6
                after = int(image[slot])

                message.stage = args.stage
                message.meta["last_stage"] = args.stage

                t0 = time.perf_counter()
                outgoing = encode(message)
                encode_us = (time.perf_counter() - t0) * 1e6

                t0 = time.perf_counter()
                sender.send(outgoing)
                send_us = (time.perf_counter() - t0) * 1e6

                cache_hit = shared.cache_hit
                nbytes = image.nbytes
                del image
                shared.close()

                print(
                    f"stage={args.stage} seq={message.sequence:03d} "
                    f"image={nbytes / 1024**3:.3f} GiB "
                    f"recv={recv_us:.2f} us decode={decode_us:.2f} us "
                    f"open={open_us:.2f} us cache_hit={cache_hit} "
                    f"mutate={mutate_us:.2f} us encode={encode_us:.2f} us "
                    f"send={send_us:.2f} us value={before}->{after}"
                )

        info = client.cache_info()
        print(
            f"stage {args.stage} done: open_p50={statistics.median(open_times):.2f} us "
            f"cache_size={info['size']} hits={info['hits']} "
            f"misses={info['misses']} evictions={info['evictions']}"
        )


if __name__ == "__main__":
    main()
