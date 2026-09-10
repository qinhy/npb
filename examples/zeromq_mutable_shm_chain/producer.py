from __future__ import annotations

import argparse
import time

import numpy as np
from npb import ZmqPush, encode
from mstore import connect

from common import ChainMessage, DEFAULT_MSTORE_ENDPOINT, ShmArrayRef, endpoint, new_frame_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gib", type=float, default=2.0)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--mstore-endpoint", default=DEFAULT_MSTORE_ENDPOINT)
    parser.add_argument(
        "--token-ttl",
        type=float,
        default=3600.0,
        help="seconds before the chain's write capability expires",
    )
    args = parser.parse_args()

    size = int(args.gib * 1024**3)
    if size <= 0:
        raise SystemExit("--gib must produce a positive image size")
    if args.count <= 0:
        raise SystemExit("--count must be > 0")
    if args.count > size:
        raise SystemExit("--count must not exceed the uint8 array length")
    if args.token_ttl <= 0:
        raise SystemExit("--token-ttl must be > 0")

    with connect(args.mstore_endpoint, cache_size=0) as client:
        client.ping()  # establish persistent control connection before timing create

        print("allocating ndarray directly in mutable-shm-store...")
        t0 = time.perf_counter()
        owner = client.create(
            shape=(size,),
            dtype=np.uint8,
            metadata={"purpose": "NPB + mutable-shm-store + ZeroMQ cache-hit demo"},
        )
        alloc_ms = (time.perf_counter() - t0) * 1000.0

        image = owner.numpy()
        t0 = time.perf_counter()
        image.fill(7)
        # Each in-flight control message owns a different byte slot. That makes the
        # multi-message demo race-free while every frame still references the SAME
        # large shared-memory object.
        image[: args.count] = 0
        init_ms = (time.perf_counter() - t0) * 1000.0

        # Do not put the owner/admin token on the pipeline.
        chain_token = owner.issue("write", expires_in=args.token_ttl)
        ref = ShmArrayRef(
            object_id=owner.object_id,
            token=chain_token,
            generation=owner.generation,
        )

        print(f"source image     : {image.nbytes / 1024**3:.3f} GiB")
        print(f"mstore object    : {owner.object_id}")
        print(f"generation       : {owner.generation}")
        print(f"frames           : {args.count}")
        print(f"allocate         : {alloc_ms:.3f} ms")
        print(f"initialize       : {init_ms:.3f} ms")

        with ZmqPush.connect(
            endpoint(0),
            hwm=max(4, args.count),
            linger_ms=2000,
            send_timeout_ms=5000,
        ) as sender:
            for sequence in range(args.count):
                message = ChainMessage(
                    frame_id=new_frame_id(),
                    sequence=sequence,
                    stage=0,
                    meta={"purpose": "in-place mutable shared-memory cache-hit chain"},
                    image=ref,
                )

                t0 = time.perf_counter()
                control = encode(message)
                encode_us = (time.perf_counter() - t0) * 1e6

                t0 = time.perf_counter()
                sender.send(control)
                send_us = (time.perf_counter() - t0) * 1e6

                print(
                    f"sent seq={sequence:03d} "
                    f"control={control.nbytes / 1024:.1f} KiB "
                    f"encode={encode_us:.2f} us send={send_us:.2f} us"
                )

        # The daemon owns the region; producer can unmap and exit.
        del image
        owner.close()


if __name__ == "__main__":
    main()
