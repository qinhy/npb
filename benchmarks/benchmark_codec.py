"""Microbenchmarks for the in-memory NPB codec.

Run the default suite with::

    uv run python benchmarks/benchmark_codec.py

The harness intentionally uses only the standard library plus NPB's runtime
dependencies, so a development checkout can run it without extra packages.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pydantic

from npb import BinaryModel, binary_schema, decode, decode_auto, encode, encoded_size, peek

MIB = 1024**2


class Image(BinaryModel):
    name: str
    pixels: np.ndarray


@binary_schema("benchmarks.capture", version=1)
class Capture(BinaryModel):
    capture_id: str
    images: list[Image]


@dataclass(frozen=True)
class Result:
    payload_bytes: int
    encoded_bytes: int
    operation: str
    iterations: int
    repeats: int
    median_seconds: float
    min_seconds: float
    max_seconds: float
    stdev_seconds: float

    @property
    def operations_per_second(self) -> float:
        return 1.0 / self.median_seconds

    @property
    def gib_per_second(self) -> float | None:
        if self.operation in {"encoded_size", "peek"}:
            return None
        return self.payload_bytes / self.median_seconds / 1024**3


def parse_size(value: str) -> int:
    """Parse a positive byte count, optionally suffixed by KiB/MiB/GiB."""
    normalized = value.strip().lower().replace("ib", "b")
    suffixes = {"kb": 1024, "mb": 1024**2, "gb": 1024**3, "b": 1}
    for suffix, multiplier in suffixes.items():
        if normalized.endswith(suffix):
            number = normalized[: -len(suffix)]
            break
    else:
        number = normalized
        multiplier = 1

    try:
        size = int(float(number) * multiplier)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid size: {value!r}") from exc
    if size <= 0:
        raise argparse.ArgumentTypeError("sizes must be greater than zero")
    return size


def make_capture(payload_bytes: int, array_count: int) -> Capture:
    """Build a representative model whose arrays total ``payload_bytes``."""
    if array_count <= 0:
        raise ValueError("array_count must be greater than zero")

    quotient, remainder = divmod(payload_bytes, array_count)
    images = []
    for index in range(array_count):
        array_bytes = quotient + (index < remainder)
        pixels = np.arange(array_bytes, dtype=np.uint8)
        images.append(Image(name=f"camera-{index}", pixels=pixels))
    return Capture(capture_id="benchmark-capture", images=images)


def _run_batch(function: Callable[[], Any], iterations: int) -> float:
    started = time.perf_counter_ns()
    for _ in range(iterations):
        function()
    return (time.perf_counter_ns() - started) / 1e9


def _calibrate(function: Callable[[], Any], target_seconds: float) -> int:
    iterations = 1
    while iterations < 1_000_000:
        elapsed = _run_batch(function, iterations)
        if elapsed >= target_seconds:
            return iterations
        if elapsed > 0:
            estimate = int(iterations * target_seconds / elapsed)
            iterations = max(iterations + 1, min(estimate, iterations * 10))
        else:
            iterations *= 10
    return iterations


def measure(
    operation: str,
    function: Callable[[], Any],
    *,
    payload_bytes: int,
    encoded_bytes: int,
    repeats: int,
    target_seconds: float,
) -> Result:
    """Measure one operation with adaptive iteration counts."""
    for _ in range(3):
        function()

    iterations = _calibrate(function, target_seconds)
    samples = []
    for _ in range(repeats):
        gc.collect()
        gc.disable()
        try:
            elapsed = _run_batch(function, iterations)
        finally:
            gc.enable()
        samples.append(elapsed / iterations)

    return Result(
        payload_bytes=payload_bytes,
        encoded_bytes=encoded_bytes,
        operation=operation,
        iterations=iterations,
        repeats=repeats,
        median_seconds=statistics.median(samples),
        min_seconds=min(samples),
        max_seconds=max(samples),
        stdev_seconds=statistics.pstdev(samples),
    )


def benchmark_size(
    payload_bytes: int,
    *,
    array_count: int,
    repeats: int,
    target_seconds: float,
) -> list[Result]:
    capture = make_capture(payload_bytes, array_count)
    binary = encode(capture)
    output = np.empty(binary.nbytes, dtype=np.uint8)

    restored = decode(Capture, binary)
    assert sum(image.pixels.nbytes for image in restored.images) == payload_bytes
    assert all(np.shares_memory(binary, image.pixels) for image in restored.images)

    cases: list[tuple[str, Callable[[], Any]]] = [
        ("encoded_size", lambda: encoded_size(capture)),
        ("encode/allocate", lambda: encode(capture)),
        ("encode/preallocated", lambda: encode(capture, out=output)),
        ("decode/auto", lambda: decode_auto(binary)),
        ("decode/typed", lambda: decode(Capture, binary)),
        ("peek", lambda: peek(binary)),
    ]
    return [
        measure(
            name,
            function,
            payload_bytes=payload_bytes,
            encoded_bytes=binary.nbytes,
            repeats=repeats,
            target_seconds=target_seconds,
        )
        for name, function in cases
    ]


def _format_duration(seconds: float) -> str:
    if seconds < 1e-6:
        return f"{seconds * 1e9:.1f} ns"
    if seconds < 1e-3:
        return f"{seconds * 1e6:.1f} us"
    return f"{seconds * 1e3:.2f} ms"


def _format_size(size: int) -> str:
    if size < MIB:
        return f"{size / 1024:.3g} KiB"
    return f"{size / MIB:.3g} MiB"


def print_results(results: Sequence[Result]) -> None:
    headings = ("payload", "operation", "median", "stdev", "ops/s", "payload GiB/s")
    rows = []
    for result in results:
        variability = result.stdev_seconds / result.median_seconds * 100
        throughput = result.gib_per_second
        rows.append(
            (
                _format_size(result.payload_bytes),
                result.operation,
                _format_duration(result.median_seconds),
                f"{variability:.1f}%",
                f"{result.operations_per_second:,.0f}",
                "n/a" if throughput is None else f"{throughput:.2f}",
            )
        )

    widths = [max(len(row[index]) for row in [headings, *rows]) for index in range(len(headings))]
    print("  ".join(value.ljust(width) for value, width in zip(headings, widths, strict=True)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(width) for value, width in zip(row, widths, strict=True)))


def environment() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pydantic": pydantic.__version__,
    }


def write_json(path: Path, results: Sequence[Result], arguments: argparse.Namespace) -> None:
    document = {
        "environment": environment(),
        "configuration": {
            "sizes": arguments.sizes,
            "arrays": arguments.arrays,
            "repeats": arguments.repeats,
            "target_seconds": arguments.target_seconds,
        },
        "results": [
            {
                **asdict(result),
                "operations_per_second": result.operations_per_second,
                "payload_gib_per_second": result.gib_per_second,
            }
            for result in results
        ],
    }
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes",
        type=parse_size,
        nargs="+",
        default=[1024, MIB, 16 * MIB],
        metavar="SIZE",
        help="array payload sizes (default: 1KiB 1MiB 16MiB)",
    )
    parser.add_argument("--arrays", type=int, default=1, help="arrays per model (default: 1)")
    parser.add_argument("--repeats", type=int, default=7, help="samples per operation (default: 7)")
    parser.add_argument(
        "--target-seconds",
        type=float,
        default=0.2,
        help="minimum duration of each sample (default: 0.2)",
    )
    parser.add_argument("--json", type=Path, help="also write machine-readable results")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="smoke test with one 1 MiB payload and short samples",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    if arguments.arrays <= 0:
        parser.error("--arrays must be greater than zero")
    if arguments.repeats <= 0:
        parser.error("--repeats must be greater than zero")
    if arguments.target_seconds <= 0:
        parser.error("--target-seconds must be greater than zero")
    if arguments.quick:
        arguments.sizes = [MIB]
        arguments.repeats = 2
        arguments.target_seconds = 0.01

    details = environment()
    print(
        f"Python {details['python']} | NumPy {details['numpy']} | "
        f"Pydantic {details['pydantic']} | {details['platform']}"
    )
    print(f"arrays/model: {arguments.arrays}; repeats: {arguments.repeats}\n")

    results = []
    for size in arguments.sizes:
        results.extend(
            benchmark_size(
                size,
                array_count=arguments.arrays,
                repeats=arguments.repeats,
                target_seconds=arguments.target_seconds,
            )
        )
    print_results(results)

    if arguments.json is not None:
        write_json(arguments.json, results, arguments)
        print(f"\nwrote {arguments.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
