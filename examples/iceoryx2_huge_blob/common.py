from __future__ import annotations

import os
import ctypes
from pathlib import Path
import tempfile
import time

import numpy as np

from npb import BinaryModel, binary_schema


SERVICE_NAME = "npb/huge-blob/v1"
DEFAULT_GIB = 2.0
MARKER_BYTES = 4096


class Tensor(BinaryModel):
    name: str
    unit: str
    values: np.ndarray


@binary_schema("org.example.npb.iceoryx2.huge-blob", version=1)
class HugeBlob(BinaryModel):
    message_id: str
    created_ns: int
    meta: dict[str, object]

    # The genuinely huge binary leaf.
    blob: np.ndarray

    # Extra nested binary leaves show that this is not merely a raw-blob
    # transport demo.
    tensors: list[Tensor]
    calibration: dict[str, np.ndarray]


def make_sparse_blob(size_bytes: int, path: Path | None = None) -> tuple[np.memmap, Path]:
    """Create a disk-backed sparse blob without first allocating it in RAM."""
    if size_bytes < 2 * MARKER_BYTES:
        raise ValueError(f"blob must be at least {2 * MARKER_BYTES} bytes")

    if path is None:
        fd, raw_path = tempfile.mkstemp(prefix="npb-huge-", suffix=".blob")
        os.close(fd)
        path = Path(raw_path)

    # truncate() normally creates a sparse file on filesystems that support it.
    with path.open("wb") as f:
        f.truncate(size_bytes)

    blob = np.memmap(path, mode="r+", dtype=np.uint8, shape=(size_bytes,))

    # Put deterministic data at both ends. The middle stays sparse/zero in the
    # source file, but NPB still copies the complete multi-GB logical blob into
    # the iceoryx2 shared-memory sample.
    marker = np.arange(MARKER_BYTES, dtype=np.uint32).astype(np.uint8)
    blob[:MARKER_BYTES] = marker
    blob[-MARKER_BYTES:] = marker[::-1]
    blob.flush()

    return blob, path


def make_message(blob: np.ndarray) -> HugeBlob:
    return HugeBlob(
        message_id=f"huge-{time.time_ns()}",
        created_ns=time.time_ns(),
        meta={
            "purpose": "NPB direct-to-iceoryx2 huge shared-memory demo",
            "blob_bytes": int(blob.nbytes),
            "blob_gib": float(blob.nbytes / (1024**3)),
            "layout": {
                "semantic": "Pydantic + JSON manifest",
                "binary": "raw NumPy leaves",
                "transport": "one dynamic iceoryx2 shared-memory sample",
            },
        },
        blob=blob,
        tensors=[
            Tensor(
                name="imu",
                unit="m/s^2",
                values=np.arange(300_000, dtype=np.float32).reshape(100_000, 3),
            ),
            Tensor(
                name="timestamps",
                unit="ns",
                values=np.arange(100_000, dtype=np.int64),
            ),
        ],
        calibration={
            "camera_matrix": np.eye(3, dtype=np.float64),
            "distortion": np.zeros(8, dtype=np.float32),
        },
    )


def check_markers(blob: np.ndarray) -> None:
    marker = np.arange(MARKER_BYTES, dtype=np.uint32).astype(np.uint8)
    if not np.array_equal(blob[:MARKER_BYTES], marker):
        raise RuntimeError("front marker mismatch")
    if not np.array_equal(blob[-MARKER_BYTES:], marker[::-1]):
        raise RuntimeError("tail marker mismatch")


def slice_as_numpy(payload) -> np.ndarray:
    """Zero-copy NumPy uint8 view over iceoryx2 0.9.3 Slice[c_uint8].

    iceoryx2 0.9.3 exposes Slice.as_ptr() + Slice.len(), but not
    Slice.as_memory_view().  Build a ctypes pointer view over the shared
    memory and let NumPy wrap it without copying.
    """
    length = payload.len()
    ptr = ctypes.cast(
        payload.as_ptr(),
        ctypes.POINTER(ctypes.c_uint8),
    )

    return np.ctypeslib.as_array(
        ptr,
        shape=(length,),
    )


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def print_profile_summary(
    title: str,
    samples: dict[str, list[float]],
    *,
    unit: str = "ms",
) -> None:
    """Pretty-print min/p50/p95/max/mean for repeated benchmark samples."""
    print()
    print(f"=== {title} ===")
    print(
        f"{'metric':<18}"
        f"{'count':>8}"
        f"{'min':>12}"
        f"{'p50':>12}"
        f"{'p95':>12}"
        f"{'max':>12}"
        f"{'mean':>12}"
        f"  {unit}"
    )

    for name, values in samples.items():
        if not values:
            continue

        arr = np.asarray(values, dtype=np.float64)

        print(
            f"{name:<18}"
            f"{len(values):>8d}"
            f"{arr.min():>12.3f}"
            f"{percentile(values, 50):>12.3f}"
            f"{percentile(values, 95):>12.3f}"
            f"{arr.max():>12.3f}"
            f"{arr.mean():>12.3f}"
            f"  {unit}"
        )
