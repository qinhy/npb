"""NPB encoder and decoders."""

from __future__ import annotations

from typing import Any, TypeVar

import numpy as np
from pydantic import BaseModel
from pydantic_core import from_json, to_json

from ._errors import BufferTooSmallError, FormatError, SchemaMismatchError
from ._format import (
    ARRAY_ALIGNMENT,
    FORMAT_VERSION,
    FRAME_SIZE,
    HEADER,
    HEADER_SIZE,
    MAGIC,
    BinaryInfo,
    align_up,
    parse_header,
)
from ._schema import schema_identity

T = TypeVar("T", bound=BaseModel)


def _validate_ndarray(arr: np.ndarray) -> np.ndarray:
    """Return a C-contiguous array that is safe for raw-byte serialization."""
    if arr.dtype.hasobject:
        raise TypeError("NumPy object dtypes are not supported")
    if arr.dtype.fields is not None:
        raise TypeError("NumPy structured dtypes are not supported")
    return np.ascontiguousarray(arr)


def _manifest_tree(
    node: Any,
    arrays: list[tuple[int, np.ndarray]],
    cursor: list[int],
) -> Any:
    """Replace ndarray leaves with JSON-friendly binary references."""
    if isinstance(node, np.ndarray):
        arr = _validate_ndarray(node)

        rel_offset = align_up(cursor[0], ARRAY_ALIGNMENT)
        cursor[0] = rel_offset + arr.nbytes
        arrays.append((rel_offset, arr))

        return {
            "$bin": {
                "kind": "ndarray",
                "offset": rel_offset,
                "nbytes": arr.nbytes,
                "dtype": arr.dtype.str,
                "shape": list(arr.shape),
            }
        }

    if isinstance(node, dict):
        return {key: _manifest_tree(value, arrays, cursor) for key, value in node.items()}

    if isinstance(node, (list, tuple)):
        return [_manifest_tree(value, arrays, cursor) for value in node]

    return node


def _json_fallback(value: Any) -> Any:
    """Handle small NumPy scalar values that occur in metadata."""
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"value is not JSON serializable: {type(value).__name__}")


def _prepare_output(out: np.ndarray | None, total_size: int, data_start: int) -> np.ndarray:
    """Allocate or slice a writable caller-provided output buffer."""
    if out is None:
        binary = np.empty(total_size, dtype=np.uint8)
    else:
        if not isinstance(out, np.ndarray):
            raise TypeError("out must be a numpy.ndarray")
        if out.dtype != np.uint8:
            raise TypeError("out dtype must be np.uint8")
        if out.ndim != 1:
            raise ValueError("out must be a 1-D array")
        if not out.flags.c_contiguous:
            raise ValueError("out must be C-contiguous")
        if not out.flags.writeable:
            raise ValueError("out must be writable")
        if out.nbytes < total_size:
            raise BufferTooSmallError(
                f"output buffer too small: need {total_size:,} bytes, got {out.nbytes:,}"
            )
        binary = out[:total_size]

    # Deterministic header/metadata frame and metadata padding.
    binary[:data_start] = 0
    return binary


def encode(model: BaseModel, *, out: np.ndarray | None = None) -> np.ndarray:
    """Encode a Pydantic model into one frame-aligned ``np.uint8`` array.

    Parameters
    ----------
    model:
        Top-level Pydantic model decorated with :func:`npb.binary_schema`.
    out:
        Optional writable, C-contiguous, 1-D ``np.uint8`` array to reuse.
        If larger than required, the returned value is a view covering only
        the encoded region.

    Returns
    -------
    numpy.ndarray
        Self-contained, frame-aligned encoded data.
    """
    if not isinstance(model, BaseModel):
        raise TypeError("encode() requires a Pydantic BaseModel instance")

    schema_id, schema_version = schema_identity(type(model))
    python_tree = model.model_dump(mode="python")

    arrays: list[tuple[int, np.ndarray]] = []
    cursor = [0]
    manifest = _manifest_tree(python_tree, arrays, cursor)

    data_bytes = cursor[0]
    metadata_json = to_json(manifest, fallback=_json_fallback)
    metadata_bytes = len(metadata_json)

    data_start = align_up(HEADER_SIZE + metadata_bytes, FRAME_SIZE)
    total_used = data_start + data_bytes
    total_size = align_up(max(total_used, FRAME_SIZE), FRAME_SIZE)

    binary = _prepare_output(out, total_size, data_start)

    HEADER.pack_into(
        binary,
        0,
        MAGIC,
        FORMAT_VERSION,
        0,
        FRAME_SIZE,
        schema_id.bytes,
        schema_version,
        metadata_bytes,
        data_start,
        data_bytes,
        total_size,
    )

    binary[HEADER_SIZE : HEADER_SIZE + metadata_bytes] = np.frombuffer(
        metadata_json,
        dtype=np.uint8,
    )

    previous_end = 0

    for rel_offset, arr in arrays:
        # Zero tiny alignment gaps if the caller reused an old buffer.
        if rel_offset > previous_end:
            binary[data_start + previous_end : data_start + rel_offset] = 0

        absolute_offset = data_start + rel_offset
        binary[absolute_offset : absolute_offset + arr.nbytes] = arr.view(np.uint8).reshape(-1)
        previous_end = rel_offset + arr.nbytes

    # Zero frame padding after the meaningful data region.
    final_data_end = data_start + data_bytes
    if final_data_end < total_size:
        binary[final_data_end:total_size] = 0

    return binary


def _restore_tree(
    node: Any,
    binary: np.ndarray,
    *,
    data_start: int,
    data_bytes: int,
) -> Any:
    """Replace binary-reference objects with zero-copy NumPy views."""
    if isinstance(node, dict):
        if set(node) == {"$bin"}:
            ref = node["$bin"]

            if isinstance(ref, dict) and ref.get("kind") == "ndarray":
                try:
                    rel_offset = int(ref["offset"])
                    nbytes = int(ref["nbytes"])
                    dtype = np.dtype(ref["dtype"])
                    shape = tuple(int(dim) for dim in ref["shape"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise FormatError("invalid ndarray reference") from exc

                if dtype.hasobject or dtype.fields is not None:
                    raise FormatError("unsupported ndarray dtype in binary reference")
                if rel_offset < 0 or nbytes < 0:
                    raise FormatError("negative ndarray offset or size")
                if any(dim < 0 for dim in shape):
                    raise FormatError("negative ndarray dimension")
                if dtype.itemsize <= 0 or nbytes % dtype.itemsize != 0:
                    raise FormatError("invalid ndarray byte size")

                expected_items = 1
                for dim in shape:
                    expected_items *= dim

                if expected_items * dtype.itemsize != nbytes:
                    raise FormatError("ndarray shape does not match its byte size")

                rel_end = rel_offset + nbytes
                if rel_end > data_bytes:
                    raise FormatError("ndarray reference exceeds the DATA section")

                absolute_offset = data_start + rel_offset
                absolute_end = absolute_offset + nbytes
                if absolute_end > binary.nbytes:
                    raise FormatError("ndarray reference exceeds the encoded array")

                return np.frombuffer(
                    binary,
                    dtype=dtype,
                    count=expected_items,
                    offset=absolute_offset,
                ).reshape(shape)

        return {
            key: _restore_tree(
                value,
                binary,
                data_start=data_start,
                data_bytes=data_bytes,
            )
            for key, value in node.items()
        }

    if isinstance(node, list):
        return [
            _restore_tree(
                value,
                binary,
                data_start=data_start,
                data_bytes=data_bytes,
            )
            for value in node
        ]

    return node


def _decode_tree(binary: np.ndarray, info: BinaryInfo) -> dict[str, Any]:
    """Decode metadata and reconstruct ndarray leaves."""
    metadata_start = HEADER_SIZE
    metadata_end = metadata_start + info.metadata_bytes

    # pydantic-core currently accepts bytes/bytearray/str, so this is the only
    # small copy during decode. ndarray payloads remain zero-copy.
    metadata_json = binary[metadata_start:metadata_end].tobytes()

    try:
        manifest = from_json(metadata_json)
    except ValueError as exc:
        raise FormatError("invalid JSON metadata") from exc

    tree = _restore_tree(
        manifest,
        binary,
        data_start=info.data_start,
        data_bytes=info.data_bytes,
    )

    if not isinstance(tree, dict):
        raise FormatError("top-level NPB value must be an object")

    return tree


def peek(binary: np.ndarray) -> BinaryInfo:
    """Inspect the fixed NPB header without decoding metadata or arrays."""
    return parse_header(binary)


def decode_auto(binary: np.ndarray) -> dict[str, Any]:
    """Decode to plain Python containers with zero-copy ``np.ndarray`` leaves.

    This function never imports or executes a class named by the encoded data.
    """
    info = parse_header(binary)
    return _decode_tree(binary, info)


def decode(model_type: type[T], binary: np.ndarray) -> T:
    """Decode and validate using an explicitly supplied Pydantic model class."""
    if not isinstance(model_type, type) or not issubclass(model_type, BaseModel):
        raise TypeError("model_type must be a Pydantic BaseModel class")

    expected_id, expected_version = schema_identity(model_type)
    info = parse_header(binary)

    if info.schema_id != expected_id:
        raise SchemaMismatchError(
            f"schema UUID mismatch: binary={info.schema_id}, class={expected_id}"
        )

    if info.schema_version != expected_version:
        raise SchemaMismatchError(
            "schema version mismatch: "
            f"binary={info.schema_version}, class={expected_version}"
        )

    return model_type.model_validate(_decode_tree(binary, info))
