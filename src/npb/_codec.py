"""NPB encoder and decoders."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import suppress
from typing import Any, TypeVar

import numpy as np
from pydantic import BaseModel
from pydantic_core import from_json, to_json

from ._blob import DEFAULT_OBJECT_ID_LENGTH, BlobRef, BlobStore
from ._errors import (
    BlobStoreRequiredError,
    BufferTooSmallError,
    FormatError,
    SchemaMismatchError,
)
from ._format import (
    ARRAY_ALIGNMENT,
    FLAG_EXTENDED_MANIFEST,
    FORMAT_VERSION,
    FRAME_SIZE,
    HEADER,
    HEADER_SIZE,
    MAGIC,
    BinaryInfo,
    BinaryLike,
    align_up,
    as_binary_array,
    parse_header,
    require_frame_size,
)
from ._schema import schema_identity

T = TypeVar("T", bound=BaseModel)

# Manifest objects with exactly one of these keys are binary references.
_BIN = "$bin"
_BLOB = "$blob"
# User dictionaries that would be mistaken for a reference are written with
# one extra leading "$" (and FLAG_EXTENDED_MANIFEST); the decoder strips it.
_ESCAPE = "$"
_ESCAPED_PREFIX = "$$"

# Leaves that can never contain an ndarray; checked first because they are by
# far the most common nodes in a manifest.
_PLAIN_SCALARS = frozenset({str, int, float, bool, type(None)})


def _validate_ndarray(arr: np.ndarray) -> None:
    """Validate that an ndarray is safe for raw-byte serialization."""
    if arr.dtype.hasobject:
        raise TypeError("NumPy object dtypes are not supported")
    if arr.dtype.fields is not None:
        raise TypeError("NumPy structured dtypes are not supported")


def _needs_escape(key: object) -> bool:
    return isinstance(key, str) and (
        key in (_BIN, _BLOB) or key.startswith(_ESCAPED_PREFIX)
    )


class _ManifestBuilder:
    """Replace binary leaves of a dumped model with manifest references.

    With ``dry_run=True`` nothing is written to the blob store: arrays that
    would be externalized are represented by the store's ``preview_ref`` (or
    a fixed-size placeholder), which is enough to compute the exact layout.
    """

    __slots__ = (
        "_external",
        "_inline",
        "_preview",
        "blob_store",
        "created",
        "cursor",
        "dry_run",
        "extended",
        "externalize_min_bytes",
        "pending_puts",
        "segments",
    )

    def __init__(
        self,
        *,
        blob_store: BlobStore | None,
        externalize_min_bytes: int | None,
        dry_run: bool,
    ) -> None:
        self.blob_store = blob_store
        self.externalize_min_bytes = (
            externalize_min_bytes if blob_store is not None else None
        )
        self.dry_run = dry_run

        #: Inline payloads in DATA-section order: (relative offset, payload).
        self.segments: list[tuple[int, np.ndarray | bytes | bytearray]] = []
        self.cursor = 0
        #: True once the manifest uses a feature NPB <= 0.3 cannot read.
        self.extended = False
        #: References known to have been newly created by this builder.
        self.created: list[BlobRef] = []
        #: Number of arrays a dry run found that the real run must store.
        self.pending_puts = 0

        self._preview: Callable[[np.ndarray], BlobRef] | None = (
            getattr(blob_store, "preview_ref", None)
            if self.externalize_min_bytes is not None
            else None
        )
        # The same ndarray object may be referenced from several fields.
        # Store it once and point every occurrence at the same payload.
        self._inline: dict[int, int] = {}
        self._external: dict[int, BlobRef] = {}

    # -- leaves ---------------------------------------------------------

    def _placeholder(self, array: np.ndarray) -> BlobRef:
        return BlobRef(
            store=str(self.blob_store.kind),  # type: ignore[union-attr]
            object_id="0" * DEFAULT_OBJECT_ID_LENGTH,
            offset=0,
            nbytes=array.nbytes,
            dtype=array.dtype.str,
            shape=tuple(array.shape),
        )

    def _externalize(self, array: np.ndarray) -> BlobRef:
        preview = self._preview(array) if self._preview is not None else None

        if self.dry_run:
            self.pending_puts += 1
            return preview if preview is not None else self._placeholder(array)

        ref = self.blob_store.put_array(array)  # type: ignore[union-attr]

        # Only a store with ``preview_ref`` lets us tell a new object from a
        # reused one, and only new objects may be deleted on rollback.
        if preview is not None and ref != preview:
            self.created.append(ref)

        return ref

    def _array(self, array: np.ndarray) -> dict[str, Any]:
        _validate_ndarray(array)
        key = id(array)

        threshold = self.externalize_min_bytes
        if threshold is not None and array.nbytes >= threshold:
            ref = self._external.get(key)
            if ref is None:
                ref = self._externalize(array)
                self._external[key] = ref
            return {_BLOB: ref.to_manifest()}

        rel_offset = self._inline.get(key)
        if rel_offset is None:
            rel_offset = align_up(self.cursor, ARRAY_ALIGNMENT)
            self.cursor = rel_offset + array.nbytes
            self.segments.append((rel_offset, array))
            self._inline[key] = rel_offset

        return {
            _BIN: {
                "kind": "ndarray",
                "offset": rel_offset,
                "nbytes": array.nbytes,
                "dtype": array.dtype.str,
                "shape": list(array.shape),
            }
        }

    def _bytes(self, value: bytes | bytearray) -> Any:
        # Valid UTF-8 keeps its historical JSON-string representation, so
        # everything NPB <= 0.3 could encode still produces identical output.
        try:
            value.decode("utf-8")
        except UnicodeDecodeError:
            pass
        else:
            return value

        rel_offset = self.cursor
        self.cursor = rel_offset + len(value)
        self.segments.append((rel_offset, value))
        self.extended = True

        return {_BIN: {"kind": "bytes", "offset": rel_offset, "nbytes": len(value)}}

    # -- tree -----------------------------------------------------------

    def build(self, node: Any) -> Any:
        if type(node) in _PLAIN_SCALARS:
            return node

        if isinstance(node, np.ndarray):
            return self._array(node)

        if isinstance(node, dict):
            build = self.build
            result = {key: build(value) for key, value in node.items()}

            if len(result) == 1:
                (key,) = result
                if _needs_escape(key):
                    self.extended = True
                    return {_ESCAPE + key: result[key]}

            return result

        if isinstance(node, (list, tuple, set, frozenset)):
            build = self.build
            return [build(value) for value in node]

        if isinstance(node, BlobRef):
            return {_BLOB: node.to_manifest()}

        if isinstance(node, (bytes, bytearray)):
            return self._bytes(node)

        return node


class _Plan:
    """Exact layout of one encoded container."""

    __slots__ = (
        "data_bytes",
        "data_start",
        "flags",
        "frame_size",
        "metadata_json",
        "segments",
        "total_size",
    )

    def __init__(self, builder: _ManifestBuilder, manifest: Any, frame_size: int) -> None:
        self.metadata_json: bytes = to_json(manifest, fallback=_json_fallback)
        self.segments = builder.segments
        self.flags = FLAG_EXTENDED_MANIFEST if builder.extended else 0
        self.frame_size = frame_size
        self.data_bytes = builder.cursor
        self.data_start = align_up(HEADER_SIZE + len(self.metadata_json), frame_size)
        self.total_size = align_up(
            max(self.data_start + self.data_bytes, frame_size),
            frame_size,
        )


def _json_fallback(value: Any) -> Any:
    """Handle small NumPy scalar values that occur in metadata."""
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"value is not JSON serializable: {type(value).__name__}")


def _dump_model(
    model: BaseModel,
    *,
    blob_store: BlobStore | None,
    externalize_min_bytes: int | None,
    frame_size: int,
) -> tuple[Any, int, Any]:
    """Validate the encode arguments and dump the model to a Python tree."""
    if not isinstance(model, BaseModel):
        raise TypeError("expected a Pydantic BaseModel instance")

    if frame_size is not FRAME_SIZE:  # the default needs no validation
        require_frame_size(frame_size)

    if externalize_min_bytes is not None:
        if isinstance(externalize_min_bytes, bool) or not isinstance(
            externalize_min_bytes, int
        ):
            raise TypeError("externalize_min_bytes must be an integer or None")
        if externalize_min_bytes < 0:
            raise ValueError("externalize_min_bytes must be >= 0")
        if blob_store is None:
            raise ValueError(
                "blob_store is required when externalize_min_bytes is provided"
            )

    schema_id, schema_version = schema_identity(type(model))
    return schema_id, schema_version, model.model_dump(mode="python")


def _build_plan(
    python_tree: Any,
    *,
    blob_store: BlobStore | None,
    externalize_min_bytes: int | None,
    frame_size: int,
    dry_run: bool,
) -> tuple[_ManifestBuilder, _Plan]:
    builder = _ManifestBuilder(
        blob_store=blob_store,
        externalize_min_bytes=externalize_min_bytes,
        dry_run=dry_run,
    )

    try:
        plan = _Plan(builder, builder.build(python_tree), frame_size)
    except BaseException:
        _rollback(blob_store, builder.created)
        raise

    return builder, plan


def _rollback(blob_store: BlobStore | None, created: list[BlobRef]) -> None:
    """Best-effort removal of objects created by an encode that failed."""
    if blob_store is None:
        return
    for ref in created:
        with suppress(Exception):
            blob_store.delete(ref)


def _check_output(out: np.ndarray, total_size: int | None) -> None:
    """Validate a caller-provided output buffer without writing to it.

    ``total_size=None`` checks only the buffer's type and layout.
    """
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
    if total_size is not None and out.nbytes < total_size:
        raise BufferTooSmallError(
            f"output buffer too small: need {total_size:,} bytes, got {out.nbytes:,}"
        )


def _write_container(
    plan: _Plan,
    out: np.ndarray | None,
    schema_id: Any,
    schema_version: int,
    *,
    out_checked: bool,
) -> np.ndarray:
    """Allocate or reuse the output buffer and write the whole container."""
    total_size = plan.total_size
    data_start = plan.data_start

    if out is None:
        binary = np.empty(total_size, dtype=np.uint8)
    else:
        if not out_checked:
            _check_output(out, total_size)
        binary = out[:total_size]

    # Deterministic header/metadata frames and metadata padding.
    binary[:data_start] = 0

    metadata_bytes = len(plan.metadata_json)

    HEADER.pack_into(
        binary,
        0,
        MAGIC,
        FORMAT_VERSION,
        plan.flags,
        plan.frame_size,
        schema_id.bytes,
        schema_version,
        metadata_bytes,
        data_start,
        plan.data_bytes,
        total_size,
    )

    binary[HEADER_SIZE : HEADER_SIZE + metadata_bytes] = np.frombuffer(
        plan.metadata_json,
        dtype=np.uint8,
    )

    previous_end = 0

    for rel_offset, payload in plan.segments:
        # Zero tiny alignment gaps if the caller reused an old buffer.
        if rel_offset > previous_end:
            binary[data_start + previous_end : data_start + rel_offset] = 0

        absolute_offset = data_start + rel_offset

        if isinstance(payload, np.ndarray):
            nbytes = payload.nbytes
            if nbytes:
                _write_array(binary, absolute_offset, payload)
        else:
            nbytes = len(payload)
            if nbytes:
                binary[absolute_offset : absolute_offset + nbytes] = np.frombuffer(
                    payload,
                    dtype=np.uint8,
                )

        previous_end = rel_offset + nbytes

    # Zero frame padding after the meaningful data region.
    final_data_end = data_start + plan.data_bytes
    if final_data_end < total_size:
        binary[final_data_end:total_size] = 0

    return binary


def _write_array(binary: np.ndarray, absolute_offset: int, source: np.ndarray) -> None:
    """Copy one ndarray payload into the container in C order, exactly once."""
    # Drop ndarray subclasses (matrix, memmap, ...) without copying.
    arr = np.asarray(source)

    if arr.flags.c_contiguous:
        # One memcpy. reshape(-1) first so 0-d arrays can be viewed as bytes.
        binary[absolute_offset : absolute_offset + arr.nbytes] = arr.reshape(-1).view(
            np.uint8
        )
        return

    # Strided / Fortran-ordered input: let NumPy iterate the source straight
    # into the destination instead of materialising a contiguous temporary,
    # which for multi-GiB arrays would double both the copy and peak memory.
    destination = np.frombuffer(
        binary,
        dtype=arr.dtype,
        count=arr.size,
        offset=absolute_offset,
    ).reshape(arr.shape)
    np.copyto(destination, arr, casting="no")


def encoded_size(
    model: BaseModel,
    *,
    blob_store: BlobStore | None = None,
    externalize_min_bytes: int | None = None,
    frame_size: int = FRAME_SIZE,
) -> int:
    """Return the number of bytes :func:`encode` will produce.

    The ndarray payloads are inspected but not copied, and nothing is written
    to ``blob_store``. This is useful when the destination storage must be
    loaned or reserved first, such as an iceoryx2 dynamic shared-memory
    sample.

    The result is exact for self-contained containers, and for externalized
    arrays whenever the store implements ``preview_ref`` (``VineyardStore``
    does). For other stores NPB assumes 32-character object IDs; a store with
    longer IDs can then need one more frame than reported, in which case
    ``encode(..., out=...)`` raises :class:`BufferTooSmallError`.
    """
    _schema_id, _schema_version, python_tree = _dump_model(
        model,
        blob_store=blob_store,
        externalize_min_bytes=externalize_min_bytes,
        frame_size=frame_size,
    )
    _builder, plan = _build_plan(
        python_tree,
        blob_store=blob_store,
        externalize_min_bytes=externalize_min_bytes,
        frame_size=frame_size,
        dry_run=True,
    )
    return plan.total_size


def encode(
    model: BaseModel,
    *,
    out: np.ndarray | None = None,
    blob_store: BlobStore | None = None,
    externalize_min_bytes: int | None = None,
    frame_size: int = FRAME_SIZE,
) -> np.ndarray:
    """Encode a Pydantic model into one frame-aligned ``np.uint8`` array.

    Parameters
    ----------
    model:
        Top-level Pydantic model decorated with :func:`npb.binary_schema`.
    out:
        Optional writable, C-contiguous, 1-D ``np.uint8`` array to reuse.
        If larger than required, the returned value is a view covering only
        the encoded region.
    blob_store:
        Optional immutable external blob store, such as ``VineyardStore``.
    externalize_min_bytes:
        When provided, ndarray leaves at least this large are stored in
        ``blob_store`` and represented by tiny references instead of being
        copied into the NPB container. Use ``0`` to externalize all arrays.
    frame_size:
        Size of the fixed frames the container is padded to. Must be a power
        of two between ``MIN_FRAME_SIZE`` (64) and 1 GiB. The default,
        ``FRAME_SIZE`` (64 KiB), is the only size NPB <= 0.3 can read; a small
        frame keeps control messages that carry only references compact.

    Returns
    -------
    numpy.ndarray
        Self-contained, frame-aligned encoded data.

    Notes
    -----
    Everything that can fail without touching the store (unsupported dtypes,
    metadata that is not JSON serializable, an ``out`` buffer that is too
    small) is checked before any array is written to ``blob_store``.
    """
    schema_id, schema_version, python_tree = _dump_model(
        model,
        blob_store=blob_store,
        externalize_min_bytes=externalize_min_bytes,
        frame_size=frame_size,
    )

    # Pre-flight pass. Without externalization this already is the final plan.
    builder, plan = _build_plan(
        python_tree,
        blob_store=blob_store,
        externalize_min_bytes=externalize_min_bytes,
        frame_size=frame_size,
        dry_run=True,
    )

    # The dry-run layout is exact unless arrays are about to be stored in a
    # store that cannot preview its references. In that one case only the
    # buffer's type is checked here and its capacity is checked against the
    # real layout, rather than risk rejecting a buffer that would have been
    # big enough.
    exact = not builder.pending_puts or hasattr(blob_store, "preview_ref")

    if out is not None:
        _check_output(out, plan.total_size if exact else None)

    if builder.pending_puts:
        builder, plan = _build_plan(
            python_tree,
            blob_store=blob_store,
            externalize_min_bytes=externalize_min_bytes,
            frame_size=frame_size,
            dry_run=False,
        )

    try:
        return _write_container(plan, out, schema_id, schema_version, out_checked=exact)
    except BaseException:
        _rollback(blob_store, builder.created)
        raise


def _index(value: object) -> int:
    """Return a manifest integer, rejecting bools, floats and strings."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("expected an integer")
    return value


class _Restorer:
    """Replace manifest references with zero-copy views into ``binary``."""

    __slots__ = ("binary", "blob_store", "data_bytes", "data_start", "extended", "unresolved")

    def __init__(
        self,
        binary: np.ndarray,
        info: BinaryInfo,
        blob_store: BlobStore | None,
    ) -> None:
        self.binary = binary
        self.data_start = info.data_start
        self.data_bytes = info.data_bytes
        self.blob_store = blob_store
        self.extended = bool(info.flags & FLAG_EXTENDED_MANIFEST)
        #: True if at least one ``$blob`` reference was left as a BlobRef.
        self.unresolved = False

    def _span(self, ref: dict[str, Any]) -> tuple[int, int]:
        """Validate and return the absolute (offset, nbytes) of a ``$bin`` leaf."""
        try:
            rel_offset = _index(ref["offset"])
            nbytes = _index(ref["nbytes"])
        except (KeyError, TypeError) as exc:
            raise FormatError("invalid binary reference") from exc

        if rel_offset < 0 or nbytes < 0:
            raise FormatError("negative binary offset or size")
        if rel_offset + nbytes > self.data_bytes:
            raise FormatError("binary reference exceeds the DATA section")

        absolute_offset = self.data_start + rel_offset
        if absolute_offset + nbytes > self.binary.nbytes:
            raise FormatError("binary reference exceeds the encoded array")

        return absolute_offset, nbytes

    def _ndarray(self, ref: dict[str, Any]) -> np.ndarray:
        try:
            dtype_str = ref["dtype"]
            if not isinstance(dtype_str, str):
                raise TypeError("dtype must be a string")
            dtype = np.dtype(dtype_str)
            raw_shape = ref["shape"]
            if not isinstance(raw_shape, list):
                raise TypeError("shape must be an array")
            shape = tuple(_index(dim) for dim in raw_shape)
        except (KeyError, TypeError, ValueError, SyntaxError) as exc:
            raise FormatError("invalid ndarray reference") from exc

        if dtype.hasobject or dtype.fields is not None or dtype.subdtype is not None:
            raise FormatError("unsupported ndarray dtype in binary reference")
        if any(dim < 0 for dim in shape):
            raise FormatError("negative ndarray dimension")

        absolute_offset, nbytes = self._span(ref)

        if dtype.itemsize <= 0 or nbytes % dtype.itemsize != 0:
            raise FormatError("invalid ndarray byte size")

        expected_items = 1
        for dim in shape:
            expected_items *= dim

        if expected_items * dtype.itemsize != nbytes:
            raise FormatError("ndarray shape does not match its byte size")

        try:
            return np.frombuffer(
                self.binary,
                dtype=dtype,
                count=expected_items,
                offset=absolute_offset,
            ).reshape(shape)
        except (ValueError, TypeError) as exc:
            raise FormatError("invalid ndarray reference") from exc

    def _blob(self, value: object) -> Any:
        try:
            ref = BlobRef.from_manifest(value)
        except (TypeError, ValueError) as exc:
            raise FormatError("invalid external blob reference") from exc

        blob_store = self.blob_store
        if blob_store is None:
            self.unresolved = True
            return ref

        if blob_store.kind != ref.store:
            raise FormatError(
                f"blob reference requires store {ref.store!r}, "
                f"but decoder received {blob_store.kind!r}"
            )

        return blob_store.get_array(ref)

    def restore(self, node: Any) -> Any:
        if isinstance(node, dict):
            if len(node) == 1:
                (key,) = node

                if key == _BLOB:
                    return self._blob(node[key])

                if key == _BIN:
                    ref = node[key]
                    kind = ref.get("kind") if isinstance(ref, dict) else None

                    if kind == "ndarray":
                        return self._ndarray(ref)

                    if self.extended:
                        # In an extended manifest user dictionaries are
                        # escaped, so an unescaped "$bin" is always a leaf.
                        if kind == "bytes":
                            offset, nbytes = self._span(ref)
                            return self.binary[offset : offset + nbytes].tobytes()
                        raise FormatError(f"unsupported binary leaf kind: {kind!r}")

                elif self.extended and key.startswith(_ESCAPED_PREFIX):
                    return {key[1:]: self.restore(node[key])}

            restore = self.restore
            return {key: restore(value) for key, value in node.items()}

        if isinstance(node, list):
            restore = self.restore
            return [restore(value) for value in node]

        return node


def _decode_tree(
    binary: np.ndarray,
    info: BinaryInfo,
    *,
    blob_store: BlobStore | None,
) -> tuple[dict[str, Any], bool]:
    """Decode metadata and reconstruct binary leaves.

    Returns the tree and whether it still contains unresolved ``BlobRef``s.
    """
    metadata_start = HEADER_SIZE
    metadata_end = metadata_start + info.metadata_bytes

    # pydantic-core currently accepts bytes/bytearray/str, so this is the only
    # small copy during decode. ndarray payloads remain zero-copy.
    metadata_json = binary[metadata_start:metadata_end].tobytes()

    try:
        manifest = from_json(metadata_json)
    except ValueError as exc:
        raise FormatError("invalid JSON metadata") from exc

    restorer = _Restorer(binary, info, blob_store)

    try:
        tree = restorer.restore(manifest)
    except RecursionError as exc:
        raise FormatError("JSON metadata is nested too deeply") from exc

    if not isinstance(tree, dict):
        raise FormatError("top-level NPB value must be an object")

    return tree, restorer.unresolved


def peek(binary: BinaryLike) -> BinaryInfo:
    """Inspect the fixed NPB header without decoding metadata or arrays."""
    return parse_header(as_binary_array(binary))


def decode_auto(
    binary: BinaryLike,
    *,
    blob_store: BlobStore | None = None,
) -> dict[str, Any]:
    """Decode to plain Python containers with zero-copy ``np.ndarray`` leaves.

    ``binary`` may be a 1-D ``np.uint8`` array or any ``bytes``-like object;
    the latter is wrapped without copying and kept alive by the decoded
    arrays.

    This function never imports or executes a class named by the encoded data.
    """
    binary = as_binary_array(binary)
    info = parse_header(binary)
    tree, _unresolved = _decode_tree(binary, info, blob_store=blob_store)
    return tree


def decode(
    model_type: type[T],
    binary: BinaryLike,
    *,
    blob_store: BlobStore | None = None,
    strict: bool | None = None,
) -> T:
    """Decode and validate using an explicitly supplied Pydantic model class.

    Parameters
    ----------
    model_type:
        Pydantic model class decorated with :func:`npb.binary_schema`.
    binary:
        A 1-D ``np.uint8`` array or a ``bytes``-like object.
    blob_store:
        Store used to resolve external ``$blob`` references.
    strict:
        Forwarded to ``model_validate``. Metadata travels as JSON, so values
        such as ``datetime`` and ``tuple`` arrive as strings and lists; a
        model configured with ``strict=True`` rejects those unless you pass
        ``strict=False`` here. ``None`` keeps the model's own setting.
    """
    if not isinstance(model_type, type) or not issubclass(model_type, BaseModel):
        raise TypeError("model_type must be a Pydantic BaseModel class")

    expected_id, expected_version = schema_identity(model_type)
    binary = as_binary_array(binary)
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

    tree, unresolved = _decode_tree(binary, info, blob_store=blob_store)

    if unresolved:
        raise BlobStoreRequiredError(
            "this NPB message contains external blob references; "
            "pass the matching blob_store=... to decode()"
        )

    return model_type.model_validate(tree, strict=strict)
