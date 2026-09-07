"""NPB exception hierarchy."""


class NPBError(Exception):
    """Base exception for NPB."""


class FormatError(NPBError, ValueError):
    """Raised when an encoded NPB container is malformed or unsupported."""


class SchemaMismatchError(NPBError, ValueError):
    """Raised when typed decoding is attempted with the wrong schema."""


class BufferTooSmallError(NPBError, ValueError):
    """Raised when a caller-provided output buffer is too small."""


class BlobStoreError(NPBError, RuntimeError):
    """Raised when an external blob store operation fails."""


class BlobStoreRequiredError(BlobStoreError):
    """Raised when typed decoding requires an external blob store."""
