"""Pydantic integration and stable schema identities."""

from __future__ import annotations

from collections.abc import Callable
from typing import ClassVar, TypeVar
from uuid import UUID, uuid5

from pydantic import BaseModel, ConfigDict

_SCHEMA_NAMESPACE = UUID("5e066b26-0f8a-5b6a-a764-6b2c39a83fb9")

ModelT = TypeVar("ModelT", bound=type[BaseModel])


class BinaryModel(BaseModel):
    """Convenience Pydantic base class that allows NumPy array fields."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    __binary_schema_name__: ClassVar[str | None] = None
    __binary_schema_id__: ClassVar[UUID | None] = None
    __binary_schema_version__: ClassVar[int | None] = None


def binary_schema(name: str, *, version: int = 1) -> Callable[[ModelT], ModelT]:
    """Attach a stable binary schema identity to a top-level Pydantic model.

    Parameters
    ----------
    name:
        Stable application-level name such as ``"com.example.capture"``.
        The name is deterministically converted to a UUID.
    version:
        Non-negative application schema version.
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError("schema name must be a non-empty string")
    if not isinstance(version, int) or isinstance(version, bool) or version < 0:
        raise ValueError("schema version must be a non-negative integer")

    normalized_name = name.strip()
    schema_id = uuid5(_SCHEMA_NAMESPACE, normalized_name)

    def decorate(model_type: ModelT) -> ModelT:
        if not issubclass(model_type, BaseModel):
            raise TypeError("@binary_schema can only decorate Pydantic BaseModel classes")

        model_type.__binary_schema_name__ = normalized_name
        model_type.__binary_schema_id__ = schema_id
        model_type.__binary_schema_version__ = version
        return model_type

    return decorate


def schema_identity(model_type: type[BaseModel]) -> tuple[UUID, int]:
    """Return the declared schema UUID and version for a model."""
    schema_id = getattr(model_type, "__binary_schema_id__", None)
    schema_version = getattr(model_type, "__binary_schema_version__", None)

    if not isinstance(schema_id, UUID) or not isinstance(schema_version, int):
        raise TypeError(
            f"{model_type.__name__} has no binary schema identity; "
            "decorate the top-level model with @binary_schema(...)"
        )

    if schema_version < 0:
        raise TypeError("__binary_schema_version__ must be non-negative")

    return schema_id, schema_version
