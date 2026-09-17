"""Fixed-layout NPB structs for low-overhead control/RPC messages.

This module is intentionally independent of Pydantic on the binary hot path.
Pydantic models are generated lazily only for JSON/OpenAPI/debug use.
"""

from __future__ import annotations

from ipaddress import IPv4Address
import struct
from typing import Any, ClassVar, Literal, Optional

from pydantic import BaseModel, Field as PydanticField, create_model


_MISSING = object()


class FixedField:
    """Base class for one logical field in a :class:`FixedStruct`."""

    size: int = 0
    python_type: Any = Any

    def __init__(self, default: Any = _MISSING) -> None:
        self.default = default
        self.name = ""
        self.offset = -1

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name

    def __get__(self, obj: Any, owner: type | None = None) -> Any:
        if obj is None:
            return self
        if self.name in obj.__dict__:
            return obj.__dict__[self.name]
        if self.default is not _MISSING:
            return self.default
        raise AttributeError(self.name)

    def __set__(self, obj: Any, value: Any) -> None:
        obj.__dict__[self.name] = value

    def validate_default(self) -> None:
        if self.default is not _MISSING:
            self.validate(self.default)

    def validate(self, value: Any) -> None:
        pass

    def pack_into(self, buffer: Any, base_offset: int, value: Any) -> None:
        raise NotImplementedError

    def unpack_from(self, buffer: Any, base_offset: int) -> Any:
        raise NotImplementedError

    def pydantic_annotation(self) -> Any:
        return self.python_type

    def pydantic_field(self) -> Any:
        if self.default is _MISSING:
            return ...
        return self.default

    def layout_rows(self, prefix: str = "") -> list[dict[str, Any]]:
        return [
            {
                "field": f"{prefix}{self.name}",
                "offset": self.offset,
                "size": self.size,
                "encoding": type(self).__name__,
            }
        ]


class _StructField(FixedField):
    fmt: str
    min_value: int | float | None = None
    max_value: int | float | None = None

    def __init__(self, fmt: str, python_type: Any, default: Any = _MISSING) -> None:
        super().__init__(default)
        self._struct = struct.Struct("<" + fmt)
        self.size = self._struct.size
        self.python_type = python_type

    def pack_into(self, buffer: Any, base_offset: int, value: Any) -> None:
        self.validate(value)
        self._struct.pack_into(buffer, base_offset + self.offset, value)

    def unpack_from(self, buffer: Any, base_offset: int) -> Any:
        return self._struct.unpack_from(buffer, base_offset + self.offset)[0]

    def pydantic_field(self) -> Any:
        default = ... if self.default is _MISSING else self.default
        kwargs: dict[str, Any] = {}
        if self.min_value is not None:
            kwargs["ge"] = self.min_value
        if self.max_value is not None:
            kwargs["le"] = self.max_value
        return PydanticField(default=default, **kwargs)


class UInt8(_StructField):
    min_value = 0
    max_value = 0xFF
    def __init__(self, default: int = 0) -> None:
        super().__init__("B", int, default)

    def validate(self, value: Any) -> None:
        if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 0xFF:
            raise ValueError(f"{self.name} must be uint8")


class UInt16(_StructField):
    min_value = 0
    max_value = 0xFFFF
    def __init__(self, default: int = 0) -> None:
        super().__init__("H", int, default)

    def validate(self, value: Any) -> None:
        if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 0xFFFF:
            raise ValueError(f"{self.name} must be uint16")


class UInt32(_StructField):
    min_value = 0
    max_value = 0xFFFFFFFF
    def __init__(self, default: int = 0) -> None:
        super().__init__("I", int, default)

    def validate(self, value: Any) -> None:
        if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 0xFFFFFFFF:
            raise ValueError(f"{self.name} must be uint32")


class UInt64(_StructField):
    min_value = 0
    max_value = 0xFFFFFFFFFFFFFFFF
    def __init__(self, default: int = 0) -> None:
        super().__init__("Q", int, default)

    def validate(self, value: Any) -> None:
        if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 0xFFFFFFFFFFFFFFFF:
            raise ValueError(f"{self.name} must be uint64")


class Int8(_StructField):
    min_value = -(1 << 7)
    max_value = (1 << 7) - 1
    def __init__(self, default: int = 0) -> None:
        super().__init__("b", int, default)


class Int16(_StructField):
    min_value = -(1 << 15)
    max_value = (1 << 15) - 1
    def __init__(self, default: int = 0) -> None:
        super().__init__("h", int, default)


class Int32(_StructField):
    min_value = -(1 << 31)
    max_value = (1 << 31) - 1
    def __init__(self, default: int = 0) -> None:
        super().__init__("i", int, default)


class Int64(_StructField):
    min_value = -(1 << 63)
    max_value = (1 << 63) - 1
    def __init__(self, default: int = 0) -> None:
        super().__init__("q", int, default)


class Float32(_StructField):
    def __init__(self, default: float = 0.0) -> None:
        super().__init__("f", float, default)

    def pack_into(self, buffer: Any, base_offset: int, value: Any) -> None:
        self._struct.pack_into(buffer, base_offset + self.offset, float(value))


class Float64(_StructField):
    def __init__(self, default: float = 0.0) -> None:
        super().__init__("d", float, default)

    def pack_into(self, buffer: Any, base_offset: int, value: Any) -> None:
        self._struct.pack_into(buffer, base_offset + self.offset, float(value))


class Bool8(_StructField):
    def __init__(self, default: bool = False) -> None:
        super().__init__("B", bool, default)

    def validate(self, value: Any) -> None:
        if not isinstance(value, bool):
            raise TypeError(f"{self.name} must be bool")

    def pack_into(self, buffer: Any, base_offset: int, value: Any) -> None:
        self.validate(value)
        self._struct.pack_into(buffer, base_offset + self.offset, int(value))

    def unpack_from(self, buffer: Any, base_offset: int) -> bool:
        return bool(super().unpack_from(buffer, base_offset))


class FixedStr(FixedField):
    python_type = str

    def __init__(self, size: int, default: Any = _MISSING, *, encoding: str = "utf-8") -> None:
        if size <= 0:
            raise ValueError("FixedStr size must be > 0")
        super().__init__(default)
        self.size = size
        self.encoding = encoding
        self._struct = struct.Struct(f"<{size}s")

    def validate(self, value: Any) -> None:
        if not isinstance(value, str):
            raise TypeError(f"{self.name} must be str")
        raw = value.encode(self.encoding)
        if len(raw) > self.size:
            raise ValueError(f"{self.name} exceeds {self.size} encoded bytes")

    def pack_into(self, buffer: Any, base_offset: int, value: Any) -> None:
        self.validate(value)
        self._struct.pack_into(buffer, base_offset + self.offset, value.encode(self.encoding))

    def unpack_from(self, buffer: Any, base_offset: int) -> str:
        raw = self._struct.unpack_from(buffer, base_offset + self.offset)[0]
        return raw.rstrip(b"\x00").decode(self.encoding)

    def pydantic_field(self) -> Any:
        default = ... if self.default is _MISSING else self.default
        return PydanticField(default=default, max_length=self.size)

    def layout_rows(self, prefix: str = "") -> list[dict[str, Any]]:
        rows = super().layout_rows(prefix)
        rows[0]["encoding"] = f"utf8[{self.size}]"
        return rows


class IPv4(FixedField):
    size = 4
    python_type = str

    def __init__(self, default: Any = _MISSING) -> None:
        super().__init__(default)
        self._struct = struct.Struct("<4s")

    def validate(self, value: Any) -> None:
        IPv4Address(value)

    def pack_into(self, buffer: Any, base_offset: int, value: Any) -> None:
        packed = IPv4Address(value).packed
        self._struct.pack_into(buffer, base_offset + self.offset, packed)

    def unpack_from(self, buffer: Any, base_offset: int) -> str:
        raw = self._struct.unpack_from(buffer, base_offset + self.offset)[0]
        return str(IPv4Address(raw))

    def pydantic_field(self) -> Any:
        default = ... if self.default is _MISSING else self.default
        return PydanticField(default=default, json_schema_extra={"format": "ipv4"})


class Enum8(FixedField):
    size = 1
    python_type = str

    def __init__(self, *values: str, default: Any = _MISSING) -> None:
        if not values:
            raise ValueError("Enum8 requires at least one value")
        if len(values) > 256:
            raise ValueError("Enum8 supports at most 256 values")
        if len(set(values)) != len(values):
            raise ValueError("Enum8 values must be unique")
        super().__init__(default)
        self.values = tuple(values)
        self._to_int = {value: i for i, value in enumerate(self.values)}
        self._struct = struct.Struct("<B")

    def validate(self, value: Any) -> None:
        if value not in self._to_int:
            raise ValueError(f"{self.name} must be one of {self.values!r}")

    def pack_into(self, buffer: Any, base_offset: int, value: Any) -> None:
        self.validate(value)
        self._struct.pack_into(buffer, base_offset + self.offset, self._to_int[value])

    def unpack_from(self, buffer: Any, base_offset: int) -> str:
        index = self._struct.unpack_from(buffer, base_offset + self.offset)[0]
        try:
            return self.values[index]
        except IndexError as exc:
            raise ValueError(f"invalid {self.name} enum value {index}") from exc

    def pydantic_annotation(self) -> Any:
        return Literal.__getitem__(self.values)

    def layout_rows(self, prefix: str = "") -> list[dict[str, Any]]:
        rows = super().layout_rows(prefix)
        rows[0]["encoding"] = "enum8:" + "|".join(self.values)
        return rows


class FixedPointU16(FixedField):
    size = 2
    python_type = float

    def __init__(self, scale: float, default: float = 0.0) -> None:
        if scale <= 0:
            raise ValueError("scale must be > 0")
        super().__init__(default)
        self.scale = float(scale)
        self._struct = struct.Struct("<H")

    def validate(self, value: Any) -> None:
        raw = round(float(value) / self.scale)
        if not 0 <= raw <= 0xFFFF:
            raise ValueError(f"{self.name} is out of uint16 fixed-point range")

    def pack_into(self, buffer: Any, base_offset: int, value: Any) -> None:
        self.validate(value)
        raw = round(float(value) / self.scale)
        self._struct.pack_into(buffer, base_offset + self.offset, raw)

    def unpack_from(self, buffer: Any, base_offset: int) -> float:
        raw = self._struct.unpack_from(buffer, base_offset + self.offset)[0]
        return raw * self.scale

    def pydantic_field(self) -> Any:
        default = ... if self.default is _MISSING else self.default
        return PydanticField(default=default, ge=0.0, le=0xFFFF * self.scale)

    def layout_rows(self, prefix: str = "") -> list[dict[str, Any]]:
        rows = super().layout_rows(prefix)
        rows[0]["encoding"] = f"u16*{self.scale:g}"
        return rows


class Bit(FixedField):
    """Boolean stored in a shared one-byte bit group."""

    size = 0
    python_type = bool

    def __init__(self, bit: int, default: bool = False, *, group: str = "flags") -> None:
        if not 0 <= bit <= 7:
            raise ValueError("Bit index must be 0..7")
        super().__init__(default)
        self.bit = bit
        self.group = group
        self.mask = 1 << bit

    def validate(self, value: Any) -> None:
        if not isinstance(value, bool):
            raise TypeError(f"{self.name} must be bool")

    def pack_into(self, buffer: Any, base_offset: int, value: Any) -> None:
        self.validate(value)
        pos = base_offset + self.offset
        current = buffer[pos]
        if value:
            buffer[pos] = current | self.mask
        else:
            buffer[pos] = current & ~self.mask

    def unpack_from(self, buffer: Any, base_offset: int) -> bool:
        return bool(buffer[base_offset + self.offset] & self.mask)

    def layout_rows(self, prefix: str = "") -> list[dict[str, Any]]:
        return [
            {
                "field": f"{prefix}{self.name}",
                "offset": self.offset,
                "size": "bit",
                "encoding": f"{self.group}[{self.bit}]",
            }
        ]


class Nested(FixedField):
    """Fixed-size nested struct, physically inlined with zero wire overhead."""

    def __init__(self, struct_type: type[FixedStruct], default: Any = _MISSING) -> None:
        super().__init__(default)
        self.struct_type = struct_type
        self.size = struct_type.byte_size()
        self.python_type = struct_type

    def validate(self, value: Any) -> None:
        if not isinstance(value, self.struct_type):
            raise TypeError(f"{self.name} must be {self.struct_type.__name__}")

    def pack_into(self, buffer: Any, base_offset: int, value: Any) -> None:
        self.validate(value)
        value.pack_into(buffer, base_offset + self.offset)

    def unpack_from(self, buffer: Any, base_offset: int) -> Any:
        return self.struct_type.unpack_from(buffer, base_offset + self.offset)

    def pydantic_annotation(self) -> Any:
        return self.struct_type.pydantic_model()

    def pydantic_field(self) -> Any:
        if self.default is _MISSING:
            return ...
        if isinstance(self.default, FixedStruct):
            return self.default.to_dict()
        return self.default

    def layout_rows(self, prefix: str = "") -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for row in self.struct_type.layout():
            row = dict(row)
            row["field"] = f"{prefix}{self.name}.{row['field']}"
            row["offset"] = self.offset + int(row["offset"])
            rows.append(row)
        return rows


class OptionalNested(Nested):
    """Optional fixed nested struct: one shared presence bit + inline payload."""

    def __init__(
        self,
        struct_type: type[FixedStruct],
        *,
        bit: int,
        group: str = "presence",
        default: Any = None,
    ) -> None:
        super().__init__(struct_type, default)
        self.presence_bit = bit
        self.presence_group = group
        self.presence_offset = -1
        self.presence_mask = 1 << bit
        if not 0 <= bit <= 7:
            raise ValueError("presence bit must be 0..7")

    def validate(self, value: Any) -> None:
        if value is not None and not isinstance(value, self.struct_type):
            raise TypeError(f"{self.name} must be {self.struct_type.__name__} or None")

    def pack_into(self, buffer: Any, base_offset: int, value: Any) -> None:
        self.validate(value)
        p = base_offset + self.presence_offset
        if value is None:
            buffer[p] = buffer[p] & ~self.presence_mask
            # payload is intentionally left zeroed by FixedStruct.pack_into().
            return
        buffer[p] = buffer[p] | self.presence_mask
        value.pack_into(buffer, base_offset + self.offset)

    def unpack_from(self, buffer: Any, base_offset: int) -> Any:
        present = bool(buffer[base_offset + self.presence_offset] & self.presence_mask)
        if not present:
            return None
        return self.struct_type.unpack_from(buffer, base_offset + self.offset)

    def pydantic_annotation(self) -> Any:
        return Optional[self.struct_type.pydantic_model()]

    def layout_rows(self, prefix: str = "") -> list[dict[str, Any]]:
        rows = super().layout_rows(prefix)
        rows.insert(
            0,
            {
                "field": f"{prefix}{self.name}?",
                "offset": self.presence_offset,
                "size": "bit",
                "encoding": f"{self.presence_group}[{self.presence_bit}]",
            },
        )
        return rows


class FixedArray(FixedField):
    """Fixed-count array of one primitive field or fixed nested struct."""

    def __init__(self, element: FixedField | type[FixedStruct], count: int, default: Any = _MISSING) -> None:
        if count <= 0:
            raise ValueError("FixedArray count must be > 0")
        super().__init__(default)
        self.count = count
        if isinstance(element, type) and issubclass(element, FixedStruct):
            self.element = Nested(element)
            self.element_type = element
        elif isinstance(element, FixedField):
            if isinstance(element, (Bit, OptionalNested)):
                raise TypeError("Bit/OptionalNested are not valid FixedArray elements")
            self.element = element
            self.element_type = element.pydantic_annotation()
        else:
            raise TypeError("element must be FixedField or FixedStruct type")
        # Array element offsets are always relative to each array slot. Keep the
        # descriptor immutable after class creation so concurrent pack/unpack calls
        # cannot race by mutating shared schema state.
        self.element.offset = 0
        self.size = self.element.size * count
        self.python_type = list[self.element_type]  # type: ignore[index]

    def validate(self, value: Any) -> None:
        if not isinstance(value, (list, tuple)) or len(value) != self.count:
            raise ValueError(f"{self.name} must contain exactly {self.count} elements")

    def pack_into(self, buffer: Any, base_offset: int, value: Any) -> None:
        self.validate(value)
        for i, item in enumerate(value):
            slot = base_offset + self.offset + i * self.element.size
            self.element.pack_into(buffer, slot, item)

    def unpack_from(self, buffer: Any, base_offset: int) -> list[Any]:
        result = []
        for i in range(self.count):
            slot = base_offset + self.offset + i * self.element.size
            result.append(self.element.unpack_from(buffer, slot))
        return result

    def pydantic_annotation(self) -> Any:
        return list[self.element.pydantic_annotation()]  # type: ignore[index]

    def pydantic_field(self) -> Any:
        default = ... if self.default is _MISSING else self.default
        return PydanticField(default=default, min_length=self.count, max_length=self.count)

    def layout_rows(self, prefix: str = "") -> list[dict[str, Any]]:
        return [
            {
                "field": f"{prefix}{self.name}",
                "offset": self.offset,
                "size": self.size,
                "encoding": f"fixed-array[{self.count}]",
            }
        ]


class FixedStructMeta(type):
    def __new__(mcls, name: str, bases: tuple[type, ...], namespace: dict[str, Any]):
        cls = super().__new__(mcls, name, bases, namespace)

        fields: list[tuple[str, FixedField]] = []
        for base in bases:
            fields.extend(getattr(base, "__fixed_fields__", ()))
        fields.extend((key, value) for key, value in namespace.items() if isinstance(value, FixedField))

        cursor = 0
        bit_groups: dict[str, int] = {}
        used_bits: dict[str, set[int]] = {}

        for field_name, field in fields:
            field.name = field_name

            if isinstance(field, Bit):
                if field.group not in bit_groups:
                    bit_groups[field.group] = cursor
                    used_bits[field.group] = set()
                    cursor += 1
                if field.bit in used_bits[field.group]:
                    raise TypeError(f"duplicate bit {field.bit} in group {field.group!r}")
                used_bits[field.group].add(field.bit)
                field.offset = bit_groups[field.group]
                field.validate_default()
                continue

            if isinstance(field, OptionalNested):
                if field.presence_group not in bit_groups:
                    bit_groups[field.presence_group] = cursor
                    used_bits[field.presence_group] = set()
                    cursor += 1
                if field.presence_bit in used_bits[field.presence_group]:
                    raise TypeError(
                        f"duplicate bit {field.presence_bit} in group {field.presence_group!r}"
                    )
                used_bits[field.presence_group].add(field.presence_bit)
                field.presence_offset = bit_groups[field.presence_group]
                field.offset = cursor
                cursor += field.size
                field.validate_default()
                continue

            field.offset = cursor
            cursor += field.size
            field.validate_default()

        cls.__fixed_fields__ = tuple(fields)
        cls.__fixed_size__ = cursor
        cls.__fixed_bit_groups__ = dict(bit_groups)
        cls.__pydantic_model_cache__ = None
        return cls


class FixedStruct(metaclass=FixedStructMeta):
    """Compact fixed-layout Python object with lazy Pydantic/OpenAPI projection."""

    __fixed_fields__: ClassVar[tuple[tuple[str, FixedField], ...]]
    __fixed_size__: ClassVar[int]
    __fixed_bit_groups__: ClassVar[dict[str, int]]
    __pydantic_model_cache__: ClassVar[type[BaseModel] | None]

    def __init__(self, **kwargs: Any) -> None:
        known = {name for name, _ in self.__fixed_fields__}
        unknown = set(kwargs) - known
        if unknown:
            raise TypeError(f"unknown fields: {sorted(unknown)!r}")

        for name, field in self.__fixed_fields__:
            if name in kwargs:
                value = kwargs[name]
            elif field.default is not _MISSING:
                value = field.default
            else:
                raise TypeError(f"missing required field: {name}")
            field.validate(value)
            setattr(self, name, value)

    @classmethod
    def byte_size(cls) -> int:
        return cls.__fixed_size__

    @classmethod
    def layout(cls) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        seen_groups: set[tuple[str, int]] = set()
        for _name, field in cls.__fixed_fields__:
            if isinstance(field, Bit):
                key = (field.group, field.offset)
                if key not in seen_groups:
                    seen_groups.add(key)
            rows.extend(field.layout_rows())
        return rows

    def pack_into(self, buffer: Any, offset: int = 0) -> Any:
        view = memoryview(buffer).cast("B")
        end = offset + self.byte_size()
        if offset < 0 or end > len(view):
            raise ValueError(f"buffer too small: need at least {end} bytes")
        # Deterministic output, and required for shared bit groups/absent optionals.
        view[offset:end] = b"\x00" * self.byte_size()
        for name, field in self.__fixed_fields__:
            field.pack_into(view, offset, getattr(self, name))
        return buffer

    def to_bytes(self) -> bytes:
        out = bytearray(self.byte_size())
        self.pack_into(out)
        return bytes(out)

    def to_bytearray(self) -> bytearray:
        out = bytearray(self.byte_size())
        self.pack_into(out)
        return out

    @classmethod
    def unpack_from(cls, buffer: Any, offset: int = 0) -> FixedStruct:
        view = memoryview(buffer).cast("B")
        end = offset + cls.byte_size()
        if offset < 0 or end > len(view):
            raise ValueError(f"buffer too small: need at least {end} bytes")
        values = {
            name: field.unpack_from(view, offset)
            for name, field in cls.__fixed_fields__
        }
        return cls(**values)

    @classmethod
    def from_bytes(cls, data: bytes | bytearray | memoryview) -> FixedStruct:
        if len(data) != cls.byte_size():
            raise ValueError(f"expected {cls.byte_size()} bytes, got {len(data)}")
        return cls.unpack_from(data)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name, field in self.__fixed_fields__:
            value = getattr(self, name)
            if isinstance(value, FixedStruct):
                result[name] = value.to_dict()
            elif isinstance(value, list):
                result[name] = [item.to_dict() if isinstance(item, FixedStruct) else item for item in value]
            else:
                result[name] = value
        return result

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> FixedStruct:
        kwargs: dict[str, Any] = {}
        for name, field in cls.__fixed_fields__:
            item = value[name] if name in value else field.default
            if isinstance(field, (Nested, OptionalNested)) and item is not None and isinstance(item, dict):
                item = field.struct_type.from_dict(item)
            elif isinstance(field, FixedArray) and isinstance(item, list):
                if isinstance(field.element, Nested):
                    item = [field.element.struct_type.from_dict(x) if isinstance(x, dict) else x for x in item]
            kwargs[name] = item
        return cls(**kwargs)

    @classmethod
    def pydantic_model(cls) -> type[BaseModel]:
        cached = cls.__pydantic_model_cache__
        if cached is not None:
            return cached

        model_fields: dict[str, tuple[Any, Any]] = {}
        for name, field in cls.__fixed_fields__:
            model_fields[name] = (field.pydantic_annotation(), field.pydantic_field())

        model = create_model(f"{cls.__name__}Json", **model_fields)
        cls.__pydantic_model_cache__ = model
        return model

    @classmethod
    def json_schema(cls) -> dict[str, Any]:
        return cls.pydantic_model().model_json_schema()

    @classmethod
    def from_pydantic(cls, model: BaseModel) -> FixedStruct:
        return cls.from_dict(model.model_dump(mode="python"))

    def to_pydantic(self) -> BaseModel:
        return self.pydantic_model().model_validate(self.to_dict())

    def __repr__(self) -> str:
        body = ", ".join(f"{name}={getattr(self, name)!r}" for name, _ in self.__fixed_fields__)
        return f"{type(self).__name__}({body})"


# Concise public alias matching the protocol terminology used by NPB.
NpbStruct = FixedStruct
