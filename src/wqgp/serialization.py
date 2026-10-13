"""JSON serialization primitives for value objects.

Design decisions:
- Value objects (FieldDef, OperatorDef, ParamDomain) do NOT carry
  to_dict/from_dict. All conversion lives here. See D-020 / D-005.
- The schema version lives at the *file* level, not per-object. See D-021.
- Enum parsing rejects unknown values loudly; silent fallback to a default
  is how "why is my field GROUP everywhere?" bugs are born.
"""
from __future__ import annotations

from typing import Any

from wqgp.field import FieldDef
from wqgp.operator import OperatorDef
from wqgp.types import DataType, ParamDomain, ParamKind, SemanticKind


SERIALIZATION_SCHEMA_VERSION = 1


class SerializationError(RuntimeError):
    """Raised when data on disk cannot be parsed into a value object."""


class UnsupportedSchemaError(SerializationError):
    """Raised when a payload's schema_version is not what we expect."""


# --- enum helpers ---

def _dt_to_str(dt: DataType) -> str:
    return dt.value


def _dt_from_str(s: Any) -> DataType:
    if not isinstance(s, str):
        raise SerializationError(f"data_type must be a string, got {type(s).__name__}")
    try:
        return DataType(s)
    except ValueError as e:
        raise SerializationError(f"unknown DataType: {s!r}") from e


def _sk_to_str(sk: SemanticKind) -> str:
    return sk.value


def _sk_from_str(s: Any) -> SemanticKind:
    if not isinstance(s, str):
        raise SerializationError(
            f"semantic_kind must be a string, got {type(s).__name__}"
        )
    try:
        return SemanticKind(s)
    except ValueError as e:
        raise SerializationError(f"unknown SemanticKind: {s!r}") from e


def _pk_to_str(pk: ParamKind) -> str:
    return pk.value


def _pk_from_str(s: Any) -> ParamKind:
    if not isinstance(s, str):
        raise SerializationError(
            f"param kind must be a string, got {type(s).__name__}"
        )
    try:
        return ParamKind(s)
    except ValueError as e:
        raise SerializationError(f"unknown ParamKind: {s!r}") from e


# --- ParamDomain ---

def param_domain_to_dict(d: ParamDomain) -> dict[str, Any]:
    out: dict[str, Any] = {"kind": _pk_to_str(d.kind)}
    if d.min_value is not None:
        out["min_value"] = d.min_value
    if d.max_value is not None:
        out["max_value"] = d.max_value
    if d.choices:
        out["choices"] = list(d.choices)
    if d.default is not None:
        out["default"] = d.default
    return out


def param_domain_from_dict(data: Any) -> ParamDomain:
    if not isinstance(data, dict):
        raise SerializationError(
            f"ParamDomain must be an object, got {type(data).__name__}"
        )
    if "kind" not in data:
        raise SerializationError("ParamDomain missing 'kind'")
    kind = _pk_from_str(data["kind"])

    choices_raw = data.get("choices", ())
    if not isinstance(choices_raw, (list, tuple)):
        raise SerializationError("ParamDomain.choices must be a list")
    choices = tuple(str(c) for c in choices_raw)

    try:
        return ParamDomain(
            kind=kind,
            min_value=data.get("min_value"),
            max_value=data.get("max_value"),
            choices=choices,
            default=data.get("default"),
        )
    except ValueError as e:
        raise SerializationError(f"invalid ParamDomain: {e}") from e


# --- FieldDef ---

def field_to_dict(f: FieldDef) -> dict[str, Any]:
    out: dict[str, Any] = {
        "field_id": f.field_id,
        "data_type": _dt_to_str(f.data_type),
        "semantic_kind": _sk_to_str(f.semantic_kind),
        "dataset": f.dataset,
        "coverage": f.coverage,
        "delay": f.delay,
    }
    if f.unit is not None:
        out["unit"] = f.unit
    return out


def field_from_dict(data: Any) -> FieldDef:
    if not isinstance(data, dict):
        raise SerializationError(
            f"FieldDef must be an object, got {type(data).__name__}"
        )
    for key in ("field_id", "data_type", "semantic_kind", "dataset"):
        if key not in data:
            raise SerializationError(f"FieldDef missing '{key}'")
    try:
        return FieldDef(
            field_id=data["field_id"],
            data_type=_dt_from_str(data["data_type"]),
            semantic_kind=_sk_from_str(data["semantic_kind"]),
            dataset=data["dataset"],
            coverage=data.get("coverage", 1.0),
            delay=data.get("delay", 1),
            unit=data.get("unit"),
        )
    except ValueError as e:
        raise SerializationError(
            f"invalid FieldDef {data.get('field_id')!r}: {e}"
        ) from e


# --- OperatorDef ---

def operator_to_dict(op: OperatorDef) -> dict[str, Any]:
    # input_types positions are frozensets after normalization:
    # - single type  -> "MATRIX"
    # - union type   -> ["MATRIX", "VECTOR"]  (sorted, deterministic)
    input_repr: list[Any] = []
    for pos in op.input_types:
        if len(pos) == 1:
            (only,) = tuple(pos)
            input_repr.append(_dt_to_str(only))
        else:
            input_repr.append(sorted(_dt_to_str(t) for t in pos))

    out: dict[str, Any] = {
        "name": op.name,
        "input_types": input_repr,
        "output_type": _dt_to_str(op.output_type),
    }
    if op.param_domains:
        # tuple of (name, ParamDomain) -> dict preserves order in JSON
        out["param_domains"] = {
            name: param_domain_to_dict(dom) for name, dom in op.param_domains
        }
    # Only emit flags that are set; keeps the file small and readable.
    if op.commutative:
        out["commutative"] = True
    if op.associative:
        out["associative"] = True
    if op.symmetric:
        out["symmetric"] = True
    if not op.deterministic:
        out["deterministic"] = False
    return out


def _parse_input_spec(raw: Any, op_name: str, pos: int) -> DataType | tuple[DataType, ...]:
    if isinstance(raw, str):
        return _dt_from_str(raw)
    if isinstance(raw, list):
        if not raw:
            raise SerializationError(
                f"operator {op_name!r} position {pos}: empty input-type union"
            )
        return tuple(_dt_from_str(x) for x in raw)
    raise SerializationError(
        f"operator {op_name!r} position {pos}: expected string or list, "
        f"got {type(raw).__name__}"
    )


def operator_from_dict(data: Any) -> OperatorDef:
    if not isinstance(data, dict):
        raise SerializationError(
            f"OperatorDef must be an object, got {type(data).__name__}"
        )
    for key in ("name", "input_types", "output_type"):
        if key not in data:
            raise SerializationError(f"OperatorDef missing '{key}'")

    name = data["name"]
    if not isinstance(name, str):
        raise SerializationError("OperatorDef.name must be a string")

    raw_inputs = data["input_types"]
    if not isinstance(raw_inputs, list):
        raise SerializationError("OperatorDef.input_types must be a list")
    input_specs = tuple(
        _parse_input_spec(raw, name, i) for i, raw in enumerate(raw_inputs)
    )

    raw_params = data.get("param_domains", {})
    if not isinstance(raw_params, dict):
        raise SerializationError("OperatorDef.param_domains must be an object")
    param_domains = tuple(
        (pname, param_domain_from_dict(pdom))
        for pname, pdom in raw_params.items()
    )

    try:
        return OperatorDef(
            name=name,
            input_types=input_specs,
            output_type=_dt_from_str(data["output_type"]),
            param_domains=param_domains,
            commutative=bool(data.get("commutative", False)),
            associative=bool(data.get("associative", False)),
            symmetric=bool(data.get("symmetric", False)),
            deterministic=bool(data.get("deterministic", True)),
        )
    except ValueError as e:
        raise SerializationError(f"invalid OperatorDef {name!r}: {e}") from e


# --- file-level schema check ---

def validate_schema_version(data: Any) -> int:
    if not isinstance(data, dict):
        raise SerializationError(
            f"payload must be an object, got {type(data).__name__}"
        )
    version = data.get("schema_version")
    if version is None:
        raise SerializationError("payload missing 'schema_version'")
    if version != SERIALIZATION_SCHEMA_VERSION:
        raise UnsupportedSchemaError(
            f"payload schema_version={version} != "
            f"expected {SERIALIZATION_SCHEMA_VERSION}"
        )
    return int(version)