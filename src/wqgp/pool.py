"""Field / Operator pools: the search space available to one GP run.

Design decisions:
- Pools build their indexes in __init__. Consumers never see the raw
  input iterable. See D-023.
- All returned tuples are in a deterministic order (by field_id / name).
  This is what makes population initialisation reproducible. See D-022.
- Cross-type semantic checks ("no fields of type VECTOR") are NOT done
  here. A run may legitimately start with an empty type bucket. See D-023.
- Pool.to_dict/from_dict call into serialization.py; no logic duplicated.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from wqgp.field import FieldDef
from wqgp.operator import OperatorDef
from wqgp.serialization import (
    field_from_dict,
    field_to_dict,
    operator_from_dict,
    operator_to_dict,
    validate_schema_version,
)
from wqgp.types import DataType


class PoolError(RuntimeError):
    """Raised on hard pool construction failures (duplicate IDs, etc.)."""


class FieldPool:
    """Immutable, indexed collection of FieldDefs.

    Indexes:
      - by field_id       (unique)
      - by DataType       (tuple, sorted by field_id)
    """

    __slots__ = ("_by_id", "_by_type")

    def __init__(self, fields: Iterable[FieldDef]) -> None:
        by_id: dict[str, FieldDef] = {}
        by_type: dict[DataType, list[FieldDef]] = defaultdict(list)
        for f in fields:
            if f.field_id in by_id:
                raise PoolError(f"duplicate field_id: {f.field_id!r}")
            by_id[f.field_id] = f
            by_type[f.data_type].append(f)

        frozen_by_type: dict[DataType, tuple[FieldDef, ...]] = {
            dt: tuple(sorted(bucket, key=lambda f: f.field_id))
            for dt, bucket in by_type.items()
        }

        self._by_id: Mapping[str, FieldDef] = MappingProxyType(by_id)
        self._by_type: Mapping[DataType, tuple[FieldDef, ...]] = (
            MappingProxyType(frozen_by_type)
        )

    # --- lookups ---

    def by_id(self, field_id: str) -> FieldDef:
        try:
            return self._by_id[field_id]
        except KeyError as e:
            raise KeyError(f"no field with id {field_id!r}") from e

    def by_type(self, dt: DataType) -> tuple[FieldDef, ...]:
        return self._by_type.get(dt, ())

    def all(self) -> tuple[FieldDef, ...]:
        """All fields, globally sorted by field_id (deterministic)."""
        return tuple(sorted(self._by_id.values(), key=lambda f: f.field_id))

    def __len__(self) -> int:
        return len(self._by_id)

    def __contains__(self, field_id: object) -> bool:
        return isinstance(field_id, str) and field_id in self._by_id

    def __iter__(self):
        return iter(self.all())
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FieldPool):
            return NotImplemented
        # Identity of fields is fully determined by (field_id -> FieldDef)
        # because duplicate field_ids are rejected at construction.
        return self._by_id == other._by_id

    def __hash__(self) -> int:
        # MappingProxyType is unhashable; derive a stable hash from the
        # sorted field_id tuple. FieldDef itself is already hashable.
        return hash(tuple(sorted(self._by_id)))

    def __repr__(self) -> str:
        return f"FieldPool(n={len(self)})"


class OperatorPool:
    """Immutable, indexed collection of OperatorDefs.

    Indexes:
      - by name           (unique)
      - by arity          (tuple, sorted by name)
      - by output type    (tuple, sorted by name)
    """

    __slots__ = ("_by_name", "_by_arity", "_by_output")

    def __init__(self, operators: Iterable[OperatorDef]) -> None:
        by_name: dict[str, OperatorDef] = {}
        by_arity: dict[int, list[OperatorDef]] = defaultdict(list)
        by_output: dict[DataType, list[OperatorDef]] = defaultdict(list)
        for op in operators:
            if op.name in by_name:
                raise PoolError(f"duplicate operator name: {op.name!r}")
            by_name[op.name] = op
            by_arity[op.arity].append(op)
            by_output[op.output_type].append(op)

        def _sorted(bucket: list[OperatorDef]) -> tuple[OperatorDef, ...]:
            return tuple(sorted(bucket, key=lambda op: op.name))

        self._by_name: Mapping[str, OperatorDef] = MappingProxyType(by_name)
        self._by_arity: Mapping[int, tuple[OperatorDef, ...]] = (
            MappingProxyType({k: _sorted(v) for k, v in by_arity.items()})
        )
        self._by_output: Mapping[DataType, tuple[OperatorDef, ...]] = (
            MappingProxyType({k: _sorted(v) for k, v in by_output.items()})
        )

    # --- lookups ---

    def by_name(self, name: str) -> OperatorDef:
        try:
            return self._by_name[name]
        except KeyError as e:
            raise KeyError(f"no operator named {name!r}") from e

    def by_arity(self, arity: int) -> tuple[OperatorDef, ...]:
        return self._by_arity.get(arity, ())

    def by_output_type(self, dt: DataType) -> tuple[OperatorDef, ...]:
        return self._by_output.get(dt, ())

    def all(self) -> tuple[OperatorDef, ...]:
        return tuple(sorted(self._by_name.values(), key=lambda op: op.name))

    def __len__(self) -> int:
        return len(self._by_name)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._by_name

    def __iter__(self):
        return iter(self.all())
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, OperatorPool):
            return NotImplemented
        return self._by_name == other._by_name

    def __hash__(self) -> int:
        return hash(tuple(sorted(self._by_name)))

    def __repr__(self) -> str:
        return f"OperatorPool(n={len(self)})"


@dataclass(frozen=True, slots=True)
class Pool:
    """The complete search space: fields + operators."""
    fields: FieldPool
    operators: OperatorPool

    # --- IO ---

    @classmethod
    def from_dict(cls, data: Any) -> "Pool":
        validate_schema_version(data)
        if not isinstance(data, dict):
            raise PoolError("pool payload must be an object")
        if "fields" not in data or "operators" not in data:
            raise PoolError("pool payload must contain 'fields' and 'operators'")
        fields_raw = data["fields"]
        ops_raw = data["operators"]
        if not isinstance(fields_raw, list):
            raise PoolError("'fields' must be a list")
        if not isinstance(ops_raw, list):
            raise PoolError("'operators' must be a list")
        fields = [field_from_dict(d) for d in fields_raw]
        operators = [operator_from_dict(d) for d in ops_raw]
        return cls(
            fields=FieldPool(fields),
            operators=OperatorPool(operators),
        )

    @classmethod
    def from_file(cls, path: Path | str) -> "Pool":
        p = Path(path)
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            raise PoolError(f"cannot read pool file {p}: {e}") from e
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        from wqgp.serialization import SERIALIZATION_SCHEMA_VERSION
        return {
            "schema_version": SERIALIZATION_SCHEMA_VERSION,
            "fields": [field_to_dict(f) for f in self.fields.all()],
            "operators": [operator_to_dict(op) for op in self.operators.all()],
        }

    def save(self, path: Path | str) -> None:
        """Atomically write the pool to JSON. Same tmp+replace pattern
        as checkpoints (see D-017)."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)

    # --- introspection ---

    def summary(self) -> dict[str, int]:
        from collections import Counter
        field_types = Counter(f.data_type.value for f in self.fields)
        return {
            "n_fields": len(self.fields),
            "n_operators": len(self.operators),
            **{f"fields_{k}": v for k, v in sorted(field_types.items())},
        }