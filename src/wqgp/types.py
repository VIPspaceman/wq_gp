"""Core type system for the WQ GP alpha mining engine.

Design decisions vs. the reference project (VIPspaceman/wq):
- BRAIN's native DataType vocabulary is preserved 1:1.
- CATEGORICAL is NOT a DataType; it is a SemanticKind attached to a field.
  Reason: BRAIN returns "GROUP" for sector/industry fields. If we mapped
  categorical -> its own DataType, every API response would fail type
  validation. See DECISIONS.md D-001.
- ParamDomain is a single frozen dataclass with a discriminant (`kind`)
  instead of a class hierarchy. See DECISIONS.md D-002.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Union


class DataType(str, Enum):
    """Native BRAIN data types.

    Kept 1:1 with the platform's API vocabulary.
    Do NOT add CATEGORICAL here -- see SemanticKind.
    """
    MATRIX = "MATRIX"
    VECTOR = "VECTOR"
    GROUP = "GROUP"
    UNIVERSE = "UNIVERSE"
    SCALAR = "SCALAR"


class SemanticKind(str, Enum):
    """Semantic role attached to a field, orthogonal to its DataType.

    - CATEGORICAL: sector / industry / country / exchange. Almost always
      sits on a DataType.GROUP field.
    - NUMERIC:     ordinary numeric series (close, volume, returns, ...).
    - TIME:        timestamps or time-derived series.
    """
    CATEGORICAL = "CATEGORICAL"
    NUMERIC = "NUMERIC"
    TIME = "TIME"


class ParamKind(str, Enum):
    """Discriminant for ParamDomain."""
    INT_RANGE = "INT_RANGE"
    FLOAT_RANGE = "FLOAT_RANGE"
    CHOICE = "CHOICE"


@dataclass(frozen=True, slots=True)
class ParamDomain:
    """Domain of a single operator parameter.

    One frozen dataclass with a discriminant instead of a class hierarchy:
    - JSON round-trip is trivial (no registry, no polymorphic loader).
    - Frozen equality works out of the box for operator fingerprints.
    - Field count is small; the discriminant keeps it readable.

    Invariants (enforced in __post_init__):
      INT_RANGE   : min_value <= max_value, both int-valued, default in range
      FLOAT_RANGE : min_value <= max_value, default in range
      CHOICE      : choices non-empty, default in choices (if provided)
    """
    kind: ParamKind
    min_value: float | None = None
    max_value: float | None = None
    choices: tuple[str, ...] = ()
    default: float | int | str | None = None

    def __post_init__(self) -> None:
        if self.kind in (ParamKind.INT_RANGE, ParamKind.FLOAT_RANGE):
            if self.min_value is None or self.max_value is None:
                raise ValueError(f"{self.kind} requires min_value and max_value")
            if self.min_value > self.max_value:
                raise ValueError(
                    f"min_value ({self.min_value}) > max_value ({self.max_value})"
                )
            if self.default is not None and not self.contains(self.default):
                raise ValueError(f"default {self.default!r} out of range")
        elif self.kind is ParamKind.CHOICE:
            if not self.choices:
                raise ValueError("CHOICE requires non-empty choices")
            if self.default is not None and self.default not in self.choices:
                raise ValueError(f"default {self.default!r} not in choices")

    # --- constructors: keep call sites terse and unambiguous ---
    @classmethod
    def int_range(
        cls, lo: int, hi: int, default: int | None = None
    ) -> "ParamDomain":
        return cls(
            kind=ParamKind.INT_RANGE,
            min_value=lo,
            max_value=hi,
            default=default,
        )

    @classmethod
    def float_range(
        cls, lo: float, hi: float, default: float | None = None
    ) -> "ParamDomain":
        return cls(
            kind=ParamKind.FLOAT_RANGE,
            min_value=lo,
            max_value=hi,
            default=default,
        )

    @classmethod
    def choice(
        cls, options: tuple[str, ...], default: str | None = None
    ) -> "ParamDomain":
        return cls(kind=ParamKind.CHOICE, choices=options, default=default)

    # --- validation ---
    def contains(self, value: float | int | str) -> bool:
        """True iff `value` is a legal member of this domain."""
        if self.kind is ParamKind.INT_RANGE:
            if not isinstance(value, int) or isinstance(value, bool):
                return False
            return self.min_value <= value <= self.max_value  # type: ignore[operator]
        if self.kind is ParamKind.FLOAT_RANGE:
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                return False
            return self.min_value <= value <= self.max_value  # type: ignore[operator]
        if self.kind is ParamKind.CHOICE:
            return isinstance(value, str) and value in self.choices
        return False


# Convenience alias for signatures that accept a scalar constant / param.
ScalarValue = Union[int, float, str]