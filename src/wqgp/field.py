"""FieldDef: a BRAIN data field descriptor.

Design decisions vs. the reference project (VIPspaceman/wq):
- CATEGORICAL/GROUP cross-check is enforced in __post_init__, so bad field
  definitions die at construction time, not at evaluation time.
  See DECISIONS.md D-006.
- No to_dict / from_dict yet. Serialization is Phase 2's job, and the
  loader will decide the on-disk schema. See DECISIONS.md D-005.
"""
from __future__ import annotations

from dataclasses import dataclass

from wqgp.types import DataType, SemanticKind


@dataclass(frozen=True, slots=True)
class FieldDef:
    """Immutable descriptor for a single BRAIN data field.

    Invariants (enforced in __post_init__):
      - field_id is non-empty and has no surrounding whitespace
      - dataset is non-empty
      - coverage is in [0.0, 1.0]
      - delay >= 0
      - semantic_kind=CATEGORICAL implies data_type=GROUP (one-way)
    """
    field_id: str
    data_type: DataType
    semantic_kind: SemanticKind
    dataset: str
    coverage: float = 1.0
    delay: int = 1
    unit: str | None = None

    def __post_init__(self) -> None:
        if not self.field_id or not self.field_id.strip():
            raise ValueError("field_id must be a non-empty string")
        if self.field_id != self.field_id.strip():
            raise ValueError(
                f"field_id has surrounding whitespace: {self.field_id!r}"
            )
        if not self.dataset or not self.dataset.strip():
            raise ValueError("dataset must be a non-empty string")
        if not 0.0 <= self.coverage <= 1.0:
            raise ValueError(
                f"coverage must be in [0.0, 1.0], got {self.coverage!r}"
            )
        if self.delay < 0:
            raise ValueError(f"delay must be >= 0, got {self.delay!r}")
        if (
            self.semantic_kind is SemanticKind.CATEGORICAL
            and self.data_type is not DataType.GROUP
        ):
            raise ValueError(
                f"CATEGORICAL field {self.field_id!r} must have "
                f"data_type=GROUP, got {self.data_type.value}"
            )

    @property
    def is_categorical(self) -> bool:
        """True iff this field is used as a grouping key."""
        return self.semantic_kind is SemanticKind.CATEGORICAL