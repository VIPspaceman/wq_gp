"""Audit a local operator signature table against BRAIN's operator list.

Design decisions:
- One function, one job: audit. No class, no state. See D-101.
- Never writes to the signature file. Output includes paste-ready
  skeletons, but the user decides what to add. See D-102.
- Skeletons use "MATRIX" as an obvious placeholder, not a guess. See
  D-103.
- The audit reuses PoolBuilder.fetch_operator_names without closing
  the transport, because the caller owns its lifecycle. See D-104.

This module never infers operator signatures. That is a standing rule
(see D-084). Missing operators are named and left for the user to fill
in. The tool provides the paste-ready JSON shape, not the answer.
"""
from __future__ import annotations

from dataclasses import dataclass

from wqgp.brain.client import BrainCredentials
from wqgp.brain.pool_builder import (
    OperatorSignatureTable,
    PoolBuilder,
)
from wqgp.brain.transport import HttpTransport


__all__ = [
    "SignatureAudit",
    "audit_signatures",
    "format_skeletons",
]


@dataclass(frozen=True, slots=True)
class SignatureAudit:
    """Result of comparing BRAIN's operator names to a local table.

    Invariants (enforced in __post_init__):
      - brain_operator_count == len(missing_from_table) + len(common)
      - table_operator_count == len(extra_in_table) + len(common)
      - missing_from_table, extra_in_table, common are pairwise disjoint
      - the three tuples are sorted
    """
    brain_operator_count: int
    table_operator_count: int
    missing_from_table: tuple[str, ...]
    extra_in_table: tuple[str, ...]
    common: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            len(self.missing_from_table) + len(self.common)
            != self.brain_operator_count
        ):
            raise ValueError(
                "brain_operator_count does not equal "
                "len(missing) + len(common)"
            )
        if (
            len(self.extra_in_table) + len(self.common)
            != self.table_operator_count
        ):
            raise ValueError(
                "table_operator_count does not equal "
                "len(extra) + len(common)"
            )
        missing = set(self.missing_from_table)
        extra = set(self.extra_in_table)
        common = set(self.common)
        if missing & extra:
            raise ValueError("missing_from_table and extra_in_table overlap")
        if missing & common:
            raise ValueError("missing_from_table and common overlap")
        if extra & common:
            raise ValueError("extra_in_table and common overlap")

    def is_complete(self) -> bool:
        """True iff every BRAIN operator has a signature in the table."""
        return not self.missing_from_table


def audit_signatures(
    credentials: BrainCredentials,
    transport: HttpTransport,
    table: OperatorSignatureTable,
    *,
    http_timeout: float = 30.0,
) -> SignatureAudit:
    """Compare BRAIN's operator names to `table` and return an audit.

    Does NOT close `transport`; the caller owns its lifecycle. See D-104.

    Raises PoolBuilderError / PoolBuilderProtocolError on login, HTTP,
    or response-shape failures. The caller decides how to surface them.
    """
    builder = PoolBuilder(
        credentials=credentials,
        transport=transport,
        http_timeout=http_timeout,
    )
    # Intentionally not calling builder.close(): we do not own the
    # transport. See D-104.
    brain_names, _pages = builder.fetch_operator_names()

    brain_set = set(brain_names)
    table_set = set(table.names())

    missing = tuple(sorted(brain_set - table_set))
    extra = tuple(sorted(table_set - brain_set))
    common = tuple(sorted(brain_set & table_set))

    return SignatureAudit(
        brain_operator_count=len(brain_names),
        table_operator_count=len(table),
        missing_from_table=missing,
        extra_in_table=extra,
        common=common,
    )


def format_skeletons(
    names: tuple[str, ...] | list[str],
    *,
    limit: int = 20,
) -> str:
    """Render paste-ready JSON skeletons for the first `limit` names.

    The type placeholder is "MATRIX", which is very likely wrong for
    most operators. It exists to show the JSON shape, not to answer the
    question. See D-103.
    """
    if limit < 0:
        raise ValueError("limit must be >= 0")
    shown = list(names)[:limit]
    lines = [
        f'  {{"name": "{name}", '
        f'"input_types": ["MATRIX"], "output_type": "MATRIX"}},'
        for name in shown
    ]
    remaining = len(names) - len(shown)
    if remaining > 0:
        lines.append(f"  ... and {remaining} more")
    return "\n".join(lines)