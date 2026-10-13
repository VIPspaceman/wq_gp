"""Shared identifier validation. Private module.

Why a separate module: `checkpoint.py` and `persistence.py` both need the
same run_id rules. Duplicating them invites drift where a run_id is legal
in one store and illegal in the other. See DECISIONS.md D-026.
"""
from __future__ import annotations

import re


_RUN_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def validate_run_id(run_id: str) -> str:
    """Validate a run_id for filesystem/DB safety. Returns it unchanged."""
    if not isinstance(run_id, str):
        raise ValueError(
            f"run_id must be a string, got {type(run_id).__name__}"
        )
    if not run_id:
        raise ValueError("run_id must be non-empty")
    if run_id != run_id.strip():
        raise ValueError(f"run_id has surrounding whitespace: {run_id!r}")
    if not _RUN_ID_RE.match(run_id):
        raise ValueError(
            f"run_id contains illegal characters: {run_id!r} "
            f"(allowed: A-Z a-z 0-9 . _ -)"
        )
    # Defense in depth: the regex allows '..'. Path traversal must be
    # impossible even if the regex is later loosened.
    if ".." in run_id:
        raise ValueError(f"run_id must not contain '..': {run_id!r}")
    return run_id