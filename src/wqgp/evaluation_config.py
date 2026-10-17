"""The single source of truth for a simulation's identity.

Design decisions:
- EvaluationConfig is the ONE object that both generates the cache key
  and produces the BRAIN request payload. See D-138.
- settings_hash() is deterministic: sorted JSON keys, compact
  separators, SHA-256 hex. See D-139.
- cache_key() includes the fitness strategy so two runs that differ
  only in fitness semantics never share cache entries. See D-160.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from wqgp.brain.settings import BrainSettings


# Bump this when the meaning of a cache key changes. Old keys remain
# in the cache DB and are simply never matched again. See D-140.
CACHE_SCHEMA_VERSION = 2


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    """Frozen pairing of a BRAIN simulation's settings with its identity."""

    settings: BrainSettings

    def __post_init__(self) -> None:
        if not isinstance(self.settings, BrainSettings):
            raise ValueError(
                f"settings must be BrainSettings, "
                f"got {type(self.settings).__name__}"
            )

    # --- request side ---------------------------------------------------

    def to_api_dict(self) -> dict[str, Any]:
        return self.settings.to_api_dict()

    # --- identity side --------------------------------------------------

    def settings_hash(self) -> str:
        payload = json.dumps(
            self.to_api_dict(),
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def cache_key(
        self,
        fingerprint: str,
        fitness_strategy: str,
    ) -> str:
        """Return the cross-run cache key.

        Key material:
            "<version>:<fingerprint>:<settings_hash>:<fitness_strategy>"

        Both `fingerprint` and `fitness_strategy` are mandatory. Two
        configurations that differ only in how the fitness is computed
        (raw sharpe vs. feasibility-first) produce different keys and
        do not share cache entries. See D-160.
        """
        if not fingerprint:
            raise ValueError("fingerprint must be non-empty")
        if not fitness_strategy or not fitness_strategy.strip():
            raise ValueError("fitness_strategy must be non-empty")
        payload = (
            f"{CACHE_SCHEMA_VERSION}:"
            f"{fingerprint}:"
            f"{self.settings_hash()}:"
            f"{fitness_strategy}"
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()