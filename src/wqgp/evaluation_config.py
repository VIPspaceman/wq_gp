"""The single source of truth for a simulation's identity.

Design decisions:
- EvaluationConfig is the ONE object that both (a) generates the cache
  key and (b) produces the BRAIN request payload. This is the key
  invariant from the Phase 4 Step 8 design: if the cache key is
  computed from one config while the request is built from another,
  the cache can silently mis-hit. See D-138 and D-139.
- settings_hash() is deterministic: sorted JSON keys, no whitespace,
  SHA-256 hex. Two EvaluationConfig values with identical semantic
  content produce identical hashes regardless of construction order.
  See D-139.
- cache_key() combines a schema version constant with the AST
  fingerprint and the settings hash. The version is bumped when the
  key computation itself changes. See D-140.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from wqgp.brain.settings import BrainSettings


# Bump this when the meaning of a cache key changes (e.g. when
# settings_hash starts including a new field, or when the delimiter
# between components changes). Old keys remain in the cache DB and are
# simply never matched again. See D-140.
CACHE_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    """Frozen pairing of a BRAIN simulation's settings with its identity.

    Held by the Coordinator (future round) and passed unchanged to the
    cache layer (for key generation) and to the BrainClient (for
    request construction). Construct once per run; do not rebuild from
    raw values along the two paths. See D-138.
    """
    settings: BrainSettings

    def __post_init__(self) -> None:
        if not isinstance(self.settings, BrainSettings):
            raise ValueError(
                f"settings must be BrainSettings, "
                f"got {type(self.settings).__name__}"
            )

    # --- request side ---------------------------------------------------

    def to_api_dict(self) -> dict[str, Any]:
        """Return the payload dict for BRAIN's /simulations endpoint.

        Delegates to BrainSettings so there is exactly one place that
        knows the field names BRAIN expects.
        """
        return self.settings.to_api_dict()

    # --- identity side --------------------------------------------------

    def settings_hash(self) -> str:
        """Deterministic SHA-256 over the semantic content of the settings.

        Two things matter for determinism:

        1. sort_keys=True, so field ordering does not leak into the
           hash.
        2. separators=(",", ":"), so whitespace does not leak into the
           hash. json.dumps with default separators inserts spaces
           after ':' and ','; that would make {"a": 1} hash differently
           from {"a":1} if the two were produced by different code
           paths.
        """
        payload = json.dumps(
            self.to_api_dict(),
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def cache_key(self, fingerprint: str) -> str:
        """Return the cross-run cache key for `fingerprint` under these
        settings.

        The key is SHA-256 over "<version>:<fingerprint>:<settings_hash>".
        The version prefix ensures a change to the key format does not
        collide with pre-change keys. See D-140.
        """
        if not fingerprint:
            raise ValueError("fingerprint must be non-empty")
        payload = (
            f"{CACHE_SCHEMA_VERSION}:"
            f"{fingerprint}:"
            f"{self.settings_hash()}"
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()