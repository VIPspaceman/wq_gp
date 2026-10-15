"""Crash-safe checkpoint store for GP runs.

Design decisions:
- A Checkpoint carries metadata + an OPAQUE payload. See D-014.
- JSON files, one per generation. See D-015.
- RNG state is serialized explicitly. See D-016.
- Writes are atomic: tmp file + os.replace. See D-017.
- schema_version and run_id validated on load. See D-018.
- Logs go to the "wqgp.storage" channel. See D-134.
- Run checkpoints live under <root>/runs/<run_id>/. The root also
  hosts cache.db and alphas.db, so keeping runs in a subdirectory
  keeps the three stores visibly separate. See D-161.
"""
from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from wqgp._ids import validate_run_id as _validate_run_id
from wqgp.budget import BudgetSnapshot


log = logging.getLogger("wqgp.storage")


SCHEMA_VERSION = 1

_STEP_RE = re.compile(r"^step_(\d+)\.json$")
_LATEST_FILE = "latest.txt"
_RUNS_SUBDIR = "runs"


class CheckpointError(RuntimeError):
    """Base class for checkpoint failures."""


class IncompatibleSchemaError(CheckpointError):
    """Raised when a checkpoint's schema_version is not what we expect."""


class CorruptCheckpointError(CheckpointError):
    """Raised when a checkpoint file cannot be parsed."""


def _rng_state_to_json(state: tuple[Any, ...]) -> list[Any]:
    version, internal, gauss = state
    return [version, list(internal), gauss]


def _rng_state_from_json(blob: list[Any]) -> tuple[Any, ...]:
    version, internal, gauss = blob
    return (int(version), tuple(int(x) for x in internal), gauss)


@dataclass(frozen=True, slots=True)
class Checkpoint:
    """A single, self-describing restore point."""
    run_id: str
    generation: int
    budget: BudgetSnapshot
    rng_state: list[Any]
    created_at: float
    payload: dict[str, Any]
    note: str = ""
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_run_id(self.run_id)
        if self.generation < 0:
            raise ValueError("generation must be >= 0")
        if self.schema_version != SCHEMA_VERSION:
            raise IncompatibleSchemaError(
                f"checkpoint schema {self.schema_version} != "
                f"expected {SCHEMA_VERSION}"
            )
        if not isinstance(self.payload, dict):
            raise ValueError("payload must be a dict")

    @classmethod
    def capture(
        cls,
        *,
        run_id: str,
        generation: int,
        budget: BudgetSnapshot,
        rng: random.Random,
        payload: dict[str, Any],
        note: str = "",
    ) -> "Checkpoint":
        return cls(
            run_id=run_id,
            generation=generation,
            budget=budget,
            rng_state=_rng_state_to_json(rng.getstate()),
            created_at=time.time(),
            payload=payload,
            note=note,
        )

    def restore_rng(self) -> random.Random:
        rng = random.Random()
        rng.setstate(_rng_state_from_json(self.rng_state))
        return rng


class CheckpointStore:
    """Filesystem-backed checkpoint store.

    Layout:
        <root>/cache.db                  (owned by EvaluationCache)
        <root>/alphas.db                 (owned by AlphaRegistry)
        <root>/runs/<run_id>/step_NNNNNN.json
        <root>/runs/<run_id>/latest.txt
    """

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._runs_dir.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        return self._root

    @property
    def _runs_dir(self) -> Path:
        return self._root / _RUNS_SUBDIR

    # --- paths ---

    def _run_dir(self, run_id: str) -> Path:
        _validate_run_id(run_id)
        return self._runs_dir / run_id

    def _step_path(self, run_id: str, generation: int) -> Path:
        if generation < 0:
            raise ValueError("generation must be >= 0")
        return self._run_dir(run_id) / f"step_{generation:06d}.json"

    def _latest_path(self, run_id: str) -> Path:
        return self._run_dir(run_id) / _LATEST_FILE

    # --- write ---

    def save(self, checkpoint: Checkpoint) -> Path:
        run_dir = self._run_dir(checkpoint.run_id)
        run_dir.mkdir(parents=True, exist_ok=True)

        target = self._step_path(checkpoint.run_id, checkpoint.generation)
        tmp = target.with_suffix(target.suffix + ".tmp")

        data = asdict(checkpoint)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, target)

        latest_tmp = self._latest_path(checkpoint.run_id).with_suffix(".txt.tmp")
        with open(latest_tmp, "w", encoding="utf-8") as f:
            f.write(str(checkpoint.generation))
            f.flush()
            os.fsync(f.fileno())
        os.replace(latest_tmp, self._latest_path(checkpoint.run_id))

        log.debug(
            "[storage] checkpoint saved: run=%s gen=%d",
            checkpoint.run_id, checkpoint.generation,
        )
        return target

    # --- read ---

    def load(self, run_id: str, generation: int) -> Checkpoint:
        path = self._step_path(run_id, generation)
        if not path.exists():
            raise FileNotFoundError(f"checkpoint not found: {path}")
        return self._read(path)

    def load_latest(self, run_id: str) -> Checkpoint | None:
        latest = self._latest_path(run_id)
        if latest.exists():
            try:
                gen = int(latest.read_text(encoding="utf-8").strip())
            except (ValueError, OSError) as e:
                raise CorruptCheckpointError(
                    f"unreadable latest pointer: {latest}"
                ) from e
            try:
                return self.load(run_id, gen)
            except FileNotFoundError as e:
                raise CorruptCheckpointError(
                    f"latest pointer references missing step: {gen}"
                ) from e

        gens = self.list_generations(run_id)
        if not gens:
            return None
        return self.load(run_id, gens[-1])

    def _read(self, path: Path) -> Checkpoint:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            raise CorruptCheckpointError(f"cannot read {path}: {e}") from e

        try:
            budget = BudgetSnapshot(**data["budget"])
            ckpt = Checkpoint(
                run_id=data["run_id"],
                generation=int(data["generation"]),
                budget=budget,
                rng_state=data["rng_state"],
                created_at=float(data["created_at"]),
                payload=data["payload"],
                note=data.get("note", ""),
                schema_version=int(data.get("schema_version", -1)),
            )
        except IncompatibleSchemaError:
            raise
        except (KeyError, TypeError, ValueError) as e:
            raise CorruptCheckpointError(
                f"malformed checkpoint {path}: {e}"
            ) from e

        log.debug(
            "[storage] checkpoint loaded: run=%s gen=%d",
            ckpt.run_id, ckpt.generation,
        )
        return ckpt

    # --- introspection ---

    def list_generations(self, run_id: str) -> list[int]:
        run_dir = self._run_dir(run_id)
        if not run_dir.exists():
            return []
        gens: list[int] = []
        for entry in run_dir.iterdir():
            m = _STEP_RE.match(entry.name)
            if m:
                gens.append(int(m.group(1)))
        return sorted(gens)

    def list_runs(self) -> list[str]:
        if not self._runs_dir.exists():
            return []
        return sorted(
            p.name for p in self._runs_dir.iterdir() if p.is_dir()
        )

    # --- maintenance ---

    def delete_run(self, run_id: str) -> None:
        run_dir = self._run_dir(run_id)
        if not run_dir.exists():
            return
        for entry in run_dir.iterdir():
            if entry.is_file():
                entry.unlink()
        run_dir.rmdir()