"""Project-wide settings loaded from a single JSON file.

Design decisions:
- One file, two blocks: `logging` and `defaults`. See D-117.
- CLI flags override the file's defaults; the file overrides code
  defaults. Same precedence rule as D-088, applied to the new blocks.
- The `defaults` block only carries values that are commonly the same
  across runs. `run_id` and `pool_path` deliberately do NOT have
  defaults; they must be supplied per run.
- Unknown top-level keys are IGNORED so the same file can carry
  run-config keys for the CLI's own loader. Unknown keys inside the
  `logging` and `defaults` blocks are still rejected. See D-121.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from wqgp.logging_setup import LoggingSettings


# --- defaults ---------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class DefaultsSettings:
    """Default values for run parameters.

    These are consulted only when the corresponding CLI flag is not
    given and the run-level config does not supply the value.
    """
    # BRAIN simulation settings.
    instrument_type: str = "EQUITY"
    region: str = "USA"
    universe: str = "TOP3000"
    delay: int = 1
    decay: int = 4
    neutralization: str = "INDUSTRY"

    # Concurrency.
    n_workers: int = 4
    max_concurrent: int = 2

    # GP loop.
    seed: int = 42
    max_generations: int = 10
    population_size: int = 20
    elite_count: int = 2
    tournament_size: int = 3
    crossover_rate: float = 0.7
    max_breeding_attempts: int = 200

    # Paths.
    checkpoint_root: str = "./checkpoints"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DefaultsSettings":
        if not isinstance(data, dict):
            raise ValueError("defaults config must be an object")
        known = {f.name for f in cls.__dataclass_fields__.values()}
        unknown = set(data) - known
        if unknown:
            raise ValueError(
                f"unknown defaults keys: {sorted(unknown)}"
            )
        kwargs: dict[str, Any] = {}
        for key, value in data.items():
            field_type = cls.__dataclass_fields__[key].type
            if field_type is int or field_type == "int":
                value = int(value)
            elif field_type is float or field_type == "float":
                value = float(value)
            kwargs[key] = value
        return cls(**kwargs)


# --- top-level --------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Settings:
    """Aggregate of the blocks we recognize in the config file.

    Unknown top-level keys are ignored, so a file that also carries
    `run_id` / `pool_path` / `budget` (read by the CLI's own loader)
    is accepted. The `logging` and `defaults` blocks, if present, are
    validated strictly. See D-121.

    `raw` preserves the original dict so future blocks can be read by
    consumers without a schema change here.
    """
    logging: LoggingSettings = field(default_factory=LoggingSettings)
    defaults: DefaultsSettings = field(default_factory=DefaultsSettings)
    raw: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def default(cls) -> "Settings":
        return cls()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Settings":
        if not isinstance(data, dict):
            raise ValueError("settings file must be a JSON object")

        logging_block = data.get("logging", {})
        defaults_block = data.get("defaults", {})

        return cls(
            logging=LoggingSettings.from_dict(logging_block),
            defaults=DefaultsSettings.from_dict(defaults_block),
            raw=dict(data),
        )

    @classmethod
    def load(cls, path: Path | str | None) -> "Settings":
        """Load settings from `path`. A None path returns the defaults."""
        if path is None:
            return cls.default()
        p = Path(path)
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"settings file not found: {p}") from e
        except json.JSONDecodeError as e:
            raise ValueError(
                f"settings file {p} is not valid JSON: {e}"
            ) from e
        return cls.from_dict(data)