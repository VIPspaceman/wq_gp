"""Project-wide settings loaded from a single JSON file.

Design decisions:
- One file, three blocks: `credentials`, `logging`, `defaults`.
  See D-117 and D-135.
- CLI flags override the file's defaults; the file overrides code
  defaults. Same precedence rule as D-088.
- Unknown top-level keys are IGNORED so the same file can carry
  run-config keys for the CLI's own loader. Unknown keys inside the
  recognized blocks are still rejected. See D-121.
- Credentials in the config file are supported but env vars take
  precedence. See D-135.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from wqgp.logging_setup import LoggingSettings


# --- credentials ------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class CredentialsSettings:
    """BRAIN login read from the config file.

    Both fields are optional. `is_complete()` returns True only when
    both are set, so a half-filled block is treated as absent rather
    than as a partial credential. See D-135.
    """
    username: str | None = None
    password: str | None = None

    def is_complete(self) -> bool:
        return bool(self.username and self.password)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CredentialsSettings":
        if not isinstance(data, dict):
            raise ValueError("credentials block must be an object")
        known = {"username", "password"}
        unknown = set(data) - known
        if unknown:
            raise ValueError(
                f"unknown credentials keys: {sorted(unknown)}"
            )
        username = data.get("username")
        password = data.get("password")
        if username is not None and not isinstance(username, str):
            raise ValueError("credentials.username must be a string")
        if password is not None and not isinstance(password, str):
            raise ValueError("credentials.password must be a string")
        return cls(username=username, password=password)


# --- defaults ---------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class DefaultsSettings:
    """Default values for run parameters.

    Consulted only when the corresponding CLI flag is not given and
    the run-level config does not supply the value.
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
    `run_id` / `pool_path` / `budget` (read by the CLI's run-config
    loader) is accepted. Recognized blocks are validated strictly.
    See D-121.
    """
    logging: LoggingSettings = field(default_factory=LoggingSettings)
    defaults: DefaultsSettings = field(default_factory=DefaultsSettings)
    credentials: CredentialsSettings = field(
        default_factory=CredentialsSettings
    )
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
        credentials_block = data.get("credentials", {})

        return cls(
            logging=LoggingSettings.from_dict(logging_block),
            defaults=DefaultsSettings.from_dict(defaults_block),
            credentials=CredentialsSettings.from_dict(credentials_block),
            raw=dict(data),
        )

    @classmethod
    def load(cls, path: Path | str | None) -> "Settings":
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


# --- helpers ----------------------------------------------------------------

def is_insecure_credentials_file(path: Path) -> bool:
    """True iff the file is readable by anyone other than its owner.

    On POSIX systems this checks the group/other read bits. On Windows
    the mode is a stub, and the check returns False. See D-135.
    """
    if not path.exists():
        return False
    try:
        mode = path.stat().st_mode & 0o777
    except OSError:
        return False
    return bool(mode & 0o077)