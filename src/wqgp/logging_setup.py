"""Structured logging configuration.

Design decisions:
- Logging is configured from a frozen LoggingSettings object, not from
  scattered logging.basicConfig() calls. See D-116.
- Component-level filtering uses Python's own logger hierarchy: a
  logger named "wqgp.brain.client" set to DEBUG overrides its parent
  "wqgp" set to INFO. See D-119.
- Style is a name ("default" / "compact" / "minimal"), not a raw
  format string. Exposing Python's format syntax to a JSON config is
  a footgun for non-Python users. See D-116.
- An optional log file, if set, receives the same records as stderr.
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


# --- styles -----------------------------------------------------------------

_STYLES: dict[str, str] = {
    "default": "%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    "compact": "%(levelname)-7s %(name)s: %(message)s",
    "minimal": "%(message)s",
}

_VALID_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


def _normalize_level(level: str) -> str:
    upper = level.upper()
    if upper not in _VALID_LEVELS:
        raise ValueError(
            f"unknown log level {level!r}; "
            f"expected one of {', '.join(_VALID_LEVELS)}"
        )
    return upper


# --- settings ---------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class LoggingSettings:
    """Frozen logging configuration.

    - level: the root logger's level.
    - style: one of the named styles in _STYLES.
    - components: per-logger overrides, as a tuple of (name, level)
      pairs. The name may be a full logger name or a prefix; Python's
      logging hierarchy makes "wqgp.brain" apply to "wqgp.brain.client"
      unless the latter has its own setting.
    - file: optional path; when set, records go to both stderr and the
      file.
    """
    level: str = "INFO"
    style: str = "default"
    components: tuple[tuple[str, str], ...] = ()
    file: str | None = None

    def __post_init__(self) -> None:
        _normalize_level(self.level)
        if self.style not in _STYLES:
            raise ValueError(
                f"unknown logging style {self.style!r}; "
                f"expected one of {', '.join(sorted(_STYLES))}"
            )
        for name, lvl in self.components:
            if not name or not name.strip():
                raise ValueError("component logger name must be non-empty")
            _normalize_level(lvl)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LoggingSettings":
        """Build from a JSON-shaped dict. Unknown keys are rejected."""
        if not isinstance(data, dict):
            raise ValueError("logging config must be an object")

        known = {"level", "style", "components", "file"}
        unknown = set(data) - known
        if unknown:
            raise ValueError(
                f"unknown logging keys: {sorted(unknown)}"
            )

        raw_components = data.get("components", {})
        if not isinstance(raw_components, dict):
            raise ValueError("logging.components must be an object")
        components = tuple(
            (str(k), str(v)) for k, v in raw_components.items()
        )

        file_raw = data.get("file")
        if file_raw is not None and not isinstance(file_raw, str):
            raise ValueError("logging.file must be a string or null")

        return cls(
            level=str(data.get("level", "INFO")),
            style=str(data.get("style", "default")),
            components=components,
            file=file_raw,
        )


# --- setup ------------------------------------------------------------------

_CONFIGURED = False


def setup_logging(settings: LoggingSettings) -> None:
    """Apply `settings` to the root logger and component loggers.

    Idempotent: a second call replaces the previous configuration
    rather than adding a second handler, so tests and CLI re-entry do
    not double-log.
    """
    global _CONFIGURED

    root = logging.getLogger()
    # Clear any handlers we or basicConfig installed earlier.
    for h in list(root.handlers):
        root.removeHandler(h)

    level = getattr(logging, _normalize_level(settings.level))
    root.setLevel(level)

    fmt = logging.Formatter(_STYLES[settings.style])

    stderr_handler = logging.StreamHandler(stream=sys.stderr)
    stderr_handler.setFormatter(fmt)
    root.addHandler(stderr_handler)

    if settings.file:
        path = Path(settings.file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)

    # Reset every component logger we previously configured back to
    # NOTSET, so removing an override from the config takes effect.
    for name in _previously_overridden():
        logging.getLogger(name).setLevel(logging.NOTSET)
    _remember_overridden((name for name, _ in settings.components))

    for name, lvl in settings.components:
        logging.getLogger(name).setLevel(
            getattr(logging, _normalize_level(lvl))
        )

    _CONFIGURED = True


# Component logger names we last touched. Reset on each setup_logging.
_LAST_OVERRIDES: set[str] = set()


def _previously_overridden() -> tuple[str, ...]:
    return tuple(_LAST_OVERRIDES)


def _remember_overridden(names) -> None:
    _LAST_OVERRIDES.clear()
    _LAST_OVERRIDES.update(names)


def is_configured() -> bool:
    return _CONFIGURED