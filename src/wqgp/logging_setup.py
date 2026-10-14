"""Structured logging configuration.

Design decisions:
- Logging is configured from a frozen LoggingSettings object. See D-116.
- Channels are short names ("brain", "gp") mapped to "wqgp.<name>"
  loggers. Full logger names ("wqgp.brain.client") are also accepted
  and used verbatim. See D-132.
- OFF is a valid channel level; it maps to a value above CRITICAL so
  every record on that channel is dropped. See D-132.
- Style is a name ("default" / "compact" / "minimal"), not a raw
  format string. See D-116.
- An optional log file receives the same records as stderr.
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

_VALID_LEVELS = (
    "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "OFF",
)

# A level above CRITICAL that no record will ever reach, so any logger
# set to it drops everything.
_OFF_LEVEL = logging.CRITICAL + 10


def _normalize_level(level: str) -> str:
    upper = level.upper()
    if upper not in _VALID_LEVELS:
        raise ValueError(
            f"unknown log level {level!r}; "
            f"expected one of {', '.join(_VALID_LEVELS)}"
        )
    return upper


def _level_to_int(level: str) -> int:
    """Convert a level name to its integer value. OFF -> sentinel."""
    upper = _normalize_level(level)
    if upper == "OFF":
        return _OFF_LEVEL
    return getattr(logging, upper)


# --- settings ---------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class LoggingSettings:
    """Frozen logging configuration.

    - level: the root logger's level (applies unless a channel overrides).
    - style: one of the named styles in _STYLES.
    - channels: per-logger overrides. Keys may be short names ("brain")
      that resolve to "wqgp.brain", or full logger names
      ("wqgp.brain.client") that are used verbatim.
    - file: optional path; when set, records go to both stderr and the
      file.
    """
    level: str = "INFO"
    style: str = "default"
    channels: tuple[tuple[str, str], ...] = ()
    file: str | None = None

    def __post_init__(self) -> None:
        _normalize_level(self.level)
        if self.style not in _STYLES:
            raise ValueError(
                f"unknown logging style {self.style!r}; "
                f"expected one of {', '.join(sorted(_STYLES))}"
            )
        for name, lvl in self.channels:
            if not name or not name.strip():
                raise ValueError("channel name must be non-empty")
            _normalize_level(lvl)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LoggingSettings":
        """Build from a JSON-shaped dict. Unknown keys are rejected.

        Accepts both `channels` (new) and `components` (legacy). If
        both are present, `channels` wins on key conflicts.
        """
        if not isinstance(data, dict):
            raise ValueError("logging config must be an object")

        known = {"level", "style", "channels", "components", "file"}
        unknown = set(data) - known
        if unknown:
            raise ValueError(
                f"unknown logging keys: {sorted(unknown)}"
            )

        merged: dict[str, str] = {}

        raw_components = data.get("components", {})
        if not isinstance(raw_components, dict):
            raise ValueError("logging.components must be an object")
        for k, v in raw_components.items():
            merged[str(k)] = str(v)

        raw_channels = data.get("channels", {})
        if not isinstance(raw_channels, dict):
            raise ValueError("logging.channels must be an object")
        for k, v in raw_channels.items():
            merged[str(k)] = str(v)

        channels = tuple(sorted(merged.items()))

        file_raw = data.get("file")
        if file_raw is not None and not isinstance(file_raw, str):
            raise ValueError("logging.file must be a string or null")

        return cls(
            level=str(data.get("level", "INFO")),
            style=str(data.get("style", "default")),
            channels=channels,
            file=file_raw,
        )


# --- setup ------------------------------------------------------------------

_CONFIGURED = False


def _resolve_logger_name(name: str) -> str:
    """Short name -> 'wqgp.<name>'. Names already under 'wqgp' stay."""
    if name.startswith("wqgp.") or name == "wqgp":
        return name
    return f"wqgp.{name}"


def setup_logging(settings: LoggingSettings) -> None:
    """Apply `settings` to the root logger and channel loggers.

    Idempotent: a second call replaces the previous configuration
    rather than adding a second handler, so tests and CLI re-entry do
    not double-log.
    """
    global _CONFIGURED

    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)

    root.setLevel(_level_to_int(settings.level))

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

    # Reset previously configured channel loggers to NOTSET so removing
    # an override from the config takes effect.
    for name in _LAST_OVERRIDES:
        logging.getLogger(name).setLevel(logging.NOTSET)
    _LAST_OVERRIDES.clear()

    for short_name, lvl in settings.channels:
        full_name = _resolve_logger_name(short_name)
        logging.getLogger(full_name).setLevel(_level_to_int(lvl))
        _LAST_OVERRIDES.add(full_name)

    _CONFIGURED = True


# Channel logger names we last touched. Reset on each setup_logging.
_LAST_OVERRIDES: set[str] = set()


def is_configured() -> bool:
    return _CONFIGURED