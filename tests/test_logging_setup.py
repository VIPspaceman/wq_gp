"""Tests for structured logging configuration. Phase 4 Step 5 gate."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from wqgp.logging_setup import (
    LoggingSettings,
    setup_logging,
)


@pytest.fixture(autouse=True)
def _clean_loggers():
    """Reset root and channel loggers between tests."""
    yield
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(logging.WARNING)
    for name in (
        "wqgp", "wqgp.brain", "wqgp.brain.client",
        "wqgp.gp", "wqgp.execution", "wqgp.coordinator",
        "wqgp.notification", "wqgp.cli",
    ):
        logging.getLogger(name).setLevel(logging.NOTSET)


# --- LoggingSettings validation --------------------------------------------

def test_settings_default() -> None:
    s = LoggingSettings()
    assert s.level == "INFO"
    assert s.style == "default"
    assert s.channels == ()
    assert s.file is None


def test_settings_rejects_unknown_level() -> None:
    with pytest.raises(ValueError):
        LoggingSettings(level="BOGUS")


def test_settings_rejects_unknown_style() -> None:
    with pytest.raises(ValueError):
        LoggingSettings(style="fancy")


def test_settings_rejects_empty_channel_name() -> None:
    with pytest.raises(ValueError):
        LoggingSettings(channels=(("", "DEBUG"),))


def test_settings_rejects_bad_channel_level() -> None:
    with pytest.raises(ValueError):
        LoggingSettings(channels=(("brain", "LOUD"),))


def test_settings_accepts_off_level() -> None:
    s = LoggingSettings(channels=(("brain", "OFF"),))
    assert s.channels == (("brain", "OFF"),)


def test_settings_level_is_case_insensitive() -> None:
    s = LoggingSettings(level="info", channels=(("brain", "debug"),))
    assert s.level == "info"  # stored as-is
    setup_logging(s)  # no error


# --- from_dict --------------------------------------------------------------

def test_from_dict_empty() -> None:
    s = LoggingSettings.from_dict({})
    assert s.level == "INFO"
    assert s.channels == ()


def test_from_dict_full() -> None:
    s = LoggingSettings.from_dict({
        "level": "DEBUG",
        "style": "compact",
        "channels": {"brain": "DEBUG"},
        "file": "/tmp/x.log",
    })
    assert s.level == "DEBUG"
    assert s.style == "compact"
    assert s.channels == (("brain", "DEBUG"),)
    assert s.file == "/tmp/x.log"


def test_from_dict_legacy_components_alias() -> None:
    """`components` from D-119 still parses into `channels`."""
    s = LoggingSettings.from_dict({
        "components": {"wqgp.brain.client": "DEBUG"},
    })
    assert s.channels == (("wqgp.brain.client", "DEBUG"),)


def test_from_dict_channels_wins_over_components() -> None:
    s = LoggingSettings.from_dict({
        "components": {"brain": "DEBUG"},
        "channels": {"brain": "INFO"},
    })
    assert s.channels == (("brain", "INFO"),)


def test_from_dict_merges_components_and_channels() -> None:
    s = LoggingSettings.from_dict({
        "components": {"brain": "DEBUG"},
        "channels": {"gp": "INFO"},
    })
    assert dict(s.channels) == {"brain": "DEBUG", "gp": "INFO"}


def test_from_dict_rejects_unknown_key() -> None:
    with pytest.raises(ValueError):
        LoggingSettings.from_dict({"levl": "INFO"})


def test_from_dict_rejects_non_dict() -> None:
    with pytest.raises(ValueError):
        LoggingSettings.from_dict([1, 2, 3])  # type: ignore[arg-type]


def test_from_dict_rejects_bad_channels_type() -> None:
    with pytest.raises(ValueError):
        LoggingSettings.from_dict({"channels": ["brain"]})


def test_from_dict_rejects_bad_components_type() -> None:
    with pytest.raises(ValueError):
        LoggingSettings.from_dict({"components": ["brain"]})


# --- setup_logging ----------------------------------------------------------

def test_setup_sets_root_level() -> None:
    setup_logging(LoggingSettings(level="DEBUG"))
    assert logging.getLogger().level == logging.DEBUG


def test_setup_installs_stderr_handler() -> None:
    setup_logging(LoggingSettings())
    root = logging.getLogger()
    assert any(
        isinstance(h, logging.StreamHandler)
        for h in root.handlers
    )


def test_setup_applies_channel_level_short_name() -> None:
    """Short name 'brain' maps to logger 'wqgp.brain'."""
    setup_logging(LoggingSettings(
        level="INFO",
        channels=(("brain", "DEBUG"),),
    ))
    assert logging.getLogger("wqgp.brain").level == logging.DEBUG
    # A sibling channel is not affected.
    assert logging.getLogger("wqgp.gp").level == logging.NOTSET


def test_setup_channel_full_name_used_verbatim() -> None:
    """A dotted name under 'wqgp' is used as-is."""
    setup_logging(LoggingSettings(
        level="INFO",
        channels=(("wqgp.brain.client", "DEBUG"),),
    ))
    assert (
        logging.getLogger("wqgp.brain.client").level == logging.DEBUG
    )


def test_setup_off_level_blocks_channel() -> None:
    """OFF maps to a level above CRITICAL."""
    setup_logging(LoggingSettings(
        level="DEBUG",
        channels=(("brain", "OFF"),),
    ))
    brain_level = logging.getLogger("wqgp.brain").level
    assert brain_level > logging.CRITICAL


def test_setup_resets_stale_channel_overrides() -> None:
    setup_logging(LoggingSettings(
        channels=(("brain", "DEBUG"),),
    ))
    assert logging.getLogger("wqgp.brain").level == logging.DEBUG

    setup_logging(LoggingSettings())  # no channels
    assert logging.getLogger("wqgp.brain").level == logging.NOTSET


def test_setup_is_idempotent() -> None:
    setup_logging(LoggingSettings())
    setup_logging(LoggingSettings())
    root = logging.getLogger()
    stream_handlers = [
        h for h in root.handlers
        if isinstance(h, logging.StreamHandler)
        and not isinstance(h, logging.FileHandler)
    ]
    assert len(stream_handlers) == 1


def test_setup_writes_to_file(tmp_path: Path) -> None:
    log_path = tmp_path / "wqgp.log"
    setup_logging(LoggingSettings(file=str(log_path)))
    logging.getLogger("wqgp.test").warning("hello file")
    for h in logging.getLogger().handlers:
        h.flush()
    assert log_path.exists()
    assert "hello file" in log_path.read_text(encoding="utf-8")


def test_setup_creates_parent_dir_for_file(tmp_path: Path) -> None:
    log_path = tmp_path / "nested" / "dir" / "wqgp.log"
    setup_logging(LoggingSettings(file=str(log_path)))
    assert log_path.parent.exists()


def test_compact_style_omits_timestamp() -> None:
    setup_logging(LoggingSettings(style="compact"))
    root = logging.getLogger()
    handler = next(
        h for h in root.handlers
        if isinstance(h, logging.StreamHandler)
    )
    assert "%(asctime)s" not in handler.formatter._fmt


def test_minimal_style_only_message() -> None:
    setup_logging(LoggingSettings(style="minimal"))
    root = logging.getLogger()
    handler = next(
        h for h in root.handlers
        if isinstance(h, logging.StreamHandler)
    )
    assert handler.formatter._fmt == "%(message)s"