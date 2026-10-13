"""Tests for the top-level Settings loader. Phase 4 Step 5 gate."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from wqgp.logging_setup import LoggingSettings
from wqgp.settings import DefaultsSettings, Settings


# --- DefaultsSettings -------------------------------------------------------

def test_defaults_default_values() -> None:
    d = DefaultsSettings()
    assert d.region == "USA"
    assert d.universe == "TOP3000"
    assert d.delay == 1
    assert d.n_workers == 4
    assert d.max_concurrent == 2
    assert d.seed == 42


def test_defaults_from_dict_partial() -> None:
    d = DefaultsSettings.from_dict({"region": "CHN", "delay": 0})
    assert d.region == "CHN"
    assert d.delay == 0
    assert d.universe == "TOP3000"  # untouched default


def test_defaults_coerces_numeric_strings() -> None:
    d = DefaultsSettings.from_dict({"delay": "2", "seed": "7"})
    assert d.delay == 2
    assert isinstance(d.delay, int)
    assert d.seed == 7


def test_defaults_rejects_unknown_key() -> None:
    with pytest.raises(ValueError):
        DefaultsSettings.from_dict({"regions": "USA"})


def test_defaults_rejects_non_dict() -> None:
    with pytest.raises(ValueError):
        DefaultsSettings.from_dict([1, 2])


# --- Settings ---------------------------------------------------------------

def test_settings_default() -> None:
    s = Settings.default()
    assert isinstance(s.logging, LoggingSettings)
    assert isinstance(s.defaults, DefaultsSettings)
    assert s.logging.level == "INFO"


def test_settings_from_dict_empty() -> None:
    s = Settings.from_dict({})
    assert s.logging.level == "INFO"
    assert s.defaults.region == "USA"


def test_settings_from_dict_full() -> None:
    s = Settings.from_dict({
        "logging": {"level": "DEBUG"},
        "defaults": {"region": "CHN"},
    })
    assert s.logging.level == "DEBUG"
    assert s.defaults.region == "CHN"


def test_settings_ignores_unknown_top_level_keys() -> None:
    """Run-config keys at the top level are ignored so the same file
    can be read by both Settings and the run-config loader. See D-121.
    """
    s = Settings.from_dict({
        "run_id": "x",
        "pool_path": "/p.json",
        "budget": {"total_simulations": 1},
    })
    assert s.logging.level == "INFO"
    assert s.defaults.region == "USA"


def test_settings_still_rejects_unknown_logging_key() -> None:
    """The logging block's own keys remain strict: a typo there is a
    real error, unlike an unrelated run-config key."""
    with pytest.raises(ValueError):
        Settings.from_dict({"logging": {"levl": "INFO"}})


def test_settings_still_rejects_unknown_defaults_key() -> None:
    with pytest.raises(ValueError):
        Settings.from_dict({"defaults": {"regionn": "USA"}})


def test_settings_rejects_non_dict() -> None:
    with pytest.raises(ValueError):
        Settings.from_dict([1, 2, 3])


def test_settings_preserves_raw() -> None:
    raw = {"logging": {"level": "INFO"}, "run_id": "x"}
    s = Settings.from_dict(raw)
    assert s.raw == raw


# --- Settings.load ----------------------------------------------------------

def test_load_none_returns_defaults() -> None:
    s = Settings.load(None)
    assert s.logging.level == "INFO"


def test_load_reads_file(tmp_path: Path) -> None:
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps({
        "logging": {"level": "WARNING"},
        "defaults": {"region": "EUR"},
    }))
    s = Settings.load(p)
    assert s.logging.level == "WARNING"
    assert s.defaults.region == "EUR"


def test_load_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        Settings.load(tmp_path / "nope.json")


def test_load_bad_json_raises(tmp_path: Path) -> None:
    p = tmp_path / "cfg.json"
    p.write_text("{ not json")
    with pytest.raises(ValueError):
        Settings.load(p)


def test_load_accepts_unknown_top_level_keys(tmp_path: Path) -> None:
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps({"logging": {}, "run_id": "x"}))
    s = Settings.load(p)
    assert s.defaults.region == "USA"


def test_load_rejects_unknown_key_inside_logging(tmp_path: Path) -> None:
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps({"logging": {"levl": "INFO"}}))
    with pytest.raises(ValueError):
        Settings.load(p)