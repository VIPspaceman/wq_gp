"""Tests for EvaluationConfig. Cache-key-v2 gate."""
from __future__ import annotations

import pytest

from wqgp.brain.settings import BrainSettings
from wqgp.evaluation_config import (
    CACHE_SCHEMA_VERSION,
    EvaluationConfig,
)


def _cfg(**overrides) -> EvaluationConfig:
    base = dict(
        region="USA", universe="TOP3000", delay=1,
        decay=4, neutralization="INDUSTRY",
    )
    base.update(overrides)
    return EvaluationConfig(BrainSettings(**base))


def test_config_rejects_non_brainsettings() -> None:
    with pytest.raises(ValueError):
        EvaluationConfig("not a settings")  # type: ignore[arg-type]


def test_to_api_dict_matches_brainsettings() -> None:
    s = BrainSettings(region="CHN", delay=0)
    cfg = EvaluationConfig(s)
    assert cfg.to_api_dict() == s.to_api_dict()


def test_settings_hash_is_deterministic() -> None:
    assert _cfg().settings_hash() == _cfg().settings_hash()


def test_settings_hash_differs_on_region() -> None:
    assert (
        _cfg(region="USA").settings_hash()
        != _cfg(region="CHN").settings_hash()
    )


def test_settings_hash_differs_on_decay() -> None:
    assert (
        _cfg(decay=4).settings_hash()
        != _cfg(decay=0).settings_hash()
    )


def test_settings_hash_differs_on_neutralization() -> None:
    a = _cfg(neutralization="INDUSTRY")
    b = _cfg(neutralization="SUBINDUSTRY")
    assert a.settings_hash() != b.settings_hash()


def test_settings_hash_is_hex_sha256() -> None:
    h = _cfg().settings_hash()
    assert len(h) == 64
    int(h, 16)


# --- cache_key: strategy dimension (D-160) ---------------------------------

def test_cache_key_is_deterministic() -> None:
    cfg = _cfg()
    assert (
        cfg.cache_key("abc", "sharpe")
        == cfg.cache_key("abc", "sharpe")
    )


def test_cache_key_differs_on_fingerprint() -> None:
    cfg = _cfg()
    assert (
        cfg.cache_key("aaa", "feasibility")
        != cfg.cache_key("bbb", "feasibility")
    )


def test_cache_key_differs_on_settings() -> None:
    a = _cfg(region="USA")
    b = _cfg(region="CHN")
    assert (
        a.cache_key("abc", "feasibility")
        != b.cache_key("abc", "feasibility")
    )


def test_cache_key_differs_on_fitness_strategy() -> None:
    """The core of D-160: same expression + settings, different
    strategy -> different key."""
    cfg = _cfg()
    assert (
        cfg.cache_key("abc", "feasibility")
        != cfg.cache_key("abc", "sharpe")
    )


def test_cache_key_rejects_empty_fingerprint() -> None:
    with pytest.raises(ValueError):
        _cfg().cache_key("", "feasibility")


def test_cache_key_rejects_empty_strategy() -> None:
    with pytest.raises(ValueError):
        _cfg().cache_key("abc", "")


def test_cache_key_rejects_whitespace_strategy() -> None:
    with pytest.raises(ValueError):
        _cfg().cache_key("abc", "   ")


def test_cache_key_is_hex_sha256() -> None:
    key = _cfg().cache_key("abc", "feasibility")
    assert len(key) == 64
    int(key, 16)


def test_cache_schema_version_is_positive_int() -> None:
    assert isinstance(CACHE_SCHEMA_VERSION, int)
    assert CACHE_SCHEMA_VERSION >= 1


def test_all_settings_and_strategies_are_distinct() -> None:
    """12 combinations of (config, strategy) produce 12 unique keys."""
    fp = "same-fingerprint"
    configs = [
        _cfg(region="USA", decay=4, neutralization="INDUSTRY"),
        _cfg(region="CHN", decay=4, neutralization="INDUSTRY"),
        _cfg(region="USA", decay=0, neutralization="INDUSTRY"),
        _cfg(region="USA", decay=4, neutralization="SUBINDUSTRY"),
        _cfg(region="USA", decay=4, neutralization="INDUSTRY",
             universe="TOP1000"),
        _cfg(region="USA", decay=4, neutralization="INDUSTRY", delay=0),
    ]
    keys = set()
    for c in configs:
        for s in ("feasibility", "sharpe"):
            keys.add(c.cache_key(fp, s))
    assert len(keys) == 12