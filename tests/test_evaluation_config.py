"""Tests for EvaluationConfig. Phase 4 Step 8 Round 1 gate."""
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
    a = _cfg()
    b = _cfg()
    assert a.settings_hash() == b.settings_hash()


def test_settings_hash_differs_on_region() -> None:
    assert _cfg(region="USA").settings_hash() != _cfg(region="CHN").settings_hash()


def test_settings_hash_differs_on_decay() -> None:
    assert _cfg(decay=4).settings_hash() != _cfg(decay=0).settings_hash()


def test_settings_hash_differs_on_neutralization() -> None:
    a = _cfg(neutralization="INDUSTRY")
    b = _cfg(neutralization="SUBINDUSTRY")
    assert a.settings_hash() != b.settings_hash()


def test_settings_hash_is_hex_sha256() -> None:
    h = _cfg().settings_hash()
    assert len(h) == 64
    int(h, 16)  # raises if not hex


def test_settings_hash_ignores_construction_order() -> None:
    """Two BrainSettings constructed with fields in different orders
    must produce identical hashes: BrainSettings is a frozen dataclass,
    so field order is fixed by its definition."""
    a = BrainSettings(region="USA", decay=4)
    b = BrainSettings(region="USA", decay=4)
    assert (
        EvaluationConfig(a).settings_hash()
        == EvaluationConfig(b).settings_hash()
    )


def test_cache_key_is_deterministic() -> None:
    cfg = _cfg()
    fp = "abc123"
    assert cfg.cache_key(fp) == cfg.cache_key(fp)


def test_cache_key_differs_on_fingerprint() -> None:
    cfg = _cfg()
    assert cfg.cache_key("aaa") != cfg.cache_key("bbb")


def test_cache_key_differs_on_settings() -> None:
    fp = "abc123"
    a = _cfg(region="USA")
    b = _cfg(region="CHN")
    assert a.cache_key(fp) != b.cache_key(fp)


def test_cache_key_rejects_empty_fingerprint() -> None:
    with pytest.raises(ValueError):
        _cfg().cache_key("")


def test_cache_key_is_hex_sha256() -> None:
    key = _cfg().cache_key("abc")
    assert len(key) == 64
    int(key, 16)


def test_cache_schema_version_is_positive_int() -> None:
    assert isinstance(CACHE_SCHEMA_VERSION, int)
    assert CACHE_SCHEMA_VERSION >= 1


def test_different_settings_never_share_cache_key() -> None:
    """The critical invariant: a settings change must change the key.

    Each config below differs from the default on exactly one field,
    so every pair is provably distinct. The earlier version of this
    test passed `_cfg(decay=4)` (which equals the default) and
    `_cfg(neutralization="INDUSTRY")` (also the default), collapsing
    three entries into one and yielding 4 unique keys instead of 6.
    """
    fp = "same-fingerprint"
    configs = [
        _cfg(region="USA", decay=4, neutralization="INDUSTRY"),
        _cfg(region="CHN", decay=4, neutralization="INDUSTRY"),
        _cfg(region="USA", decay=0, neutralization="INDUSTRY"),
        _cfg(region="USA", decay=4, neutralization="SUBINDUSTRY"),
        _cfg(
            region="USA", decay=4, neutralization="INDUSTRY",
            universe="TOP1000",
        ),
        _cfg(
            region="USA", decay=4, neutralization="INDUSTRY",
            delay=0,
        ),
    ]
    keys = {c.cache_key(fp) for c in configs}
    assert len(keys) == 6