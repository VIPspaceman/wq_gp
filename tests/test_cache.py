"""Tests for EvaluationCache. Phase 4 Step 8 Round 1 gate."""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from wqgp.cache import (
    CacheEntry,
    CacheStatus,
    ErrorType,
    EvaluationCache,
    IncompatibleCacheSchemaError,
)


# --- fixtures ---------------------------------------------------------------

@pytest.fixture
def cache(tmp_path: Path):
    c = EvaluationCache(tmp_path / "cache.db", ttl_days=90)
    yield c
    c.close()


def _success(
    key: str = "k1",
    fp: str = "fp1",
    sh: str = "sh1",
    sj: str = "{}",
    fitness: float = 1.5,
) -> CacheEntry:
    return CacheEntry(
        cache_key=key,
        fingerprint=fp,
        settings_hash=sh,
        settings_json=sj,
        status=CacheStatus.SUCCESS,
        fitness=fitness,
        metrics=(("sharpe", 1.5), ("turnover", 0.3)),
    )


def _failed(
    key: str = "k2",
    fp: str = "fp2",
    sh: str = "sh1",
    sj: str = "{}",
) -> CacheEntry:
    return CacheEntry(
        cache_key=key,
        fingerprint=fp,
        settings_hash=sh,
        settings_json=sj,
        status=CacheStatus.FAILED,
        error="Incompatible unit for input of log",
        error_type=ErrorType.PERMANENT,
    )


# --- CacheEntry invariants --------------------------------------------------

def test_entry_rejects_empty_cache_key() -> None:
    with pytest.raises(ValueError):
        CacheEntry(
            cache_key="", fingerprint="fp", settings_hash="sh",
            settings_json="{}", status=CacheStatus.SUCCESS, fitness=1.0,
        )


def test_entry_rejects_empty_fingerprint() -> None:
    with pytest.raises(ValueError):
        CacheEntry(
            cache_key="k", fingerprint="", settings_hash="sh",
            settings_json="{}", status=CacheStatus.SUCCESS, fitness=1.0,
        )


def test_success_requires_fitness() -> None:
    with pytest.raises(ValueError):
        CacheEntry(
            cache_key="k", fingerprint="fp", settings_hash="sh",
            settings_json="{}", status=CacheStatus.SUCCESS,
        )


def test_success_rejects_error() -> None:
    with pytest.raises(ValueError):
        CacheEntry(
            cache_key="k", fingerprint="fp", settings_hash="sh",
            settings_json="{}", status=CacheStatus.SUCCESS, fitness=1.0,
            error="something",
        )


def test_failed_requires_error() -> None:
    with pytest.raises(ValueError):
        CacheEntry(
            cache_key="k", fingerprint="fp", settings_hash="sh",
            settings_json="{}", status=CacheStatus.FAILED,
        )


def test_failed_requires_permanent_error_type() -> None:
    with pytest.raises(ValueError):
        CacheEntry(
            cache_key="k", fingerprint="fp", settings_hash="sh",
            settings_json="{}", status=CacheStatus.FAILED,
            error="something",
        )


def test_entry_rejects_nan_fitness() -> None:
    with pytest.raises(ValueError):
        CacheEntry(
            cache_key="k", fingerprint="fp", settings_hash="sh",
            settings_json="{}", status=CacheStatus.SUCCESS,
            fitness=float("nan"),
        )


# --- EvaluationCache basic --------------------------------------------------

def test_get_missing_returns_none(cache: EvaluationCache) -> None:
    assert cache.get("nonexistent") is None


def test_get_rejects_empty_key(cache: EvaluationCache) -> None:
    with pytest.raises(ValueError):
        cache.get("")


def test_put_and_get_roundtrip(cache: EvaluationCache) -> None:
    entry = _success()
    cache.put(entry)
    got = cache.get(entry.cache_key)
    assert got is not None
    assert got.fitness == 1.5
    assert got.metrics_dict == {"sharpe": 1.5, "turnover": 0.3}
    assert got.status is CacheStatus.SUCCESS


def test_failed_roundtrip(cache: EvaluationCache) -> None:
    entry = _failed()
    cache.put(entry)
    got = cache.get(entry.cache_key)
    assert got is not None
    assert got.status is CacheStatus.FAILED
    assert got.error_type is ErrorType.PERMANENT
    assert "Incompatible unit" in (got.error or "")


def test_put_replaces_existing(cache: EvaluationCache) -> None:
    cache.put(_success(key="same-key", fitness=1.0))
    cache.put(_success(key="same-key", fitness=2.0))
    got = cache.get("same-key")
    assert got is not None
    assert got.fitness == 2.0


def test_get_bumps_hit_count(cache: EvaluationCache) -> None:
    cache.put(_success())
    cache.get("k1")
    cache.get("k1")
    cache.get("k1")
    got = cache.get("k1")
    assert got is not None
    # The 4th get sees the count from the first 3 hits (0 + 3).
    assert got.hit_count == 3


# --- TTL --------------------------------------------------------------------

def test_expired_entry_is_not_returned(tmp_path: Path) -> None:
    c = EvaluationCache(tmp_path / "c.db", ttl_days=1)
    try:
        entry = _success()
        c.put(entry)
        # Rewind created_at to 2 days ago.
        c._conn.execute(
            "UPDATE evaluation_cache SET created_at = ? "
            "WHERE cache_key = ?",
            (time.time() - 2 * 86400, entry.cache_key),
        )
        c._conn.commit()
        assert c.get(entry.cache_key) is None
    finally:
        c.close()


def test_expired_entry_is_kept_until_clear(tmp_path: Path) -> None:
    c = EvaluationCache(tmp_path / "c.db", ttl_days=1)
    try:
        c.put(_success())
        c._conn.execute(
            "UPDATE evaluation_cache SET created_at = ?",
            (time.time() - 2 * 86400,),
        )
        c._conn.commit()
        s = c.stats()
        assert s.total_entries == 1
        assert s.expired_entries == 1
    finally:
        c.close()


def test_ttl_validation() -> None:
    with pytest.raises(ValueError):
        EvaluationCache(":memory:", ttl_days=0)


# --- stats ------------------------------------------------------------------

def test_stats_empty(cache: EvaluationCache) -> None:
    s = cache.stats()
    assert s.total_entries == 0
    assert s.total_hits == 0
    assert s.avg_hits_per_entry == 0.0
    assert s.oldest_created_at is None


def test_stats_counts_by_status(cache: EvaluationCache) -> None:
    cache.put(_success(key="k1", fp="fp1"))
    cache.put(_success(key="k2", fp="fp2"))
    cache.put(_failed(key="k3", fp="fp3"))
    s = cache.stats()
    assert s.total_entries == 3
    assert s.success_entries == 2
    assert s.failed_entries == 1


def test_stats_total_hits(cache: EvaluationCache) -> None:
    cache.put(_success(key="k1", fp="fp1"))
    cache.put(_success(key="k2", fp="fp2"))
    cache.get("k1")
    cache.get("k1")
    cache.get("k2")
    s = cache.stats()
    assert s.total_hits == 3
    assert abs(s.avg_hits_per_entry - 1.5) < 1e-9


# --- clear ------------------------------------------------------------------

def test_clear_all(cache: EvaluationCache) -> None:
    cache.put(_success(key="k1", fp="fp1"))
    cache.put(_success(key="k2", fp="fp2"))
    removed = cache.clear()
    assert removed == 2
    assert cache.stats().total_entries == 0


def test_clear_before_ts(cache: EvaluationCache) -> None:
    now = time.time()
    cache.put(_success(key="k1", fp="fp1"))
    cache.put(_success(key="k2", fp="fp2"))
    # Rewind k1 to 10 days ago.
    cache._conn.execute(
        "UPDATE evaluation_cache SET created_at = ? WHERE cache_key = 'k1'",
        (now - 10 * 86400,),
    )
    cache._conn.commit()

    removed = cache.clear(before_ts=now - 5 * 86400)
    assert removed == 1
    assert cache.get("k1") is None
    assert cache.get("k2") is not None


def test_clear_empty_returns_zero(cache: EvaluationCache) -> None:
    assert cache.clear() == 0


# --- persistence ------------------------------------------------------------

def test_reopen_preserves_data(tmp_path: Path) -> None:
    path = tmp_path / "c.db"
    with_ = EvaluationCache(path, ttl_days=90)
    with_.put(_success())
    with_.close()

    c2 = EvaluationCache(path, ttl_days=90)
    try:
        got = c2.get("k1")
        assert got is not None
        assert got.fitness == 1.5
    finally:
        c2.close()


def test_schema_version_mismatch_raises(tmp_path: Path) -> None:
    path = tmp_path / "c.db"
    EvaluationCache(path).close()

    import sqlite3
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA user_version=999")
    conn.commit()
    conn.close()

    with pytest.raises(IncompatibleCacheSchemaError):
        EvaluationCache(path)


def test_put_many(cache: EvaluationCache) -> None:
    entries = [_success(key=f"k{i}", fp=f"fp{i}") for i in range(5)]
    n = cache.put_many(entries)
    assert n == 5
    assert len(cache) == 5


def test_put_many_empty_returns_zero(cache: EvaluationCache) -> None:
    assert cache.put_many([]) == 0


def test_len(cache: EvaluationCache) -> None:
    assert len(cache) == 0
    cache.put(_success())
    assert len(cache) == 1