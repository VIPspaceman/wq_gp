"""Tests for fitness functions. D-163 gate."""
from __future__ import annotations

import pytest

from wqgp.alpha_registry import AlphaCheck
from wqgp.brain.client import (
    default_fitness,
    feasibility_fitness,
    sharpe_only_fitness,
)


def _chk(
    name: str,
    result: str,
    limit: float | None = None,
    value: float | None = None,
) -> AlphaCheck:
    return AlphaCheck(
        name=name, result=result, limit=limit, value=value,
    )


# --- sharpe_only_fitness ----------------------------------------------------

def test_sharpe_only_returns_sharpe() -> None:
    assert sharpe_only_fitness({"sharpe": 1.5}) == 1.5


def test_sharpe_only_none_when_missing() -> None:
    assert sharpe_only_fitness({}) is None


def test_sharpe_only_ignores_checks() -> None:
    checks = (_chk("LOW_SUB_UNIVERSE_SHARPE", "FAIL", 0.69, 0.30),)
    assert sharpe_only_fitness({"sharpe": 1.5}, checks) == 1.5


def test_sharpe_only_rejects_nan() -> None:
    assert sharpe_only_fitness({"sharpe": float("nan")}) is None


def test_sharpe_only_rejects_bool() -> None:
    assert sharpe_only_fitness({"sharpe": True}) is None


# --- feasibility_fitness: no checks (legacy) -------------------------------

def test_no_checks_returns_raw_sharpe() -> None:
    assert feasibility_fitness({"sharpe": 1.59}) == 1.59


def test_empty_checks_returns_raw_sharpe() -> None:
    assert feasibility_fitness({"sharpe": 1.59}, ()) == 1.59


def test_none_sharpe_returns_none() -> None:
    assert feasibility_fitness({}, None) is None


def test_nan_sharpe_returns_none() -> None:
    assert feasibility_fitness({"sharpe": float("nan")}, None) is None


def test_bool_sharpe_returns_none() -> None:
    assert feasibility_fitness({"sharpe": True}, None) is None


# --- v3: all PASS tier (sharpe + 4.0) --------------------------------------

def test_all_pass_adds_bonus() -> None:
    checks = (
        _chk("LOW_SHARPE", "PASS", 1.25, 1.30),
        _chk("LOW_FITNESS", "PASS", 1.0, 1.10),
    )
    result = feasibility_fitness({"sharpe": 1.30}, checks)
    assert result == pytest.approx(1.30 + 4.0)


def test_all_pass_preserves_sharpe_difference() -> None:
    """The whole point of v3: within the PASS tier, sharpe differences
    are NOT compressed. 1.30 vs 0.50 must differ by 0.80, not 0.12."""
    checks_hi = (_chk("LOW_SHARPE", "PASS", 1.25, 1.30),)
    checks_lo = (_chk("LOW_SHARPE", "PASS", 1.25, 0.50),)
    a = feasibility_fitness({"sharpe": 1.30}, checks_hi)
    b = feasibility_fitness({"sharpe": 0.50}, checks_lo)
    assert a is not None and b is not None
    assert a - b == pytest.approx(0.80)


def test_all_pass_ignores_self_correlation_pending() -> None:
    checks = (
        _chk("LOW_SHARPE", "PASS", 1.25, 1.30),
        _chk("SELF_CORRELATION", "PENDING"),
    )
    result = feasibility_fitness({"sharpe": 1.30}, checks)
    assert result == pytest.approx(1.30 + 4.0)


# --- v3: pending tier (sharpe + 2.0) ---------------------------------------

def test_pending_gets_half_bonus() -> None:
    checks = (
        _chk("LOW_SHARPE", "PASS", 1.25, 1.30),
        _chk("LOW_SUB_UNIVERSE_SHARPE", "PENDING"),
    )
    result = feasibility_fitness({"sharpe": 1.30}, checks)
    assert result == pytest.approx(1.30 + 2.0)


# --- v3: fail tier (sharpe - penalties) ------------------------------------

def test_one_fail_with_numeric_gap() -> None:
    """Concrete case from the A/B: sharpe=1.35, turnover=0.915 fails
    HIGH_TURNOVER (limit 0.7). gap = 0.215/0.7 = 0.307.
    score = 1.35 - 0.35 - 0.5*0.307 = 0.846."""
    checks = (
        _chk("HIGH_TURNOVER", "FAIL", 0.7, 0.915),
    )
    result = feasibility_fitness({"sharpe": 1.35}, checks)
    assert result == pytest.approx(1.35 - 0.35 - 0.5 * (0.215 / 0.7))


def test_two_fails_compound_penalty() -> None:
    checks = (
        _chk("LOW_SHARPE", "FAIL", 1.25, 0.78),
        _chk("LOW_FITNESS", "FAIL", 1.0, 0.60),
    )
    result = feasibility_fitness({"sharpe": 0.78}, checks)
    assert result is not None
    # Two FAIL penalties applied
    assert result < 0.78


def test_fail_without_numeric_value_uses_unit_penalty() -> None:
    """Boolean checks (CONCENTRATED_WEIGHT) contribute 0.35 + 0.5*1.0."""
    checks = (
        _chk("CONCENTRATED_WEIGHT", "FAIL"),
    )
    result = feasibility_fitness({"sharpe": 1.5}, checks)
    assert result == pytest.approx(1.5 - 0.35 - 0.5)


def test_violation_capped_at_two() -> None:
    """A wildly-out-of-range check should not annihilate the score."""
    checks = (
        _chk("LOW_SHARPE", "FAIL", 1.25, -5.0),  # gap = 6.25/1.25 = 5.0
    )
    result = feasibility_fitness({"sharpe": 1.5}, checks)
    # gap capped at 2.0: penalty = 0.35 + 0.5*2.0 = 1.35
    assert result == pytest.approx(1.5 - 1.35)


# --- v3: pass > fail ordering ----------------------------------------------

def test_submittable_beats_high_sharpe_failure() -> None:
    """The core design goal of v3: a mediocre submittable alpha beats
    a higher-sharpe alpha that fails a check. This was the failure
    mode of the old feasibility formula (D-157)."""
    pass_checks = (
        _chk("LOW_SHARPE", "PASS", 1.25, 1.30),
    )
    fail_checks = (
        _chk("HIGH_TURNOVER", "FAIL", 0.7, 0.92),
    )
    submittable = feasibility_fitness({"sharpe": 0.50}, pass_checks)
    failing = feasibility_fitness({"sharpe": 1.35}, fail_checks)
    assert submittable is not None and failing is not None
    assert submittable > failing


def test_pass_0_5_beats_fail_at_3_0() -> None:
    """Even a PASS with very low sharpe should beat a FAIL with very
    high sharpe. This is what makes the tier 'soft' but still
    reliable."""
    pass_checks = (_chk("LOW_SHARPE", "PASS", 1.25, 0.50),)
    fail_checks = (_chk("HIGH_TURNOVER", "FAIL", 0.7, 0.75),)
    a = feasibility_fitness({"sharpe": 0.50}, pass_checks)
    b = feasibility_fitness({"sharpe": 3.00}, fail_checks)
    assert a is not None and b is not None
    assert a > b


def test_fail_tier_still_orders_by_sharpe() -> None:
    """Within the FAIL tier, higher sharpe still wins."""
    checks_a = (_chk("HIGH_TURNOVER", "FAIL", 0.7, 0.92),)
    checks_b = (_chk("HIGH_TURNOVER", "FAIL", 0.7, 0.92),)
    a = feasibility_fitness({"sharpe": 1.35}, checks_a)
    b = feasibility_fitness({"sharpe": 0.50}, checks_b)
    assert a is not None and b is not None
    assert a > b


# --- default alias ---------------------------------------------------------

def test_default_alias_is_v3() -> None:
    checks = (_chk("LOW_SHARPE", "PASS", 1.25, 1.30),)
    a = default_fitness({"sharpe": 1.30}, checks)
    b = feasibility_fitness({"sharpe": 1.30}, checks)
    assert a == b