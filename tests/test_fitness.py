"""Tests for fitness functions. D-157 gate."""
from __future__ import annotations

import math

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


# --- feasibility_fitness: no checks -----------------------------------------

def test_feasibility_no_checks_returns_raw_sharpe() -> None:
    """Backward compat: records without checks score as raw sharpe."""
    assert feasibility_fitness({"sharpe": 1.59}) == 1.59


def test_feasibility_empty_checks_returns_raw_sharpe() -> None:
    assert feasibility_fitness({"sharpe": 1.59}, ()) == 1.59


def test_feasibility_none_sharpe_returns_none() -> None:
    assert feasibility_fitness({}, None) is None


# --- feasibility_fitness: all PASS ------------------------------------------

def test_feasibility_all_pass_lands_above_one() -> None:
    checks = (
        _chk("LOW_SHARPE", "PASS", 1.25, 1.59),
        _chk("LOW_FITNESS", "PASS", 1.0, 1.19),
        _chk("LOW_SUB_UNIVERSE_SHARPE", "PASS", 0.69, 0.85),
    )
    result = feasibility_fitness({"sharpe": 1.59}, checks)
    assert result is not None
    assert 1.0 < result <= 1.25


def test_feasibility_pass_with_pending_ignores_self_corr() -> None:
    """SELF_CORRELATION=PENDING does not block PASS tier."""
    checks = (
        _chk("LOW_SHARPE", "PASS", 1.25, 1.59),
        _chk("SELF_CORRELATION", "PENDING"),
    )
    result = feasibility_fitness({"sharpe": 1.59}, checks)
    assert result is not None
    assert result > 1.0


# --- feasibility_fitness: FAIL tier -----------------------------------------

def test_feasibility_one_fail_lands_below_negative_one() -> None:
    """The exact scenario from the D-157 decision record."""
    checks = (
        _chk("LOW_SHARPE", "PASS", 1.25, 1.59),
        _chk("LOW_SUB_UNIVERSE_SHARPE", "FAIL", 0.69, 0.65),
    )
    result = feasibility_fitness({"sharpe": 1.59}, checks)
    assert result is not None
    assert result < -1.0
    # Violation = 0.04/0.69 = 0.058; penalty = 1 + 0.25 + 0.058 - 0.165
    assert result == pytest.approx(-1.143, abs=0.01)


def test_feasibility_fail_beats_high_sharpe_that_also_fails() -> None:
    """Two FAILs: one near threshold, one far. Near wins."""
    near = (
        _chk("LOW_SUB_UNIVERSE_SHARPE", "FAIL", 0.69, 0.68),
    )
    far = (
        _chk("LOW_SUB_UNIVERSE_SHARPE", "FAIL", 0.69, 0.20),
    )
    a = feasibility_fitness({"sharpe": 1.0}, near)
    b = feasibility_fitness({"sharpe": 1.0}, far)
    assert a is not None and b is not None
    assert a > b


def test_feasibility_pass_beats_any_fail() -> None:
    """A mediocre PASS beats a great FAIL. This is the point."""
    pass_chk = (_chk("LOW_SHARPE", "PASS", 1.25, 1.30),)
    fail_chk = (
        _chk("LOW_SUB_UNIVERSE_SHARPE", "FAIL", 0.69, 0.68),
    )
    passer = feasibility_fitness({"sharpe": 1.30}, pass_chk)
    failer = feasibility_fitness({"sharpe": 3.00}, fail_chk)
    assert passer is not None and failer is not None
    assert passer > failer


# --- feasibility_fitness: multiple FAILs ------------------------------------

def test_feasibility_multiple_fails_stack_penalty() -> None:
    one = (_chk("LOW_SUB_UNIVERSE_SHARPE", "FAIL", 0.69, 0.50),)
    two = (
        _chk("LOW_SUB_UNIVERSE_SHARPE", "FAIL", 0.69, 0.50),
        _chk("CONCENTRATED_WEIGHT", "FAIL"),
    )
    a = feasibility_fitness({"sharpe": 1.0}, one)
    b = feasibility_fitness({"sharpe": 1.0}, two)
    assert a is not None and b is not None
    assert a > b


# --- feasibility_fitness: HIGH_ checks --------------------------------------

def test_feasibility_high_turnover_gap_direction() -> None:
    """HIGH_TURNOVER at 0.9 with limit 0.7: violation = 0.2/0.7."""
    checks = (
        _chk("HIGH_TURNOVER", "FAIL", 0.7, 0.9),
    )
    result = feasibility_fitness({"sharpe": 1.0}, checks)
    assert result is not None
    assert result < -1.0


def test_feasibility_low_turnover_gap_direction() -> None:
    """LOW_TURNOVER at 0.005 with limit 0.01: violation = 0.005/0.01."""
    checks = (
        _chk("LOW_TURNOVER", "FAIL", 0.01, 0.005),
    )
    result = feasibility_fitness({"sharpe": 1.0}, checks)
    assert result is not None
    assert result < -1.0


# --- feasibility_fitness: edge cases ----------------------------------------

def test_feasibility_nan_sharpe_returns_none() -> None:
    assert feasibility_fitness({"sharpe": float("nan")}, None) is None
    assert feasibility_fitness({"sharpe": float("inf")}, None) is None


def test_feasibility_bool_metric_returns_none() -> None:
    assert feasibility_fitness({"sharpe": True}, None) is None


def test_feasibility_default_alias_is_feasibility() -> None:
    """default_fitness must be the feasibility version."""
    checks = (
        _chk("LOW_SUB_UNIVERSE_SHARPE", "FAIL", 0.69, 0.65),
    )
    a = default_fitness({"sharpe": 1.59}, checks)
    b = feasibility_fitness({"sharpe": 1.59}, checks)
    assert a == b


def test_feasibility_sharpe_bonus_is_bounded() -> None:
    """The tanh bonus must never let a PASS fall below 1.0."""
    checks = (_chk("LOW_SHARPE", "PASS", 1.25, 100.0),)
    result = feasibility_fitness({"sharpe": 1e6}, checks)
    assert result is not None
    assert result <= 1.25
    # Even with a huge negative sharpe, PASS still lands > 0.75.
    checks_low = (_chk("LOW_SHARPE", "PASS", 1.25, 1.30),)
    low = feasibility_fitness({"sharpe": -1e6}, checks_low)
    assert low is not None
    assert low > 0.7