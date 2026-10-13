"""Tests for budget tracking. Phase 2 Step 1 gate."""
from __future__ import annotations

import time

import pytest

from wqgp.budget import Budget, BudgetExhausted, BudgetTracker


# --- Budget validation ---

def test_budget_rejects_negative_total() -> None:
    with pytest.raises(ValueError):
        Budget(total_simulations=-1)


def test_budget_rejects_negative_per_generation() -> None:
    with pytest.raises(ValueError):
        Budget(total_simulations=10, per_generation_simulations=-1)


def test_budget_rejects_negative_wall_time() -> None:
    with pytest.raises(ValueError):
        Budget(total_simulations=10, wall_time_seconds=-1.0)


# --- Basic consumption ---

def test_consume_simulation_increments() -> None:
    t = BudgetTracker(Budget(total_simulations=10))
    t.consume_simulation()
    t.consume_simulation(3)
    assert t.total_simulations_used == 4
    assert t.generation_simulations_used == 4
    assert t.remaining_total_simulations() == 6


def test_consume_zero_is_noop() -> None:
    t = BudgetTracker(Budget(total_simulations=10))
    t.consume_simulation(0)
    assert t.total_simulations_used == 0


def test_consume_negative_rejected() -> None:
    t = BudgetTracker(Budget(total_simulations=10))
    with pytest.raises(ValueError):
        t.consume_simulation(-1)


# --- Total budget hard cap ---

def test_total_exhaustion_raises() -> None:
    t = BudgetTracker(Budget(total_simulations=3))
    t.consume_simulation(3)
    with pytest.raises(BudgetExhausted):
        t.consume_simulation(1)


def test_total_exhaustion_leaves_state_unchanged() -> None:
    t = BudgetTracker(Budget(total_simulations=3))
    t.consume_simulation(2)
    with pytest.raises(BudgetExhausted):
        t.consume_simulation(5)
    assert t.total_simulations_used == 2  # no partial debit


def test_is_exhausted_after_total() -> None:
    t = BudgetTracker(Budget(total_simulations=2))
    assert not t.is_exhausted()
    t.consume_simulation(2)
    assert t.is_exhausted()


# --- Per-generation cap ---

def test_per_generation_cap() -> None:
    t = BudgetTracker(
        Budget(total_simulations=100, per_generation_simulations=3)
    )
    t.consume_simulation(3)
    with pytest.raises(BudgetExhausted):
        t.consume_simulation(1)


def test_start_generation_resets_per_gen_counter() -> None:
    t = BudgetTracker(
        Budget(total_simulations=100, per_generation_simulations=3)
    )
    t.consume_simulation(3)
    t.start_generation()
    assert t.generation_simulations_used == 0
    assert t.total_simulations_used == 3
    t.consume_simulation(3)  # fine again
    assert t.total_simulations_used == 6


def test_no_per_gen_limit_allows_unbounded_per_generation() -> None:
    t = BudgetTracker(Budget(total_simulations=10))
    t.consume_simulation(10)
    assert t.remaining_generation_simulations() is None


# --- Wall clock ---

def test_wall_time_zero_exhausts_immediately() -> None:
    t = BudgetTracker(Budget(total_simulations=10, wall_time_seconds=0.0))
    with pytest.raises(BudgetExhausted):
        t.consume_simulation(1)


def test_wall_time_elapsed_triggers() -> None:
    t = BudgetTracker(Budget(total_simulations=10, wall_time_seconds=0.01))
    time.sleep(0.02)
    assert t.is_exhausted()
    with pytest.raises(BudgetExhausted):
        t.consume_simulation(1)


# --- Freeze ---

def test_freeze_blocks_all_consumption() -> None:
    t = BudgetTracker(Budget(total_simulations=10))
    t.freeze()
    assert t.frozen
    with pytest.raises(BudgetExhausted):
        t.consume_simulation(1)
    with pytest.raises(BudgetExhausted):
        t.start_generation()


# --- Recovery ---

def test_recovery_restores_counters() -> None:
    t = BudgetTracker(
        Budget(total_simulations=10, per_generation_simulations=5),
        total_simulations_used=4,
        generation_simulations_used=2,
        elapsed_offset_seconds=1.5,
    )
    assert t.total_simulations_used == 4
    assert t.generation_simulations_used == 2
    assert t.elapsed_seconds() >= 1.5


def test_recovery_rejects_out_of_range_counters() -> None:
    with pytest.raises(ValueError):
        BudgetTracker(
            Budget(total_simulations=3),
            total_simulations_used=4,
        )


# --- Snapshot ---

def test_snapshot_round_trip_fields() -> None:
    t = BudgetTracker(Budget(total_simulations=10, wall_time_seconds=60.0))
    t.consume_simulation(3)
    s = t.snapshot()
    assert s.total_simulations_limit == 10
    assert s.total_simulations_used == 3
    assert s.generation_simulations_used == 3
    assert s.wall_time_limit_seconds == 60.0
    assert not s.frozen


def test_snapshot_reflects_freeze() -> None:
    t = BudgetTracker(Budget(total_simulations=10))
    t.freeze()
    assert t.snapshot().frozen