"""Tests for checkpoint store. Phase 2 Step 1 gate."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import pytest

from wqgp.budget import Budget, BudgetTracker
from wqgp.checkpoint import (
    SCHEMA_VERSION,
    Checkpoint,
    CheckpointStore,
    CorruptCheckpointError,
    IncompatibleSchemaError,
)


@pytest.fixture
def store(tmp_path: Path) -> CheckpointStore:
    return CheckpointStore(tmp_path / "ckpt")


def _make_cp(
    run_id: str = "run_a",
    generation: int = 0,
    rng: random.Random | None = None,
    payload: dict | None = None,
    note: str = "",
) -> Checkpoint:
    rng = rng or random.Random(42)
    tracker = BudgetTracker(Budget(total_simulations=100))
    tracker.consume_simulation(7)
    return Checkpoint.capture(
        run_id=run_id,
        generation=generation,
        budget=tracker.snapshot(),
        rng=rng,
        payload=payload if payload is not None else {"pop": [1, 2, 3]},
        note=note,
    )


# --- save / load round-trip ---

def test_save_and_load_roundtrip(store: CheckpointStore) -> None:
    cp = _make_cp(generation=3, note="hello")
    path = store.save(cp)
    assert path.exists()

    loaded = store.load("run_a", 3)
    assert loaded.run_id == "run_a"
    assert loaded.generation == 3
    assert loaded.note == "hello"
    assert loaded.payload == {"pop": [1, 2, 3]}
    assert loaded.budget.total_simulations_used == 7
    assert loaded.schema_version == SCHEMA_VERSION


def test_save_leaves_no_tmp_file(store: CheckpointStore) -> None:
    store.save(_make_cp())
    leftovers = list(store.root.rglob("*.tmp"))
    assert leftovers == []


# --- latest pointer ---

def test_load_latest_returns_none_on_empty(store: CheckpointStore) -> None:
    assert store.load_latest("nope") is None


def test_load_latest_returns_highest(store: CheckpointStore) -> None:
    store.save(_make_cp(generation=0, payload={"g": 0}))
    store.save(_make_cp(generation=1, payload={"g": 1}))
    store.save(_make_cp(generation=2, payload={"g": 2}))
    latest = store.load_latest("run_a")
    assert latest is not None
    assert latest.generation == 2
    assert latest.payload == {"g": 2}


def test_load_latest_falls_back_without_pointer(store: CheckpointStore) -> None:
    store.save(_make_cp(generation=5))
    (store.root / "run_a" / "latest.txt").unlink()
    latest = store.load_latest("run_a")
    assert latest is not None
    assert latest.generation == 5


def test_load_latest_raises_on_stale_pointer(store: CheckpointStore) -> None:
    store.save(_make_cp(generation=0))
    # simulate the step file being deleted out from under the pointer
    (store.root / "run_a" / "step_000000.json").unlink()
    with pytest.raises(CorruptCheckpointError):
        store.load_latest("run_a")


# --- RNG reproducibility (D-016) ---

def test_rng_state_round_trip_is_reproducible(store: CheckpointStore) -> None:
    rng = random.Random(12345)
    # advance RNG to a non-trivial state
    for _ in range(50):
        rng.random()

    # Snapshot the exact state we want restored, THEN advance a *separate*
    # Random from that snapshot to compute the expected next 5 draws.
    # Do not read expected_next from `rng` before capturing the checkpoint:
    # those reads would advance `rng` past the state we mean to store.
    state = rng.getstate()
    expected_next = [rng.random() for _ in range(5)]

    snapshot_rng = random.Random()
    snapshot_rng.setstate(state)
    cp = _make_cp(rng=snapshot_rng)
    store.save(cp)

    loaded = store.load("run_a", 0)
    restored = loaded.restore_rng()
    got_next = [restored.random() for _ in range(5)]
    assert got_next == expected_next

# --- schema version (D-018) ---

def test_schema_version_mismatch_rejected_on_construct() -> None:
    with pytest.raises(IncompatibleSchemaError):
        Checkpoint(
            run_id="r",
            generation=0,
            budget=BudgetTracker(Budget(1)).snapshot(),
            rng_state=[3, [0] * 625, None],
            created_at=0.0,
            payload={},
            schema_version=999,
        )


def test_schema_version_mismatch_rejected_on_load(store: CheckpointStore) -> None:
    store.save(_make_cp())
    path = store.root / "run_a" / "step_000000.json"
    data = json.loads(path.read_text())
    data["schema_version"] = 999
    path.write_text(json.dumps(data))
    with pytest.raises(IncompatibleSchemaError):
        store.load("run_a", 0)


# --- corrupt file handling ---

def test_corrupt_json_raises(store: CheckpointStore) -> None:
    store.save(_make_cp())
    path = store.root / "run_a" / "step_000000.json"
    path.write_text("{ not json")
    with pytest.raises(CorruptCheckpointError):
        store.load("run_a", 0)


def test_missing_file_raises_filenotfound(store: CheckpointStore) -> None:
    with pytest.raises(FileNotFoundError):
        store.load("run_a", 99)


def test_malformed_budget_raises_corrupt(store: CheckpointStore) -> None:
    store.save(_make_cp())
    path = store.root / "run_a" / "step_000000.json"
    data = json.loads(path.read_text())
    data["budget"] = {"unexpected": True}
    path.write_text(json.dumps(data))
    with pytest.raises(CorruptCheckpointError):
        store.load("run_a", 0)


# --- run_id validation (D-018) ---

@pytest.mark.parametrize(
    "bad_id",
    ["", " ", "run a", "run/a", "run\\a", "..", "../etc", "run..x"],
)
def test_illegal_run_id_rejected(store: CheckpointStore, bad_id: str) -> None:
    with pytest.raises(ValueError):
        store.save(_make_cp(run_id=bad_id))


# --- discovery ---

def test_list_generations_sorted(store: CheckpointStore) -> None:
    for g in [5, 0, 3, 1]:
        store.save(_make_cp(generation=g))
    assert store.list_generations("run_a") == [0, 1, 3, 5]


def test_list_generations_empty_for_unknown(store: CheckpointStore) -> None:
    assert store.list_generations("ghost") == []


def test_list_runs(store: CheckpointStore) -> None:
    store.save(_make_cp(run_id="alpha", generation=0))
    store.save(_make_cp(run_id="beta", generation=0))
    assert store.list_runs() == ["alpha", "beta"]


# --- maintenance ---

def test_delete_run(store: CheckpointStore) -> None:
    store.save(_make_cp(run_id="doomed"))
    store.delete_run("doomed")
    assert "doomed" not in store.list_runs()


def test_delete_run_is_idempotent(store: CheckpointStore) -> None:
    store.delete_run("never_existed")  # must not raise


# --- payload is opaque (D-014) ---

def test_payload_can_hold_nested_structures(store: CheckpointStore) -> None:
    payload = {
        "population": [{"expr": "add(close, 1)", "fitness": 0.5}],
        "cache": {"hash": "abc"},
        "meta": {"gen": 1, "note": None},
    }
    store.save(_make_cp(payload=payload))
    loaded = store.load("run_a", 0)
    assert loaded.payload == payload