"""Tests for the seed registry. D-168 / D-169 gate."""
from __future__ import annotations

import pytest

from wqgp.ast import render
from wqgp.ast_ops import count_distinct_fields, simplify_ast
from wqgp.pool import Pool
from wqgp.seeds import SEEDS, SeedSpec, load_seeds


@pytest.fixture
def pool() -> Pool:
    return Pool.from_file("pool.json")


# --- registry metadata -----------------------------------------------------

def test_registry_non_empty() -> None:
    assert len(SEEDS) >= 5


def test_every_seed_has_unique_id() -> None:
    ids = [s.seed_id for s in SEEDS]
    assert len(ids) == len(set(ids))


def test_every_seed_has_positive_historical_sharpe() -> None:
    for spec in SEEDS:
        assert spec.historical_sharpe > 0


def test_seed_spec_is_frozen() -> None:
    with pytest.raises(Exception):
        SEEDS[0].seed_id = "renamed"  # type: ignore[misc]


# --- load_seeds happy path -------------------------------------------------

def test_load_seeds_all_build(pool: Pool) -> None:
    built = load_seeds(pool)
    assert len(built) == len(SEEDS)


def test_load_seeds_each_ast_is_valid(pool: Pool) -> None:
    """Every built AST has been validated by ASTNode.call already; the
    render is a smoke check that it produces a BRAIN-compatible
    expression string."""
    for spec, ast in load_seeds(pool):
        expr = render(ast)
        assert expr, f"empty render for {spec.seed_id}"
        assert "(" in expr and ")" in expr


def test_load_seeds_each_is_multi_field(pool: Pool) -> None:
    for spec, ast in load_seeds(pool):
        n = count_distinct_fields(simplify_ast(ast))
        assert n >= 2, (
            f"seed {spec.seed_id} has only {n} distinct fields after "
            f"simplification"
        )


def test_load_seeds_produce_distinct_fingerprints(pool: Pool) -> None:
    built = load_seeds(pool)
    fps = {ast.fingerprint() for _, ast in built}
    assert len(fps) == len(built), "duplicate fingerprints across seeds"


# --- failure isolation -----------------------------------------------------

def test_one_bad_seed_does_not_block_others(
    pool: Pool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If one builder fails, the rest should still load."""

    class _Boom(SeedSpec):
        pass

    # Replace the first builder with one that always raises.
    from wqgp import seeds as seeds_mod

    original = seeds_mod.SEEDS[0].builder

    def _raise(_pool: Pool) -> None:
        raise ValueError("intentional test failure")

    # Construct a new SEEDS tuple with one broken entry.
    patched = (
        SeedSpec(
            seed_id="broken",
            builder=_raise,
            source_alpha_id="",
            historical_sharpe=1.5,
        ),
        *seeds_mod.SEEDS[1:],
    )
    monkeypatch.setattr(seeds_mod, "SEEDS", patched)

    built = seeds_mod.load_seeds(pool)
    # All non-broken seeds loaded.
    assert len(built) == len(patched) - 1
    seed_ids = {spec.seed_id for spec, _ in built}
    assert "broken" not in seed_ids


def test_all_bad_seeds_raise(
    pool: Pool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If every builder fails, load_seeds raises RuntimeError."""
    from wqgp import seeds as seeds_mod

    def _raise(_pool: Pool) -> None:
        raise ValueError("boom")

    bad = tuple(
        SeedSpec(
            seed_id=f"bad_{i}",
            builder=_raise,
            source_alpha_id="",
            historical_sharpe=1.0,
        )
        for i in range(3)
    )
    monkeypatch.setattr(seeds_mod, "SEEDS", bad)

    with pytest.raises(RuntimeError) as exc_info:
        seeds_mod.load_seeds(pool)

    msg = str(exc_info.value)
    assert "all 3 seed builders failed" in msg
    assert "ValueError" in msg


# --- specific seed shapes --------------------------------------------------

def test_three_fields_seed_has_three_fields(pool: Pool) -> None:
    for spec, ast in load_seeds(pool):
        if spec.seed_id == "three_fields":
            n = count_distinct_fields(simplify_ast(ast))
            assert n == 3
            return
    pytest.fail("seed 'three_fields' not found")


def test_debt_assets_seeds_share_fingerprint_family(pool: Pool) -> None:
    """The three debt-assets seeds are different expressions but all
    use debt and assets. This is a sanity check that the registry is
    well-curated."""
    names = {"debt_assets_ratio", "debt_assets_neg"}
    for spec, ast in load_seeds(pool):
        if spec.seed_id in names:
            assert "debt" in render(ast)
            assert "assets" in render(ast)