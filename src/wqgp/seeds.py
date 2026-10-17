"""Hand-curated seed ASTs derived from historical submittable alphas.

Design decisions:
- Each seed is a Python function that constructs an AST directly
  from a Pool, not a string to be parsed. See D-168.
- Seeds are grouped into a registry (SEEDS) with metadata (source
  alpha_id, historical sharpe, note). See D-169.
- Construction failures are isolated per seed: one bad seed is
  logged and skipped; the rest continue. If EVERY seed fails, that
  indicates a systemic pool incompatibility (missing fields,
  changed operator signatures) and raises RuntimeError. See D-169.

All seeds in this module were manually translated from
`alphas.db` entries that passed every BRAIN check except
SELF_CORRELATION. The source alpha_id and historical sharpe are
preserved so a future analysis can correlate seed lineage with
downstream offspring quality.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

from wqgp.ast import ASTNode
from wqgp.field import FieldDef
from wqgp.operator import OperatorDef
from wqgp.pool import Pool
from wqgp.types import ParamDomain


log = logging.getLogger("wqgp.seeds")


# --- helpers ---------------------------------------------------------------

def _field(pool: Pool, field_id: str) -> ASTNode:
    fd: FieldDef = pool.fields.by_id(field_id)
    return ASTNode.field_ref(fd)


def _op(pool: Pool, name: str) -> OperatorDef:
    return pool.operators.by_name(name)


def _param(pool: Pool, op_name: str, param_name: str) -> ParamDomain:
    return _op(pool, op_name).param_domain(param_name)


# --- individual seed builders ---------------------------------------------

def _build_debt_assets_ratio(pool: Pool) -> ASTNode:
    """winsorize(ts_sum(debt, 30)) / winsorize(ts_backfill(assets, 10))

    Sharpe 1.62, 2 fields. Classic leverage-style ratio.
    """
    debt = _field(pool, "debt")
    assets = _field(pool, "assets")
    ts_sum = _op(pool, "ts_sum")
    ts_backfill = _op(pool, "ts_backfill")
    winsorize = _op(pool, "winsorize")
    divide = _op(pool, "divide")

    debt_sum = ASTNode.call(ts_sum, (debt,), params=(30,))
    assets_bf = ASTNode.call(ts_backfill, (assets,), params=(10,))
    return ASTNode.call(
        divide,
        (
            ASTNode.call(winsorize, (debt_sum,)),
            ASTNode.call(winsorize, (assets_bf,)),
        ),
    )


def _build_avf4_close_spread(pool: Pool) -> ASTNode:
    """rank(winsorize(ts_backfill(anl4_afv4_dts_spe, 120)))
       - rank(ts_av_diff(close, 5))

    Sharpe 1.71, 2 fields. Analyst-estimate vs price divergence.
    """
    dts_spe = _field(pool, "anl4_afv4_dts_spe")
    close = _field(pool, "close")
    ts_backfill = _op(pool, "ts_backfill")
    ts_av_diff = _op(pool, "ts_av_diff")
    winsorize = _op(pool, "winsorize")
    rank = _op(pool, "rank")
    subtract = _op(pool, "subtract")

    left = ASTNode.call(
        rank,
        (ASTNode.call(winsorize, (
            ASTNode.call(ts_backfill, (dts_spe,), params=(120,)),
        )),),
    )
    right = ASTNode.call(
        rank,
        (ASTNode.call(ts_av_diff, (close,), params=(5,)),),
    )
    return ASTNode.call(subtract, (left, right))


def _build_close_returns_vol(pool: Pool) -> ASTNode:
    """reverse(add(winsorize(ts_delta(close, 10)),
                  zscore(ts_std_dev(returns, 20))))

    Sharpe 1.68, 2 fields. Signed momentum + volatility composite.
    """
    close = _field(pool, "close")
    returns = _field(pool, "returns")
    ts_delta = _op(pool, "ts_delta")
    ts_std_dev = _op(pool, "ts_std_dev")
    winsorize = _op(pool, "winsorize")
    zscore = _op(pool, "zscore")
    add = _op(pool, "add")
    reverse = _op(pool, "reverse")

    left = ASTNode.call(
        winsorize,
        (ASTNode.call(ts_delta, (close,), params=(10,)),),
    )
    right = ASTNode.call(
        zscore,
        (ASTNode.call(ts_std_dev, (returns,), params=(20,)),),
    )
    return ASTNode.call(reverse, (ASTNode.call(add, (left, right)),))


def _build_debt_assets_neg(pool: Pool) -> ASTNode:
    """reverse(divide(reverse(ts_sum(debt, 30)),
                     winsorize(ts_backfill(assets, 3))))

    Sharpe 1.62, 2 fields. Inverted leverage ratio.
    """
    debt = _field(pool, "debt")
    assets = _field(pool, "assets")
    ts_sum = _op(pool, "ts_sum")
    ts_backfill = _op(pool, "ts_backfill")
    winsorize = _op(pool, "winsorize")
    divide = _op(pool, "divide")
    reverse = _op(pool, "reverse")

    debt_sum = ASTNode.call(ts_sum, (debt,), params=(30,))
    assets_bf = ASTNode.call(ts_backfill, (assets,), params=(3,))
    return ASTNode.call(
        reverse,
        (
            ASTNode.call(
                divide,
                (
                    ASTNode.call(reverse, (debt_sum,)),
                    ASTNode.call(winsorize, (assets_bf,)),
                ),
            ),
        ),
    )


def _build_three_fields(pool: Pool) -> ASTNode:
    """subtract(
         rank(zscore(divide(winsorize(ts_backfill(debt, 120)),
                             winsorize(ts_backfill(assets, 120))))),
         rank(ts_av_diff(winsorize(ts_backfill(close, 120)), 5)))

    Sharpe 1.82, 3 fields. The best submittable alpha observed.
    Combines a leverage ratio with a price-divergence term.
    """
    debt = _field(pool, "debt")
    assets = _field(pool, "assets")
    close = _field(pool, "close")
    ts_backfill = _op(pool, "ts_backfill")
    ts_av_diff = _op(pool, "ts_av_diff")
    winsorize = _op(pool, "winsorize")
    rank = _op(pool, "rank")
    zscore = _op(pool, "zscore")
    divide = _op(pool, "divide")
    subtract = _op(pool, "subtract")

    debt_win = ASTNode.call(
        winsorize,
        (ASTNode.call(ts_backfill, (debt,), params=(120,)),),
    )
    assets_win = ASTNode.call(
        winsorize,
        (ASTNode.call(ts_backfill, (assets,), params=(120,)),),
    )
    close_win = ASTNode.call(
        winsorize,
        (ASTNode.call(ts_backfill, (close,), params=(120,)),),
    )

    ratio = ASTNode.call(divide, (debt_win, assets_win))
    z = ASTNode.call(zscore, (ratio,))
    left = ASTNode.call(rank, (z,))

    ts_av = ASTNode.call(ts_av_diff, (close_win,), params=(5,))
    right = ASTNode.call(rank, (ts_av,))

    return ASTNode.call(subtract, (left, right))


# --- registry --------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class SeedSpec:
    """Metadata + builder for one seed."""
    seed_id: str
    builder: Callable[[Pool], ASTNode]
    source_alpha_id: str
    historical_sharpe: float
    note: str = ""


SEEDS: tuple[SeedSpec, ...] = (
    SeedSpec(
        seed_id="debt_assets_ratio",
        builder=_build_debt_assets_ratio,
        source_alpha_id="",   # fill in from alphas.db if desired
        historical_sharpe=1.62,
        note="leverage ratio",
    ),
    SeedSpec(
        seed_id="avf4_close_spread",
        builder=_build_avf4_close_spread,
        source_alpha_id="",
        historical_sharpe=1.71,
        note="analyst estimate vs price",
    ),
    SeedSpec(
        seed_id="close_returns_vol",
        builder=_build_close_returns_vol,
        source_alpha_id="",
        historical_sharpe=1.68,
        note="momentum + volatility",
    ),
    SeedSpec(
        seed_id="debt_assets_neg",
        builder=_build_debt_assets_neg,
        source_alpha_id="",
        historical_sharpe=1.62,
        note="inverted leverage",
    ),
    SeedSpec(
        seed_id="three_fields",
        builder=_build_three_fields,
        source_alpha_id="",
        historical_sharpe=1.82,
        note="best observed: 3 fields",
    ),
)


# --- public API ------------------------------------------------------------

def load_seeds(
    pool: Pool,
) -> list[tuple[SeedSpec, ASTNode]]:
    """Build every seed. Failed seeds are skipped with a warning.

    Raises RuntimeError if EVERY seed fails, because that indicates a
    pool-level incompatibility (missing fields, changed operator
    signatures) that the caller must know about. See D-169.
    """
    built: list[tuple[SeedSpec, ASTNode]] = []
    failures: list[tuple[str, str]] = []

    for spec in SEEDS:
        try:
            ast = spec.builder(pool)
        except Exception as e:  # noqa: BLE001 -- per-seed isolation
            msg = f"{type(e).__name__}: {e}"
            log.warning(
                "[seeds] builder %s failed: %s", spec.seed_id, msg,
            )
            failures.append((spec.seed_id, msg))
            continue
        built.append((spec, ast))

    if not built:
        details = "; ".join(
            f"{sid}: {msg}" for sid, msg in failures
        )
        raise RuntimeError(
            f"all {len(SEEDS)} seed builders failed. This means the "
            f"pool is incompatible with the seed set (missing fields "
            f"or changed operator signatures). Details: {details}"
        )

    if failures:
        log.warning(
            "[seeds] %d / %d seeds built; %d skipped",
            len(built), len(SEEDS), len(failures),
        )
    return built