#!/usr/bin/env python3
"""Try to build 5 seed ASTs from the reference alphas.

Read-only. Constructs each seed AST in memory, renders it, and prints
the expression. If a seed fails to build, the traceback shows which
operator or field is missing from the pool.

Usage:
    ~/python/.venv/bin/python scripts/try_seeds.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from wqgp.ast import ASTNode, render
from wqgp.field import FieldDef
from wqgp.operator import OperatorDef
from wqgp.pool import Pool
from wqgp.types import DataType, ParamDomain


def _f(pool: Pool, name: str) -> ASTNode:
    fd: FieldDef = pool.fields.by_id(name)
    return ASTNode.field_ref(fd)


def _op(pool: Pool, name: str) -> OperatorDef:
    return pool.operators.by_name(name)


def _int_param(pool: Pool, op_name: str, param_name: str) -> ParamDomain:
    op = _op(pool, op_name)
    return op.param_domain(param_name)


def seed_1_debt_assets_ratio(pool: Pool) -> ASTNode:
    """winsorize(ts_sum(debt, 30)) / winsorize(ts_backfill(assets, 10))

    Sharpe 1.62, 2 fields.
    """
    debt = _f(pool, "debt")
    assets = _f(pool, "assets")
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


def seed_2_avf4_close_spread(pool: Pool) -> ASTNode:
    """rank(winsorize(ts_backfill(anl4_afv4_dts_spe, 120)))
       + (-1 * rank(ts_av_diff(close, 5)))

    Sharpe 1.71, 2 fields. `+(-1*x)` rendered as subtract.
    """
    dts_spe = _f(pool, "anl4_afv4_dts_spe")
    close = _f(pool, "close")
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


def seed_3_close_returns_vol(pool: Pool) -> ASTNode:
    """-1 * (winsorize(ts_delta(close, 10))
            + zscore(ts_std_dev(returns, 20)))

    Sharpe 1.68, 2 fields. Entire expression is negated.
    """
    close = _f(pool, "close")
    returns = _f(pool, "returns")
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


def seed_4_debt_assets_neg(pool: Pool) -> ASTNode:
    """-1 * (reverse(ts_sum(debt, 30)) / winsorize(ts_backfill(assets, 3)))

    Sharpe 1.62, 2 fields. Note: outer -1 applied to ratio, then
    `reverse` as negation wrapper.
    """
    debt = _f(pool, "debt")
    assets = _f(pool, "assets")
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


def seed_5_three_fields(pool: Pool) -> ASTNode:
    """rank(zscore(debt_winsorize / assets_winsorize))
       + (-1 * rank(ts_av_diff(close_winsorize, 5)))

    Sharpe 1.82, 3 fields. Combines templates 1 and 2.
    """
    debt = _f(pool, "debt")
    assets = _f(pool, "assets")
    close = _f(pool, "close")
    ts_backfill = _op(pool, "ts_backfill")
    ts_av_diff = _op(pool, "ts_av_diff")
    winsorize = _op(pool, "winsorize")
    rank = _op(pool, "rank")
    zscore = _op(pool, "zscore")
    divide = _op(pool, "divide")
    subtract = _op(pool, "subtract")

    debt_bf = ASTNode.call(ts_backfill, (debt,), params=(120,))
    assets_bf = ASTNode.call(ts_backfill, (assets,), params=(120,))
    close_bf = ASTNode.call(ts_backfill, (close,), params=(120,))

    debt_win = ASTNode.call(winsorize, (debt_bf,))
    assets_win = ASTNode.call(winsorize, (assets_bf,))
    close_win = ASTNode.call(winsorize, (close_bf,))

    ratio = ASTNode.call(divide, (debt_win, assets_win))
    z = ASTNode.call(zscore, (ratio,))
    left = ASTNode.call(rank, (z,))

    ts_av = ASTNode.call(ts_av_diff, (close_win,), params=(5,))
    right = ASTNode.call(rank, (ts_av,))

    return ASTNode.call(subtract, (left, right))


SEEDS = [
    ("debt_assets_ratio", seed_1_debt_assets_ratio),
    ("avf4_close_spread", seed_2_avf4_close_spread),
    ("close_returns_vol", seed_3_close_returns_vol),
    ("debt_assets_neg", seed_4_debt_assets_neg),
    ("three_fields", seed_5_three_fields),
]


def main() -> int:
    pool = Pool.from_file("pool.json")
    print(f"pool: {len(pool.fields)} fields, {len(pool.operators)} operators")
    print()

    success = 0
    failure = 0
    for name, builder in SEEDS:
        try:
            ast = builder(pool)
        except Exception as e:
            print(f"[FAIL] {name}: {type(e).__name__}: {e}")
            failure += 1
            continue

        expr = render(ast)
        print(f"[OK]   {name}")
        print(f"       fp = {ast.fingerprint()[:16]}...")
        print(f"       {expr}")
        print()
        success += 1

    print()
    print(f"success: {success} / {len(SEEDS)}")
    print(f"failure: {failure} / {len(SEEDS)}")
    return 0 if failure == 0 else 1


if __name__ == "__main__":
    sys.exit(main())