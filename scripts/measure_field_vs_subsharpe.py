#!/usr/bin/env python3
"""Measure whether field count correlates with SUB_UNIVERSE_SHARPE failure.

Read-only. Reads alphas.db and config/signatures.json.

Usage:
    ~/python/.venv/bin/python scripts/measure_field_vs_subsharpe.py
"""
from __future__ import annotations

import json
import re
import sqlite3
from collections import defaultdict
from pathlib import Path


FIELD_RE = re.compile(r"\b([a-z_][a-z0-9_]*)\b")

EXCLUDE_TOKENS = {
    "true", "false", "nan", "null", "none",
}


def load_operator_names(path: Path) -> set[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {op["name"] for op in data["operators"]}


def extract_fields(expr: str, operators: set[str]) -> set[str]:
    ids = set(FIELD_RE.findall(expr))
    ids -= operators
    ids = {t for t in ids if not t.isdigit() and t not in EXCLUDE_TOKENS}
    return ids


def parse_checks(checks_json: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for c in json.loads(checks_json or "[]"):
        if isinstance(c, dict) and isinstance(c.get("name"), str):
            out[c["name"]] = c
    return out


def bucket_of(n_fields: int) -> str | None:
    if n_fields == 0:
        return None
    if n_fields == 1:
        return "1"
    if n_fields == 2:
        return "2"
    return "3+"


def main() -> int:
    ops = load_operator_names(Path("config/signatures.json"))

    conn = sqlite3.connect("checkpoints/alphas.db")
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT alpha_id, expression, sharpe, turnover, checks_json "
        "FROM alphas"
    ).fetchall()
    conn.close()

    if not rows:
        print("no alphas in registry")
        return 1

    # --- counting buckets ------------------------------------------------

    # Full bucket: everything.
    full = defaultdict(lambda: {
        "n": 0, "sub_pass": 0, "sub_fail": 0, "sub_missing": 0,
        "sharpe_sum": 0.0, "turnover_sum": 0.0, "n_with_sharpe": 0,
    })

    # Clean bucket: only alphas whose LOW_SHARPE / LOW_FITNESS /
    # LOW_TURNOVER / HIGH_TURNOVER all PASS. This isolates the
    # SUB_UNIVERSE_SHARPE question from confounds.
    clean = defaultdict(lambda: {
        "n": 0, "sub_pass": 0, "sub_fail": 0, "sub_missing": 0,
    })

    # Low-sharpe-pass bucket: only LOW_SHARPE PASS (looser than clean).
    sharpe_pass = defaultdict(lambda: {
        "n": 0, "sub_pass": 0, "sub_fail": 0, "sub_missing": 0,
    })

    for row in rows:
        expr = row["expression"] or ""
        fields = extract_fields(expr, ops)
        b = bucket_of(len(fields))
        if b is None:
            continue

        checks = parse_checks(row["checks_json"])
        sub = checks.get("LOW_SUB_UNIVERSE_SHARPE")
        sub_result = (
            str(sub.get("result", "")).upper() if sub else "MISSING"
        )

        # Full bucket.
        f = full[b]
        f["n"] += 1
        if row["sharpe"] is not None:
            f["sharpe_sum"] += float(row["sharpe"])
            f["n_with_sharpe"] += 1
        if row["turnover"] is not None:
            f["turnover_sum"] += float(row["turnover"])
        if sub_result == "PASS":
            f["sub_pass"] += 1
        elif sub_result == "FAIL":
            f["sub_fail"] += 1
        else:
            f["sub_missing"] += 1

        # Control variables.
        low_sharpe_ok = (
            checks.get("LOW_SHARPE", {}).get("result") == "PASS"
        )
        low_fitness_ok = (
            checks.get("LOW_FITNESS", {}).get("result") == "PASS"
        )
        low_turnover_ok = (
            checks.get("LOW_TURNOVER", {}).get("result") == "PASS"
        )
        high_turnover_ok = (
            checks.get("HIGH_TURNOVER", {}).get("result") == "PASS"
        )

        if low_sharpe_ok:
            s = sharpe_pass[b]
            s["n"] += 1
            if sub_result == "PASS":
                s["sub_pass"] += 1
            elif sub_result == "FAIL":
                s["sub_fail"] += 1
            else:
                s["sub_missing"] += 1

        if (
            low_sharpe_ok and low_fitness_ok
            and low_turnover_ok and high_turnover_ok
        ):
            c = clean[b]
            c["n"] += 1
            if sub_result == "PASS":
                c["sub_pass"] += 1
            elif sub_result == "FAIL":
                c["sub_fail"] += 1
            else:
                c["sub_missing"] += 1

    # --- print -----------------------------------------------------------

    def print_table(title: str, table: dict, show_averages: bool) -> None:
        print(title)
        print("-" * 78)
        header = (
            f"{'fields':>7}  {'n':>5}  {'sub_pass':>8}  {'sub_fail':>8}"
            f"  {'sub_miss':>8}  {'fail%':>6}"
        )
        if show_averages:
            header += f"  {'avg_sharpe':>10}  {'avg_turnover':>12}"
        print(header)
        for b in ("1", "2", "3+"):
            row = table.get(b)
            if not row or row["n"] == 0:
                print(f"{b:>7}  {0:>5}")
                continue
            n = row["n"]
            fail_pct = 100.0 * row["sub_fail"] / n
            line = (
                f"{b:>7}  {n:>5}  {row['sub_pass']:>8}  "
                f"{row['sub_fail']:>8}  {row['sub_missing']:>8}  "
                f"{fail_pct:>5.1f}%"
            )
            if show_averages:
                avg_sharpe = (
                    row["sharpe_sum"] / row["n_with_sharpe"]
                    if row["n_with_sharpe"] > 0 else 0.0
                )
                avg_turnover = row["turnover_sum"] / n if n > 0 else 0.0
                line += f"  {avg_sharpe:>10.4f}  {avg_turnover:>12.4f}"
            print(line)
        print()

    print_table(
        "=== ALL ALPHAS (all checks) ===",
        full,
        show_averages=True,
    )

    print_table(
        "=== LOW_SHARPE PASS only (isolates sub check from sharpe) ===",
        sharpe_pass,
        show_averages=False,
    )

    print_table(
        "=== CLEAN subset: LOW_SHARPE + LOW_FITNESS + LOW_TURNOVER + "
        "HIGH_TURNOVER all PASS ===",
        clean,
        show_averages=False,
    )

    # --- interpretation hint --------------------------------------------

    print("=" * 78)
    print("INTERPRETATION")
    print("=" * 78)
    print()
    clean_1 = clean.get("1", {"n": 0, "sub_fail": 0, "sub_pass": 0})
    clean_multi = {
        "n": sum(clean.get(b, {}).get("n", 0) for b in ("2", "3+")),
        "sub_fail": sum(
            clean.get(b, {}).get("sub_fail", 0) for b in ("2", "3+")
        ),
        "sub_pass": sum(
            clean.get(b, {}).get("sub_pass", 0) for b in ("2", "3+")
        ),
    }

    if clean_1["n"] > 0 and clean_multi["n"] > 0:
        f1 = 100.0 * clean_1["sub_fail"] / clean_1["n"]
        fm = 100.0 * clean_multi["sub_fail"] / clean_multi["n"]
        print(f"  clean single-field fail rate:  {f1:.1f}% "
              f"(n={clean_1['n']})")
        print(f"  clean multi-field fail rate:   {fm:.1f}% "
              f"(n={clean_multi['n']})")
        print()
        delta = f1 - fm
        if delta > 20:
            print(f"  -> Multi-field is {delta:.1f}pp more likely to pass.")
            print("  -> Field diversity IS a root cause. Fix diversity.")
        elif delta < 5:
            print(f"  -> Difference is {delta:.1f}pp, negligible.")
            print("  -> Field diversity is NOT the root cause. Fix fitness.")
        else:
            print(f"  -> Difference is {delta:.1f}pp, moderate.")
            print("  -> Both likely matter. Fix fitness first, then diversity.")
    else:
        print("  Not enough clean-subset data to compare. Check the tables")
        print("  above for absolute counts.")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())