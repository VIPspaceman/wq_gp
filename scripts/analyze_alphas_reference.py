#!/usr/bin/env python3
"""Analyze the historical alphas reference set.

Usage:
    python scripts/analyze_alphas_reference.py \\
        [--reference data/alphas_reference.json] \\
        [--signatures signatures.json] \\
        [--checkpoint-root ./checkpoints] \\
        [--report data/alphas_reference_analysis.json]

Read-only. Does not touch the cache or any wqgp runtime state.

Normalization policy: the ONLY transformation applied before
comparing an API expression to a run-DB expression is
whitespace-stripping. The alternative -- rewriting `+(-1*x)` into
`subtract(a, b)` -- requires real parsing and is deliberately out of
scope (see D-145/D-146 decisions in the project log). If the
whitespace-normalized overlap is high, that is a strong signal the
bridge layer is worth building; if it is low, no amount of extra
normalization would change the cost/benefit.
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path


# --- regexes ----------------------------------------------------------------

_OP_PATTERN = re.compile(r"([a-z_][a-z0-9_]*)\s*\(")
_INT_PARAM = re.compile(r",\s*(\d+)\s*[,)]")


def extract_operators(expr: str) -> set[str]:
    return set(_OP_PATTERN.findall(expr))


def extract_int_params(expr: str) -> list[int]:
    return [int(m) for m in _INT_PARAM.findall(expr)]


def extract_identifiers(expr: str) -> set[str]:
    """All bare identifiers. Operators are subtracted by the caller."""
    return set(re.findall(r"\b([a-z_][a-z0-9_]*)\b", expr))


def normalize_whitespace(expr: str) -> str:
    """Strip all whitespace. The only normalization this script does."""
    return re.sub(r"\s+", "", expr)


# --- loading ----------------------------------------------------------------

def load_signatures(path: Path) -> set[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {op["name"] for op in data["operators"]}


def load_run_expressions(checkpoint_root: Path) -> set[str]:
    """Collect every expression string from every run DB."""
    exprs: set[str] = set()
    if not checkpoint_root.is_dir():
        return exprs
    for db in checkpoint_root.glob("*.db"):
        if db.name == "cache.db":
            continue
        try:
            conn = sqlite3.connect(str(db))
            cur = conn.execute(
                "SELECT DISTINCT expression FROM evaluations"
            )
            for (e,) in cur:
                if e:
                    exprs.add(e)
            conn.close()
        except sqlite3.Error:
            continue
    return exprs


# --- analysis ---------------------------------------------------------------

def classify(ops: set[str], supported: set[str]) -> str:
    """Return 'fully' / 'partial' / 'unsupported' / 'no_ops'."""
    if not ops:
        return "no_ops"
    unknown = ops - supported
    if not unknown:
        return "fully"
    if len(unknown) == len(ops):
        return "unsupported"
    return "partial"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference", default="data/alphas_reference.json",
    )
    parser.add_argument("--signatures", default="signatures.json")
    parser.add_argument("--checkpoint-root", default="./checkpoints")
    parser.add_argument(
        "--report", default="data/alphas_reference_analysis.json",
    )
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args()

    ref_path = Path(args.reference)
    if not ref_path.exists():
        print(
            f"reference file not found: {ref_path}",
            file=sys.stderr,
        )
        print(
            "run scripts/fetch_alphas_reference.py first",
            file=sys.stderr,
        )
        return 1

    payload = json.loads(ref_path.read_text(encoding="utf-8"))
    alphas = payload["alphas"]

    supported = load_signatures(Path(args.signatures))
    print(
        f"loaded {len(supported)} supported operators from "
        f"signatures.json"
    )
    print(f"loaded {len(alphas)} alphas from {ref_path}")
    print()

    run_exprs_raw = load_run_expressions(Path(args.checkpoint_root))
    run_exprs_norm = {normalize_whitespace(e) for e in run_exprs_raw}
    print(
        f"loaded {len(run_exprs_raw)} distinct run-DB expressions"
    )
    print()

    # --- per-alpha classification ---
    classes = Counter()
    op_counter: Counter = Counter()
    field_counter: Counter = Counter()
    window_counter: Counter = Counter()
    fully_supported_alphas = []

    for a in alphas:
        expr = a.get("expression") or ""
        ops = extract_operators(expr)
        cls = classify(ops, supported)
        classes[cls] += 1
        for o in ops:
            op_counter[o] += 1
        fields = extract_identifiers(expr) - ops
        fields = {f for f in fields if f not in {"true", "false", "nan"}}
        for f in fields:
            field_counter[f] += 1
        for w in extract_int_params(expr):
            window_counter[w] += 1
        if cls == "fully":
            fully_supported_alphas.append(a)

    # --- overlap with run DBs ---
    raw_hits = 0
    norm_hits = 0
    raw_examples = []
    norm_examples = []
    for a in alphas:
        expr = a.get("expression") or ""
        if expr in run_exprs_raw:
            raw_hits += 1
            if len(raw_examples) < 5:
                raw_examples.append(expr[:100])
        elif normalize_whitespace(expr) in run_exprs_norm:
            norm_hits += 1
            if len(norm_examples) < 5:
                norm_examples.append(expr[:100])

    # --- print report ---
    print("=== classification ===")
    for cls in ("fully", "partial", "unsupported", "no_ops"):
        print(f"  {cls:12} {classes.get(cls, 0)}")
    print()

    print(f"=== top {args.top_n} operators in reference set ===")
    for op, n in op_counter.most_common(args.top_n):
        mark = "" if op in supported else "  <- UNSUPPORTED"
        print(f"  {n:4}  {op}{mark}")
    print()

    unknown_ops = {
        op: n for op, n in op_counter.items() if op not in supported
    }
    print(f"=== unsupported operators ({len(unknown_ops)}) ===")
    for op, n in sorted(unknown_ops.items(), key=lambda kv: -kv[1]):
        print(f"  {n:4}  {op}")
    print()

    print(f"=== top {args.top_n} fields ===")
    for f, n in field_counter.most_common(args.top_n):
        print(f"  {n:4}  {f}")
    print()

    print(f"=== top {args.top_n} integer windows ===")
    for w, n in window_counter.most_common(args.top_n):
        print(f"  {n:4}  {w}")
    print()

    print("=== overlap with existing run DBs ===")
    print(f"  raw exact match:        {raw_hits} / {len(alphas)}")
    print(
        f"  whitespace-normalized:  {norm_hits} / {len(alphas)}"
    )
    for e in raw_examples[:3]:
        print(f"    raw:  {e}")
    for e in norm_examples[:3]:
        print(f"    norm: {e}")
    print()

    # --- decision hints ---
    print("=== decision hints ===")
    total = len(alphas)
    fully_pct = (
        100.0 * classes.get("fully", 0) / total if total else 0.0
    )
    norm_overlap_pct = 100.0 * norm_hits / total if total else 0.0

    if norm_hits > 200:
        print(
            f"  -> {norm_hits} expressions match run DBs after "
            f"normalization."
        )
        print("  -> Worth building the lightweight bridge (method D).")
    elif norm_hits > 50:
        print(
            f"  -> {norm_hits} expressions match after normalization."
        )
        print(
            "  -> Marginal; consider a text-only normalizer, no parser."
        )
    else:
        print(
            f"  -> Only {norm_hits} expressions match run DBs even "
            f"after normalization."
        )
        print(
            "  -> Bridge layer not worth building. Stay on method C."
        )

    if fully_pct < 20.0:
        print(
            f"  -> Only {fully_pct:.1f}% of reference alphas use only "
            f"supported operators."
        )
        print(
            "  -> Seed injection would require expanding the signature "
            "table first."
        )

    # --- write machine-readable report ---
    report = {
        "reference_count": total,
        "supported_operators": sorted(supported),
        "classification": dict(classes),
        "unsupported_operators": dict(unknown_ops),
        "top_operators": dict(op_counter.most_common(args.top_n)),
        "top_fields": dict(field_counter.most_common(args.top_n)),
        "top_windows": {
            str(k): v for k, v in window_counter.most_common(args.top_n)
        },
        "overlap_raw": raw_hits,
        "overlap_normalized": norm_hits,
        "overlap_normalized_pct": norm_overlap_pct,
        "fully_supported_alphas": [
            {
                "alpha_id": a["alpha_id"],
                "expression": a["expression"],
                "sharpe": a["metrics"]["sharpe"],
            }
            for a in fully_supported_alphas
        ],
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2), encoding="utf-8",
    )
    print()
    print(f"report written to {report_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())