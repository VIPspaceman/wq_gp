"""Tests for the CLI. Phase 4 Step 2 gate."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests._fakes import ScriptedTransport
from wqgp import cli
from wqgp.brain.transport import make_response
from wqgp.field import FieldDef
from wqgp.operator import OperatorDef
from wqgp.pool import FieldPool, OperatorPool, Pool
from wqgp.serialization import (
    SERIALIZATION_SCHEMA_VERSION,
    operator_to_dict,
)
from wqgp.types import DataType, ParamDomain, SemanticKind


# --- fixtures ---------------------------------------------------------------

def _matrix_field(fid: str) -> FieldDef:
    return FieldDef(fid, DataType.MATRIX, SemanticKind.NUMERIC, "pv1")


def _add() -> OperatorDef:
    return OperatorDef(
        "add",
        (DataType.MATRIX, DataType.MATRIX),
        DataType.MATRIX,
        commutative=True, associative=True, symmetric=True,
    )


def _sub() -> OperatorDef:
    return OperatorDef(
        "sub", (DataType.MATRIX, DataType.MATRIX), DataType.MATRIX,
    )


def _ts_mean() -> OperatorDef:
    return OperatorDef(
        "ts_mean",
        (DataType.MATRIX,),
        DataType.MATRIX,
        param_domains=(("window", ParamDomain.int_range(1, 512)),),
    )


def _test_pool() -> Pool:
    return Pool(
        fields=FieldPool(
            [
                _matrix_field("close"),
                _matrix_field("open"),
                _matrix_field("high"),
                _matrix_field("low"),
                _matrix_field("volume"),
                _matrix_field("vwap"),
            ]
        ),
        operators=OperatorPool([_add(), _sub(), _ts_mean()]),
    )


def _write_pool(path: Path) -> Path:
    _test_pool().save(path)
    return path


def _signatures_payload() -> dict:
    return {
        "schema_version": SERIALIZATION_SCHEMA_VERSION,
        "operators": [operator_to_dict(_add())],
    }


def _set_creds(monkeypatch) -> None:
    monkeypatch.setenv("WQ_BRAIN_USERNAME", "u")
    monkeypatch.setenv("WQ_BRAIN_PASSWORD", "p")


# --- parser-level behavior --------------------------------------------------

def test_no_command_prints_help(capsys) -> None:
    code = cli.main([])
    assert code == cli.EXIT_USAGE
    captured = capsys.readouterr()
    assert "usage" in captured.err.lower()


def test_version(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "wqgp" in captured.out.lower()


def test_unknown_command(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["frobnicate"])
    assert exc.value.code == 2  # argparse usage error


def test_unknown_log_level(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["list-runs", "--log-level", "BOGUS"])
    assert exc.value.code == 2


# --- build-pool -------------------------------------------------------------

def test_build_pool_writes_output(monkeypatch, tmp_path: Path) -> None:
    _set_creds(monkeypatch)

    scripted = ScriptedTransport()
    scripted.expect(
        "POST", "/authentication",
        make_response(201, json_body={"user": "u"}),
    )
    scripted.expect(
        "GET", "/data-fields",
        make_response(200, json_body=[
            {"id": "close", "type": "MATRIX", "dataset": "pv1",
             "coverage": 0.99, "delay": 1},
            {"id": "open", "type": "MATRIX", "dataset": "pv1",
             "coverage": 0.99, "delay": 1},
        ]),
    )
    scripted.expect(
        "GET", "/operators",
        make_response(200, json_body=[{"name": "add"}, {"name": "unknown"}]),
    )
    monkeypatch.setattr(
        cli, "_make_transport", lambda args: scripted
    )

    sig_path = tmp_path / "sig.json"
    sig_path.write_text(json.dumps(_signatures_payload()))
    out_path = tmp_path / "pool.json"

    code = cli.main([
        "build-pool",
        "--signatures", str(sig_path),
        "--output", str(out_path),
    ])
    assert code == cli.EXIT_OK
    assert out_path.exists()

    rebuilt = Pool.from_file(out_path)
    assert "close" in rebuilt.fields
    assert "open" in rebuilt.fields
    assert "add" in rebuilt.operators
    assert "unknown" not in rebuilt.operators


def test_build_pool_missing_signatures_file(
    monkeypatch, tmp_path: Path,
) -> None:
    _set_creds(monkeypatch)
    code = cli.main([
        "build-pool",
        "--signatures", str(tmp_path / "nope.json"),
        "--output", str(tmp_path / "pool.json"),
    ])
    assert code == cli.EXIT_ERROR


def test_build_pool_missing_credentials(
    monkeypatch, tmp_path: Path,
) -> None:
    monkeypatch.delenv("WQ_BRAIN_USERNAME", raising=False)
    monkeypatch.delenv("WQ_BRAIN_PASSWORD", raising=False)

    sig_path = tmp_path / "sig.json"
    sig_path.write_text(json.dumps(_signatures_payload()))

    code = cli.main([
        "build-pool",
        "--signatures", str(sig_path),
        "--output", str(tmp_path / "pool.json"),
    ])
    assert code == cli.EXIT_ERROR


# --- run --------------------------------------------------------------------

def test_run_missing_pool(tmp_path: Path) -> None:
    code = cli.main([
        "run",
        "--seed-count", "0",
        "--run-id", "r1",
        "--total-simulations", "10",
        "--dry-run",
        "--checkpoint-root", str(tmp_path / "ckpt"),
    ])
    assert code == cli.EXIT_ERROR


def test_run_missing_run_id(tmp_path: Path) -> None:
    pool_path = _write_pool(tmp_path / "pool.json")
    code = cli.main([
        "run",
        "--seed-count", "0",
        "--pool", str(pool_path),
        "--total-simulations", "10",
        "--dry-run",
        "--checkpoint-root", str(tmp_path / "ckpt"),
    ])
    assert code == cli.EXIT_ERROR


def test_run_missing_budget(tmp_path: Path) -> None:
    pool_path = _write_pool(tmp_path / "pool.json")
    code = cli.main([
        "run",
        "--seed-count", "0",
        "--pool", str(pool_path),
        "--run-id", "r1",
        "--dry-run",
        "--checkpoint-root", str(tmp_path / "ckpt"),
    ])
    assert code == cli.EXIT_ERROR


def test_run_dry_run_end_to_end(tmp_path: Path) -> None:
    pool_path = _write_pool(tmp_path / "pool.json")
    ckpt_root = tmp_path / "ckpt"

    code = cli.main([
        "run",
        "--seed-count", "0",
        "--pool", str(pool_path),
        "--run-id", "smoke",
        "--total-simulations", "20",
        "--population-size", "4",
        "--elite-count", "1",
        "--max-generations", "1",
        "--seed", "42",
        "--n-workers", "2",
        "--checkpoint-root", str(ckpt_root),
        "--dry-run",
    ])
    assert code == cli.EXIT_OK

    run_dir = ckpt_root / "runs" / "smoke"
    assert run_dir.exists()
    assert list(run_dir.glob("step_*.json"))


def test_run_config_file_provides_defaults(tmp_path: Path) -> None:
    pool_path = _write_pool(tmp_path / "pool.json")
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps({
        "run_id": "from_cfg",
        "pool_path": str(pool_path),
        "checkpoint_root": str(tmp_path / "ckpt"),
        "seed": 1,
        "max_generations": 1,
        "n_workers": 2,
        "budget": {"total_simulations": 20},
        "selection": {"population_size": 4, "elite_count": 1},
    }))

    code = cli.main([
        "run",
        "--seed-count", "0",
        "--config", str(cfg_path),
        "--dry-run",
    ])
    assert code == cli.EXIT_OK
    assert (tmp_path / "ckpt" / "runs" / "from_cfg").exists()


def test_run_flag_overrides_config(tmp_path: Path) -> None:
    pool_path = _write_pool(tmp_path / "pool.json")
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps({
        "run_id": "from_cfg",
        "pool_path": str(pool_path),
        "checkpoint_root": str(tmp_path / "ckpt"),
        "seed": 1,
        "max_generations": 5,      # overridden to 1 below
        "budget": {"total_simulations": 20},
        "selection": {"population_size": 4, "elite_count": 1},
    }))

    code = cli.main([
        "run",
        "--seed-count", "0",
        "--config", str(cfg_path),
        "--run-id", "overridden",
        "--max-generations", "1",
        "--dry-run",
    ])
    assert code == cli.EXIT_OK
    # Run id came from the flag, not the config.
    assert (tmp_path / "ckpt" / "runs" / "overridden").exists()
    assert not (tmp_path / "ckpt" / "runs" / "from_cfg").exists()


def test_run_invalid_config_json(tmp_path: Path) -> None:
    cfg_path = tmp_path / "broken.json"
    cfg_path.write_text("{ not json")
    code = cli.main([
        "run",
        "--seed-count", "0",
        "--config", str(cfg_path),
        "--dry-run",
        "--pool", str(tmp_path / "pool.json"),
        "--run-id", "r",
        "--total-simulations", "1",
        "--checkpoint-root", str(tmp_path / "ckpt"),
    ])
    assert code == cli.EXIT_ERROR


def test_run_config_root_not_object(tmp_path: Path) -> None:
    cfg_path = tmp_path / "list.json"
    cfg_path.write_text(json.dumps([1, 2, 3]))
    code = cli.main([
        "run",
        "--seed-count", "0",
        "--config", str(cfg_path),
        "--dry-run",
        "--pool", str(tmp_path / "pool.json"),
        "--run-id", "r",
        "--total-simulations", "1",
        "--checkpoint-root", str(tmp_path / "ckpt"),
    ])
    assert code == cli.EXIT_ERROR


# --- list-runs --------------------------------------------------------------

def test_list_runs_empty_root(capsys, tmp_path: Path) -> None:
    code = cli.main([
        "list-runs",
        "--checkpoint-root", str(tmp_path / "nothing"),
    ])
    assert code == cli.EXIT_OK
    captured = capsys.readouterr()
    assert "no checkpoint root" in captured.err


def test_list_runs_empty_store(capsys, tmp_path: Path) -> None:
    root = tmp_path / "ckpt"
    root.mkdir()
    code = cli.main([
        "list-runs", "--checkpoint-root", str(root),
    ])
    assert code == cli.EXIT_OK
    captured = capsys.readouterr()
    assert "no runs" in captured.err


def test_list_runs_shows_runs(capsys, tmp_path: Path) -> None:
    pool_path = _write_pool(tmp_path / "pool.json")
    ckpt_root = tmp_path / "ckpt"
    for rid in ("alpha", "beta"):
        code = cli.main([
            "run",
            "--seed-count", "0",
            "--pool", str(pool_path),
            "--run-id", rid,
            "--total-simulations", "20",
            "--population-size", "4",
            "--elite-count", "1",
            "--max-generations", "1",
            "--seed", "42",
            "--n-workers", "2",
            "--checkpoint-root", str(ckpt_root),
            "--dry-run",
        ])
        assert code == cli.EXIT_OK

    code = cli.main([
        "list-runs", "--checkpoint-root", str(ckpt_root),
    ])
    assert code == cli.EXIT_OK
    captured = capsys.readouterr()
    assert "alpha" in captured.out
    assert "beta" in captured.out

def test_list_runs_works_without_config_flag(
    capsys, tmp_path: Path,
) -> None:
    """Subcommands that do not accept --config must still work.

    Regression test for D-120: when _load_settings started reading
    args.config directly, every subcommand without the flag failed
    with an AttributeError converted to EXIT_ERROR.
    """
    root = tmp_path / "ckpt"
    root.mkdir()
    code = cli.main([
        "list-runs", "--checkpoint-root", str(root),
    ])
    assert code == cli.EXIT_OK


def test_build_pool_works_without_config_flag(
    monkeypatch, tmp_path: Path,
) -> None:
    """Same as above, for the build-pool subcommand."""
    _set_creds(monkeypatch)

    scripted = ScriptedTransport()
    scripted.expect(
        "POST", "/authentication",
        make_response(201, json_body={"user": "u"}),
    )
    scripted.expect(
        "GET", "/data-fields",
        make_response(200, json_body=[
            {"id": "close", "type": "MATRIX", "dataset": "pv1",
             "coverage": 0.99, "delay": 1},
        ]),
    )
    scripted.expect(
        "GET", "/operators",
        make_response(200, json_body=[{"name": "add"}]),
    )
    monkeypatch.setattr(
        cli, "_make_transport", lambda args: scripted
    )

    sig_path = tmp_path / "sig.json"
    sig_path.write_text(json.dumps(_signatures_payload()))
    out_path = tmp_path / "pool.json"

    code = cli.main([
        "build-pool",
        "--signatures", str(sig_path),
        "--output", str(out_path),
    ])
    assert code == cli.EXIT_OK