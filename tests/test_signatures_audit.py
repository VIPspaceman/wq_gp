"""Tests for the signatures auditor. Phase 4 Step 4 gate."""
from __future__ import annotations

import pytest

from tests._fakes import ScriptedTransport
from wqgp.brain.client import BrainCredentials
from wqgp.brain.pool_builder import (
    OperatorSignatureTable,
    PoolBuilderError,
    PoolBuilderProtocolError,
)
from wqgp.brain.transport import TransportError, make_response
from wqgp.operator import OperatorDef
from wqgp.signatures_audit import (
    SignatureAudit,
    audit_signatures,
    format_skeletons,
)
from wqgp.types import DataType, ParamDomain


# --- fixtures ---------------------------------------------------------------

def _creds() -> BrainCredentials:
    return BrainCredentials(username="u", password="p")


def _login_ok():
    return make_response(201, json_body={"user": "u"})


def _ops_page(*names: str):
    return make_response(200, json_body=[{"name": n} for n in names])


def _add() -> OperatorDef:
    return OperatorDef(
        "add",
        (DataType.MATRIX, DataType.MATRIX),
        DataType.MATRIX,
        commutative=True, associative=True, symmetric=True,
    )


def _ts_mean() -> OperatorDef:
    return OperatorDef(
        "ts_mean",
        (DataType.MATRIX,),
        DataType.MATRIX,
        param_domains=(("window", ParamDomain.int_range(1, 512)),),
    )


def _table(*ops: OperatorDef) -> OperatorSignatureTable:
    return OperatorSignatureTable({op.name: op for op in ops})


def _transport_with_operators(t: ScriptedTransport, *names: str) -> None:
    t.expect("POST", "/authentication", _login_ok())
    t.expect("GET", "/operators", _ops_page(*names))


class CloseCountingTransport(ScriptedTransport):
    """ScriptedTransport that counts close() calls."""

    def __init__(self) -> None:
        super().__init__()
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


# --- SignatureAudit invariants ---------------------------------------------

def test_audit_invariants_hold_for_valid_input() -> None:
    # brain = missing(2) + common(1) = 3
    # table = extra(1) + common(1) = 2
    SignatureAudit(
        brain_operator_count=3,
        table_operator_count=2,
        missing_from_table=("a", "b"),
        extra_in_table=("c",),
        common=("d",),
    )


def test_audit_rejects_mismatched_brain_count() -> None:
    with pytest.raises(ValueError):
        SignatureAudit(
            brain_operator_count=99,
            table_operator_count=1,
            missing_from_table=("a",),
            extra_in_table=(),
            common=(),
        )


def test_audit_rejects_mismatched_table_count() -> None:
    with pytest.raises(ValueError):
        SignatureAudit(
            brain_operator_count=1,
            table_operator_count=99,
            missing_from_table=("a",),
            extra_in_table=(),
            common=(),
        )


def test_audit_rejects_missing_extra_overlap() -> None:
    with pytest.raises(ValueError):
        SignatureAudit(
            brain_operator_count=1,
            table_operator_count=1,
            missing_from_table=("a",),
            extra_in_table=("a",),
            common=(),
        )


def test_audit_rejects_missing_common_overlap() -> None:
    with pytest.raises(ValueError):
        SignatureAudit(
            brain_operator_count=2,
            table_operator_count=1,
            missing_from_table=("a",),
            extra_in_table=(),
            common=("a",),
        )


def test_audit_is_complete_when_no_missing() -> None:
    a = SignatureAudit(
        brain_operator_count=1,
        table_operator_count=1,
        missing_from_table=(),
        extra_in_table=(),
        common=("a",),
    )
    assert a.is_complete()


def test_audit_is_incomplete_when_missing() -> None:
    a = SignatureAudit(
        brain_operator_count=2,
        table_operator_count=1,
        missing_from_table=("a",),
        extra_in_table=(),
        common=("b",),
    )
    assert not a.is_complete()


# --- audit_signatures -------------------------------------------------------

def test_audit_finds_missing_operators() -> None:
    t = ScriptedTransport()
    _transport_with_operators(t, "add", "sub", "ts_mean")
    audit = audit_signatures(_creds(), t, _table(_add()))
    assert audit.brain_operator_count == 3
    assert audit.table_operator_count == 1
    assert audit.common == ("add",)
    assert audit.missing_from_table == ("sub", "ts_mean")
    assert audit.extra_in_table == ()


def test_audit_finds_extra_operators() -> None:
    t = ScriptedTransport()
    _transport_with_operators(t, "add")
    audit = audit_signatures(_creds(), t, _table(_add(), _ts_mean()))
    assert audit.common == ("add",)
    assert audit.extra_in_table == ("ts_mean",)
    assert audit.missing_from_table == ()


def test_audit_complete_when_everything_matches() -> None:
    t = ScriptedTransport()
    _transport_with_operators(t, "add", "ts_mean")
    audit = audit_signatures(_creds(), t, _table(_add(), _ts_mean()))
    assert audit.is_complete()
    assert audit.missing_from_table == ()
    assert audit.extra_in_table == ()


def test_audit_nothing_matches() -> None:
    t = ScriptedTransport()
    _transport_with_operators(t, "a", "b")
    audit = audit_signatures(_creds(), t, _table(_add()))
    assert audit.missing_from_table == ("a", "b")
    assert audit.extra_in_table == ("add",)
    assert audit.common == ()


def test_audit_empty_table() -> None:
    t = ScriptedTransport()
    _transport_with_operators(t, "add", "sub")
    audit = audit_signatures(
        _creds(), t, OperatorSignatureTable.empty()
    )
    assert audit.missing_from_table == ("add", "sub")
    assert audit.table_operator_count == 0


def test_audit_empty_brain() -> None:
    t = ScriptedTransport()
    _transport_with_operators(t)
    audit = audit_signatures(_creds(), t, _table(_add()))
    assert audit.missing_from_table == ()
    assert audit.extra_in_table == ("add",)
    # Complete in the "no missing" sense: nothing on BRAIN to cover.
    assert audit.is_complete()


def test_audit_does_not_close_transport() -> None:
    """audit_signatures must not close the caller's transport (D-104)."""
    t = CloseCountingTransport()
    t.expect("POST", "/authentication", _login_ok())
    t.expect("GET", "/operators", _ops_page("add"))
    audit_signatures(_creds(), t, _table(_add()))
    assert t.close_calls == 0


# --- error handling ---------------------------------------------------------

def test_audit_login_failure_raises() -> None:
    t = ScriptedTransport()
    t.expect("POST", "/authentication", make_response(401))
    with pytest.raises(PoolBuilderError):
        audit_signatures(_creds(), t, _table(_add()))


def test_audit_transport_error_raises() -> None:
    t = ScriptedTransport()
    t.expect("POST", "/authentication", TransportError("dns"))
    with pytest.raises(PoolBuilderError):
        audit_signatures(_creds(), t, _table(_add()))


def test_audit_bad_response_shape_raises() -> None:
    t = ScriptedTransport()
    t.expect("POST", "/authentication", _login_ok())
    t.expect(
        "GET", "/operators",
        make_response(200, json_body={"unexpected": True}),
    )
    with pytest.raises(PoolBuilderProtocolError):
        audit_signatures(_creds(), t, _table(_add()))


# --- format_skeletons -------------------------------------------------------

def test_format_skeletons_basic() -> None:
    out = format_skeletons(("a", "b", "c"), limit=10)
    assert '"name": "a"' in out
    assert '"name": "b"' in out
    assert '"name": "c"' in out
    assert "more" not in out


def test_format_skeletons_respects_limit() -> None:
    out = format_skeletons(("a", "b", "c"), limit=2)
    assert '"name": "a"' in out
    assert '"name": "b"' in out
    assert '"name": "c"' not in out
    assert "and 1 more" in out


def test_format_skeletons_zero_limit() -> None:
    out = format_skeletons(("a", "b"), limit=0)
    assert '"name": "a"' not in out
    assert "and 2 more" in out


def test_format_skeletons_negative_limit_rejected() -> None:
    with pytest.raises(ValueError):
        format_skeletons(("a",), limit=-1)


def test_format_skeletons_empty_input_returns_empty() -> None:
    assert format_skeletons((), limit=10) == ""


def test_format_skeletons_includes_matrix_placeholder() -> None:
    out = format_skeletons(("x",), limit=1)
    assert '"MATRIX"' in out