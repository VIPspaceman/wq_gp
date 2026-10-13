"""Tests for the BRAIN pool builder. Phase 4 Step 1 gate."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tests._fakes import ScriptedTransport
from wqgp.brain.client import BrainCredentials, BrainSettings
from wqgp.brain.pool_builder import (
    OperatorSignatureTable,
    PoolBuilder,
    PoolBuilderError,
    PoolBuilderProtocolError,
    PoolBuildResult,
)
from wqgp.brain.transport import TransportError, make_response
from wqgp.operator import OperatorDef
from wqgp.serialization import (
    SERIALIZATION_SCHEMA_VERSION,
    SerializationError,
    operator_to_dict,
)
from wqgp.types import DataType, ParamDomain


# --- fixtures ---------------------------------------------------------------

def _creds() -> BrainCredentials:
    return BrainCredentials(username="u", password="p")


def _login_ok() -> Any:
    return make_response(201, json_body={"user": "u"})


def _page(payload: Any) -> Any:
    return make_response(200, json_body=payload)


def _field(
    fid: str,
    *,
    ftype: str = "MATRIX",
    dataset: Any = "pv1",
    coverage: float = 0.99,
    delay: int = 1,
    unit: Any = None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "id": fid,
        "type": ftype,
        "dataset": dataset,
        "coverage": coverage,
        "delay": delay,
    }
    if unit is not None:
        entry["unit"] = unit
    return entry


def _op_dict(name: str) -> dict[str, Any]:
    return {"name": name, "category": "Time Series", "description": "..."}


def _sig_table(*ops: OperatorDef) -> OperatorSignatureTable:
    return OperatorSignatureTable({op.name: op for op in ops})


def _ts_mean() -> OperatorDef:
    return OperatorDef(
        "ts_mean",
        (DataType.MATRIX,),
        DataType.MATRIX,
        param_domains=(("window", ParamDomain.int_range(1, 512)),),
    )


def _add() -> OperatorDef:
    return OperatorDef(
        "add",
        (DataType.MATRIX, DataType.MATRIX),
        DataType.MATRIX,
        commutative=True, associative=True, symmetric=True,
    )


def _builder(
    t: ScriptedTransport,
    *,
    page_limit: int = 50,
    max_pages: int = 10,
) -> PoolBuilder:
    return PoolBuilder(
        credentials=_creds(),
        transport=t,
        page_limit=page_limit,
        max_pages=max_pages,
        http_timeout=10.0,
    )


# --- OperatorSignatureTable -------------------------------------------------

def test_signature_table_empty() -> None:
    t = OperatorSignatureTable.empty()
    assert len(t) == 0
    assert t.get("anything") is None
    assert "anything" not in t


def test_signature_table_get_and_names() -> None:
    table = _sig_table(_add(), _ts_mean())
    assert table.get("add") is _add() or table.get("add").name == "add"
    assert table.names() == ("add", "ts_mean")


def test_signature_table_from_dict_ok() -> None:
    payload = {
        "schema_version": SERIALIZATION_SCHEMA_VERSION,
        "operators": [operator_to_dict(_add())],
    }
    table = OperatorSignatureTable.from_dict(payload)
    assert "add" in table


def test_signature_table_from_dict_wrong_schema() -> None:
    with pytest.raises(SerializationError):
        OperatorSignatureTable.from_dict(
            {"schema_version": 999, "operators": []}
        )


def test_signature_table_from_dict_not_a_list() -> None:
    with pytest.raises(SerializationError):
        OperatorSignatureTable.from_dict(
            {"schema_version": SERIALIZATION_SCHEMA_VERSION,
             "operators": {}}
        )


def test_signature_table_from_dict_duplicate_rejected() -> None:
    payload = {
        "schema_version": SERIALIZATION_SCHEMA_VERSION,
        "operators": [
            operator_to_dict(_add()),
            operator_to_dict(_add()),
        ],
    }
    with pytest.raises(SerializationError):
        OperatorSignatureTable.from_dict(payload)


def test_signature_table_from_file_ok(tmp_path: Path) -> None:
    p = tmp_path / "sig.json"
    p.write_text(json.dumps({
        "schema_version": SERIALIZATION_SCHEMA_VERSION,
        "operators": [operator_to_dict(_add())],
    }))
    table = OperatorSignatureTable.from_file(p)
    assert "add" in table


def test_signature_table_from_file_missing(tmp_path: Path) -> None:
    with pytest.raises(SerializationError):
        OperatorSignatureTable.from_file(tmp_path / "nope.json")


def test_signature_table_from_file_corrupt(tmp_path: Path) -> None:
    p = tmp_path / "sig.json"
    p.write_text("{ not json")
    with pytest.raises(SerializationError):
        OperatorSignatureTable.from_file(p)


# --- constructor validation -------------------------------------------------

def test_builder_rejects_bad_config() -> None:
    t = ScriptedTransport()
    with pytest.raises(ValueError):
        PoolBuilder(_creds(), t, page_limit=0)
    with pytest.raises(ValueError):
        PoolBuilder(_creds(), t, max_pages=0)
    with pytest.raises(ValueError):
        PoolBuilder(_creds(), t, http_timeout=0)


# --- fetch_fields -----------------------------------------------------------

def test_fetch_fields_single_page() -> None:
    t = ScriptedTransport()
    t.expect("POST", "/authentication", _login_ok())
    t.expect(
        "GET", "/data-fields",
        _page([
            _field("close"),
            _field("open"),
            _field("sector", ftype="GROUP"),
        ]),
    )
    fields, dropped, pages = _builder(t).fetch_fields()
    assert pages == 1
    assert {f.field_id for f in fields} == {"close", "open", "sector"}
    assert dropped == ()
    sector = next(f for f in fields if f.field_id == "sector")
    assert sector.data_type is DataType.GROUP


def test_fetch_fields_paginates() -> None:
    t = ScriptedTransport()
    t.expect("POST", "/authentication", _login_ok())
    # First page returns exactly page_limit items, forcing a second call.
    t.expect(
        "GET", "/data-fields",
        _page([_field(f"f{i}") for i in range(3)]),
        _page([_field("last")]),
    )
    b = _builder(t, page_limit=3)
    fields, dropped, pages = b.fetch_fields()
    assert pages == 2
    assert {f.field_id for f in fields} == {"f0", "f1", "f2", "last"}


def test_fetch_fields_unknown_type_dropped() -> None:
    t = ScriptedTransport()
    t.expect("POST", "/authentication", _login_ok())
    t.expect(
        "GET", "/data-fields",
        _page([
            _field("good"),
            _field("weird", ftype="TIME_SERIES"),  # not in our map
        ]),
    )
    fields, dropped, _ = _builder(t).fetch_fields()
    assert [f.field_id for f in fields] == ["good"]
    assert dropped == ("weird",)


def test_fetch_fields_missing_dataset_dropped() -> None:
    t = ScriptedTransport()
    t.expect("POST", "/authentication", _login_ok())
    t.expect(
        "GET", "/data-fields",
        _page([
            {"id": "naked", "type": "MATRIX"},  # no dataset key
        ]),
    )
    fields, dropped, _ = _builder(t).fetch_fields()
    assert fields == ()
    assert dropped == ("naked",)


def test_fetch_fields_dataset_as_object() -> None:
    t = ScriptedTransport()
    t.expect("POST", "/authentication", _login_ok())
    t.expect(
        "GET", "/data-fields",
        _page([_field("close", dataset={"id": "pv1", "name": "Price"})]),
    )
    fields, _, _ = _builder(t).fetch_fields()
    assert fields[0].dataset == "pv1"


def test_fetch_fields_coverage_and_delay_coerced() -> None:
    t = ScriptedTransport()
    t.expect("POST", "/authentication", _login_ok())
    t.expect(
        "GET", "/data-fields",
        _page([
            _field("c1", coverage=1.5, delay=-3),    # both out of range
            _field("c2", coverage=True, delay=True), # both bool
        ]),
    )
    fields, _, _ = _builder(t).fetch_fields()
    by_id = {f.field_id: f for f in fields}
    assert by_id["c1"].coverage == 1.0
    assert by_id["c1"].delay == 1
    assert by_id["c2"].coverage == 1.0
    assert by_id["c2"].delay == 1


def test_fetch_fields_semantic_kind_inference() -> None:
    t = ScriptedTransport()
    t.expect("POST", "/authentication", _login_ok())
    t.expect(
        "GET", "/data-fields",
        _page([
            _field("close", ftype="MATRIX"),
            _field("sector", ftype="GROUP"),
        ]),
    )
    fields, _, _ = _builder(t).fetch_fields()
    by_id = {f.field_id: f for f in fields}
    assert by_id["sector"].is_categorical
    assert not by_id["close"].is_categorical


# --- fetch_operator_names ---------------------------------------------------

def test_fetch_operator_names() -> None:
    t = ScriptedTransport()
    t.expect("POST", "/authentication", _login_ok())
    t.expect(
        "GET", "/operators",
        _page([_op_dict("add"), _op_dict("ts_mean"), _op_dict("sub")]),
    )
    names, pages = _builder(t).fetch_operator_names()
    assert names == ("add", "sub", "ts_mean")
    assert pages == 1


def test_fetch_operator_names_deduplicates() -> None:
    t = ScriptedTransport()
    t.expect("POST", "/authentication", _login_ok())
    t.expect(
        "GET", "/operators",
        _page([_op_dict("add"), _op_dict("add")]),
    )
    names, _ = _builder(t).fetch_operator_names()
    assert names == ("add",)


def test_fetch_operator_names_skips_malformed() -> None:
    t = ScriptedTransport()
    t.expect("POST", "/authentication", _login_ok())
    t.expect(
        "GET", "/operators",
        _page([
            _op_dict("add"),
            {"category": "x"},          # no name/id
            "not a dict",
            {"id": "sub"},              # 'id' is accepted as fallback
        ]),
    )
    names, _ = _builder(t).fetch_operator_names()
    assert names == ("add", "sub")


# --- build ------------------------------------------------------------------

def test_build_keeps_known_operators_drops_unknown() -> None:
    t = ScriptedTransport()
    t.expect("POST", "/authentication", _login_ok())
    t.expect(
        "GET", "/data-fields",
        _page([_field("close"), _field("open")]),
    )
    t.expect(
        "GET", "/operators",
        _page([_op_dict("add"), _op_dict("ts_mean"), _op_dict("mystery")]),
    )

    result = _builder(t).build(_sig_table(_add(), _ts_mean()))

    assert isinstance(result, PoolBuildResult)
    assert {op.name for op in result.pool.operators} == {"add", "ts_mean"}
    assert result.dropped_operator_names == ("mystery",)
    assert result.dropped_field_ids == ()
    assert result.fields_fetched == 2
    assert result.operators_fetched == 3


def test_build_summary() -> None:
    t = ScriptedTransport()
    t.expect("POST", "/authentication", _login_ok())
    t.expect(
        "GET", "/data-fields",
        _page([_field("close"), _field("bad", ftype="BOGUS")]),
    )
    t.expect(
        "GET", "/operators",
        _page([_op_dict("add"), _op_dict("unknown")]),
    )

    s = _builder(t).build(_sig_table(_add())).summary()

    assert s["fields_kept"] == 1
    assert s["fields_dropped"] == 1
    assert s["operators_kept"] == 1
    assert s["operators_dropped"] == 1
    assert s["fields_fetched"] == 2
    assert s["operators_fetched"] == 2


def test_build_empty_operators_ok() -> None:
    """A run with only fields and no operators is legal at the pool
    level (D-023)."""
    t = ScriptedTransport()
    t.expect("POST", "/authentication", _login_ok())
    t.expect("GET", "/data-fields", _page([_field("close")]))
    t.expect("GET", "/operators", _page([]))

    result = _builder(t).build(OperatorSignatureTable.empty())
    assert len(result.pool.fields) == 1
    assert len(result.pool.operators) == 0


def test_build_empty_fields_and_operators() -> None:
    t = ScriptedTransport()
    t.expect("POST", "/authentication", _login_ok())
    t.expect("GET", "/data-fields", _page([]))
    t.expect("GET", "/operators", _page([]))
    result = _builder(t).build(OperatorSignatureTable.empty())
    assert len(result.pool.fields) == 0
    assert len(result.pool.operators) == 0


# --- errors -----------------------------------------------------------------

def test_login_failure_raises() -> None:
    t = ScriptedTransport()
    t.expect("POST", "/authentication", make_response(401))
    with pytest.raises(PoolBuilderError):
        _builder(t).fetch_fields()


def test_login_transport_error_raises() -> None:
    t = ScriptedTransport()
    t.expect("POST", "/authentication", TransportError("dns"))
    with pytest.raises(PoolBuilderError):
        _builder(t).fetch_fields()


def test_fetch_401_after_login_raises() -> None:
    t = ScriptedTransport()
    t.expect("POST", "/authentication", _login_ok())
    t.expect("GET", "/data-fields", make_response(401))
    with pytest.raises(PoolBuilderError):
        _builder(t).fetch_fields()


def test_fetch_non_200_raises() -> None:
    t = ScriptedTransport()
    t.expect("POST", "/authentication", _login_ok())
    t.expect("GET", "/data-fields", make_response(500))
    with pytest.raises(PoolBuilderError):
        _builder(t).fetch_fields()


def test_fetch_non_json_raises() -> None:
    t = ScriptedTransport()
    t.expect("POST", "/authentication", _login_ok())
    t.expect("GET", "/data-fields", make_response(200, body=b"not json"))
    with pytest.raises(PoolBuilderProtocolError):
        _builder(t).fetch_fields()


def test_fetch_unexpected_shape_raises() -> None:
    t = ScriptedTransport()
    t.expect("POST", "/authentication", _login_ok())
    t.expect(
        "GET", "/data-fields",
        make_response(200, json_body={"unexpected": True}),
    )
    with pytest.raises(PoolBuilderProtocolError):
        _builder(t).fetch_fields()


def test_fetch_exceeds_max_pages() -> None:
    """Every page is DISTINCT and full; max_pages caps the loop."""
    t = ScriptedTransport()
    t.expect("POST", "/authentication", _login_ok())
    t.expect(
        "GET", "/data-fields",
        _page([_field("a0"), _field("a1")]),
        _page([_field("b0"), _field("b1")]),
        _page([_field("c0"), _field("c1")]),
    )
    with pytest.raises(PoolBuilderError):
        _builder(t, page_limit=2, max_pages=3).fetch_fields()


def test_fetch_stops_when_server_ignores_pagination() -> None:
    """If a page is byte-for-byte identical to the previous page, the
    server is ignoring limit/offset; return the first page and stop.
    See D-106."""
    t = ScriptedTransport()
    t.expect("POST", "/authentication", _login_ok())
    same_page = _page(
        [_field(f"op{i}") for i in range(3)]  # full page (== page_limit)
    )
    t.expect("GET", "/data-fields", same_page, same_page, same_page)

    fields, dropped, pages = _builder(
        t, page_limit=3, max_pages=200
    ).fetch_fields()

    # We stopped after seeing the same page twice; fields are from page 1.
    assert pages == 2
    assert {f.field_id for f in fields} == {"op0", "op1", "op2"}
    assert dropped == ()

def test_fetch_fields_includes_settings_in_query() -> None:
    """The /data-fields endpoint requires instrumentType, region,
    delay, universe as query parameters. See D-107."""
    t = ScriptedTransport()
    t.expect("POST", "/authentication", _login_ok())
    t.expect(
        "GET", "/data-fields",
        _page([_field("close")]),
    )

    b = PoolBuilder(
        credentials=_creds(),
        transport=t,
        settings=BrainSettings(
            instrument_type="EQUITY",
            region="USA",
            universe="TOP3000",
            delay=1,
        ),
        page_limit=50,
        max_pages=10,
    )
    b.fetch_fields()

    urls = [url for method, url, _ in t.calls if method == "GET"]
    assert len(urls) == 1
    url = urls[0]
    assert "instrumentType=EQUITY" in url
    assert "region=USA" in url
    assert "delay=1" in url
    assert "universe=TOP3000" in url
    assert "limit=50" in url
    assert "offset=0" in url


def test_fetch_operators_omits_settings_from_query() -> None:
    """The /operators endpoint does not accept settings parameters;
    sending them would be harmless but we don't. See D-107."""
    t = ScriptedTransport()
    t.expect("POST", "/authentication", _login_ok())
    t.expect("GET", "/operators", _page([{"name": "add"}]))

    b = PoolBuilder(
        credentials=_creds(),
        transport=t,
        settings=BrainSettings(region="CHN"),
        page_limit=50,
        max_pages=10,
    )
    b.fetch_operator_names()

    urls = [
        url for method, url, _ in t.calls
        if method == "GET" and "/operators" in url
    ]
    assert len(urls) == 1
    assert "region" not in urls[0]


def test_builder_settings_default_is_usa_top3000() -> None:
    """Constructor without settings uses BrainSettings defaults."""
    t = ScriptedTransport()
    t.expect("POST", "/authentication", _login_ok())
    t.expect("GET", "/data-fields", _page([_field("close")]))

    b = PoolBuilder(credentials=_creds(), transport=t)
    b.fetch_fields()

    urls = [url for method, url, _ in t.calls if method == "GET"]
    assert "region=USA" in urls[0]
    assert "universe=TOP3000" in urls[0]
    assert "delay=1" in urls[0]


def test_fetch_retries_on_429() -> None:
    """A 429 is retried after Retry-After; the second attempt succeeds.
    See D-108."""
    t = ScriptedTransport()
    t.expect("POST", "/authentication", _login_ok())
    t.expect(
        "GET", "/data-fields",
        make_response(429, headers={"Retry-After": "0.01"}),
        _page([_field("close")]),
    )

    sleeps: list[float] = []
    b = PoolBuilder(
        credentials=_creds(),
        transport=t,
        sleep_fn=sleeps.append,
        page_limit=50,
        max_pages=10,
        max_retries_on_429=2,
    )
    fields, _, _ = b.fetch_fields()

    assert [f.field_id for f in fields] == ["close"]
    assert sleeps == [0.01]


def test_fetch_429_reads_brain_ratelimit_reset_header() -> None:
    """BRAIN uses `ratelimit-reset` when `Retry-After` is absent.
    See D-108."""
    t = ScriptedTransport()
    t.expect("POST", "/authentication", _login_ok())
    t.expect(
        "GET", "/data-fields",
        make_response(429, headers={"ratelimit-reset": "7"}),
        _page([_field("close")]),
    )

    sleeps: list[float] = []
    b = PoolBuilder(
        credentials=_creds(),
        transport=t,
        sleep_fn=sleeps.append,
        max_retries_on_429=2,
    )
    b.fetch_fields()
    assert sleeps == [7.0]


def test_fetch_429_without_headers_uses_default() -> None:
    t = ScriptedTransport()
    t.expect("POST", "/authentication", _login_ok())
    t.expect(
        "GET", "/data-fields",
        make_response(429),  # no headers
        _page([_field("close")]),
    )

    sleeps: list[float] = []
    b = PoolBuilder(
        credentials=_creds(),
        transport=t,
        sleep_fn=sleeps.append,
        max_retries_on_429=2,
        retry_after_default=3.5,
    )
    b.fetch_fields()
    assert sleeps == [3.5]


def test_fetch_gives_up_after_max_retries() -> None:
    t = ScriptedTransport()
    t.expect("POST", "/authentication", _login_ok())
    t.expect(
        "GET", "/data-fields",
        *[make_response(429, headers={"Retry-After": "0.01"})] * 3,
    )

    sleeps: list[float] = []
    b = PoolBuilder(
        credentials=_creds(),
        transport=t,
        sleep_fn=sleeps.append,
        max_retries_on_429=2,
    )
    with pytest.raises(PoolBuilderError):
        b.fetch_fields()
    # 2 retries, then give up.
    assert len(sleeps) == 2


def test_builder_rejects_negative_max_retries() -> None:
    with pytest.raises(ValueError):
        PoolBuilder(
            credentials=_creds(),
            transport=ScriptedTransport(),
            max_retries_on_429=-1,
        )


def test_builder_rejects_bad_retry_after_default() -> None:
    with pytest.raises(ValueError):
        PoolBuilder(
            credentials=_creds(),
            transport=ScriptedTransport(),
            retry_after_default=0,
        )


def test_fetch_transport_error_raises() -> None:
    t = ScriptedTransport()
    t.expect("POST", "/authentication", _login_ok())
    t.expect("GET", "/data-fields", TransportError("reset"))
    with pytest.raises(PoolBuilderError):
        _builder(t).fetch_fields()


# --- auth reuse -------------------------------------------------------------

def test_build_reuses_authentication() -> None:
    t = ScriptedTransport()
    t.expect("POST", "/authentication", _login_ok())
    t.expect("GET", "/data-fields", _page([_field("close")]))
    t.expect("GET", "/operators", _page([]))
    _builder(t).build(OperatorSignatureTable.empty())
    login_calls = [c for c in t.calls if c[1].endswith("/authentication")]
    assert len(login_calls) == 1