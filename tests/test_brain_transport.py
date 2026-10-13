"""Tests for the HTTP transport layer. Phase 3 Step 6 gate."""
from __future__ import annotations

import json

import pytest

from wqgp.brain.transport import (
    HttpResponse,
    RequestsTransport,
    make_response,
)


# --- HttpResponse ----------------------------------------------------------

def test_http_response_header_lookup_case_insensitive() -> None:
    r = HttpResponse(
        status_code=200,
        headers=(("Content-Type", "application/json"), ("Retry-After", "5")),
        body=b"{}",
    )
    assert r.header("content-type") == "application/json"
    assert r.header("RETRY-AFTER") == "5"
    assert r.header("Missing") is None


def test_http_response_header_map() -> None:
    r = HttpResponse(
        status_code=200,
        headers=(("A", "1"), ("B", "2")),
        body=b"",
    )
    assert r.header_map() == {"A": "1", "B": "2"}


def test_http_response_json() -> None:
    r = HttpResponse(
        status_code=200,
        headers=(),
        body=json.dumps({"x": 1}).encode("utf-8"),
    )
    assert r.json() == {"x": 1}


def test_http_response_json_malformed() -> None:
    r = HttpResponse(status_code=200, headers=(), body=b"not json")
    with pytest.raises(ValueError):
        r.json()


def test_http_response_is_frozen_and_hashable() -> None:
    r = HttpResponse(status_code=200, headers=(), body=b"")
    with pytest.raises(Exception):
        r.status_code = 500  # type: ignore[misc]
    assert hash(r) == hash(
        HttpResponse(status_code=200, headers=(), body=b"")
    )


# --- make_response ---------------------------------------------------------

def test_make_response_with_json() -> None:
    r = make_response(200, json_body={"a": 1})
    assert r.status_code == 200
    assert r.json() == {"a": 1}


def test_make_response_with_headers() -> None:
    r = make_response(
        201, json_body={}, headers={"Location": "/x"}
    )
    assert r.header("location") == "/x"


def test_make_response_with_raw_body() -> None:
    r = make_response(500, body=b"server error")
    assert r.body == b"server error"


def test_make_response_empty_body() -> None:
    r = make_response(204)
    assert r.body == b""


# --- RequestsTransport (smoke) ---------------------------------------------

def test_requests_transport_close_is_idempotent() -> None:
    t = RequestsTransport()
    t.close()
    t.close()  # must not raise