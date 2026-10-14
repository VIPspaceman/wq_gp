"""Build a Pool from the BRAIN API.

Design decisions:
- Depends on HttpTransport, not requests. See D-081.
- Unknown DataType strings are DROPPED, not guessed. See D-082.
- SemanticKind inference: GROUP -> CATEGORICAL, else NUMERIC. See D-083.
- Operator signatures come from an external JSON table; operators with
  no signature are dropped. See D-084.
- Pagination via limit + offset, with a hard max_pages cap. See D-085.
- The result is a frozen PoolBuildResult. See D-086.
- If a page equals the previous page byte-for-byte, the server is
  ignoring limit/offset; we treat the first page as the whole result.
  See D-106.
- /data-fields requires simulation settings as query parameters; the
  builder takes an optional BrainSettings. See D-107.
- HTTP 429 is retried after the interval advertised in Retry-After or
  ratelimit-reset, up to max_retries_on_429. See D-108.
- Logs go to the "wqgp.pool" channel. Page-level progress is at
  DEBUG; fetch start/end summaries are at INFO. See D-134.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from wqgp.brain.client import (
    BrainAuthError,
    BrainClientError,
    BrainCredentials,
    BrainSettings,
    _BASE_URL,
    _LOGIN_URL,
)
from wqgp.brain.transport import (
    HttpTransport,
    TransportError,
)
from wqgp.field import FieldDef
from wqgp.operator import OperatorDef
from wqgp.pool import FieldPool, OperatorPool, Pool
from wqgp.serialization import (
    SERIALIZATION_SCHEMA_VERSION,
    SerializationError,
    operator_from_dict,
)
from wqgp.types import DataType, SemanticKind


log = logging.getLogger("wqgp.pool")


# --- endpoints --------------------------------------------------------------

_DATA_FIELDS_URL = f"{_BASE_URL}/data-fields"
_OPERATORS_URL = f"{_BASE_URL}/operators"


# --- errors -----------------------------------------------------------------

class PoolBuilderError(RuntimeError):
    """Raised on unrecoverable builder failures (auth, transport, 429
    after all retries exhausted)."""


class PoolBuilderProtocolError(PoolBuilderError):
    """Raised when a server response does not fit the expected shape."""


# --- type mapping (D-082) ---------------------------------------------------

_BRAIN_TYPE_TO_DATATYPE: dict[str, DataType] = {
    "MATRIX": DataType.MATRIX,
    "VECTOR": DataType.VECTOR,
    "GROUP": DataType.GROUP,
    "UNIVERSE": DataType.UNIVERSE,
    "SCALAR": DataType.SCALAR,
}


def _semantic_kind_for(data_type: DataType) -> SemanticKind:
    if data_type is DataType.GROUP:
        return SemanticKind.CATEGORICAL
    return SemanticKind.NUMERIC


# --- result type ------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class PoolBuildResult:
    """Outcome of a build. All tuples are deterministically sorted."""
    pool: Pool
    dropped_field_ids: tuple[str, ...]
    dropped_operator_names: tuple[str, ...]
    fields_fetched: int
    operators_fetched: int
    field_pages_fetched: int
    operator_pages_fetched: int

    def summary(self) -> dict[str, int]:
        return {
            "fields_kept": len(self.pool.fields),
            "fields_dropped": len(self.dropped_field_ids),
            "fields_fetched": self.fields_fetched,
            "operators_kept": len(self.pool.operators),
            "operators_dropped": len(self.dropped_operator_names),
            "operators_fetched": self.operators_fetched,
            "field_pages_fetched": self.field_pages_fetched,
            "operator_pages_fetched": self.operator_pages_fetched,
        }


# --- signature table (D-084) ------------------------------------------------

@dataclass(frozen=True, slots=True)
class OperatorSignatureTable:
    """Mapping from operator name to OperatorDef."""
    _by_name: Mapping[str, OperatorDef]

    def __post_init__(self) -> None:
        object.__setattr__(self, "_by_name", dict(self._by_name))

    def get(self, name: str) -> OperatorDef | None:
        return self._by_name.get(name)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._by_name

    def __len__(self) -> int:
        return len(self._by_name)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._by_name))

    @classmethod
    def empty(cls) -> "OperatorSignatureTable":
        return cls({})

    @classmethod
    def from_dict(cls, data: Any) -> "OperatorSignatureTable":
        if not isinstance(data, dict):
            raise SerializationError(
                f"signature table must be an object, "
                f"got {type(data).__name__}"
            )
        version = data.get("schema_version")
        if version != SERIALIZATION_SCHEMA_VERSION:
            raise SerializationError(
                f"signature table schema_version={version} != "
                f"expected {SERIALIZATION_SCHEMA_VERSION}"
            )
        raw = data.get("operators")
        if not isinstance(raw, list):
            raise SerializationError(
                "signature table 'operators' must be a list"
            )
        by_name: dict[str, OperatorDef] = {}
        for entry in raw:
            op = operator_from_dict(entry)
            if op.name in by_name:
                raise SerializationError(
                    f"duplicate operator in signature table: {op.name!r}"
                )
            by_name[op.name] = op
        return cls(by_name)

    @classmethod
    def from_file(cls, path: Path | str) -> "OperatorSignatureTable":
        p = Path(path)
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            raise SerializationError(
                f"cannot read signature table {p}: {e}"
            ) from e
        return cls.from_dict(data)


# --- pagination helper ------------------------------------------------------

def _extract_items(payload: Any, endpoint: str) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("results", "data", "items"):
            v = payload.get(key)
            if isinstance(v, list):
                return v
        if payload.get("count") == 0:
            return []
    raise PoolBuilderProtocolError(
        f"{endpoint}: response is neither a list nor an object "
        f"with a known list field"
    )


def _settings_query(settings: BrainSettings) -> dict[str, str]:
    return {
        "instrumentType": settings.instrument_type,
        "region": settings.region,
        "delay": str(settings.delay),
        "universe": settings.universe,
    }


def _parse_retry_after(resp: Any, default: float) -> float:
    for header in ("Retry-After", "ratelimit-reset"):
        raw = resp.header(header)
        if raw is None:
            continue
        try:
            wait = float(raw)
        except (TypeError, ValueError):
            continue
        if wait < 0:
            continue
        return wait
    return default


# --- builder ----------------------------------------------------------------

class PoolBuilder:
    """Fetches BRAIN's data fields and operators, and builds a Pool."""

    __slots__ = (
        "_credentials",
        "_transport",
        "_settings",
        "_page_limit",
        "_max_pages",
        "_http_timeout",
        "_sleep_fn",
        "_max_retries_on_429",
        "_retry_after_default",
        "_authenticated",
    )

    def __init__(
        self,
        credentials: BrainCredentials,
        transport: HttpTransport,
        settings: BrainSettings | None = None,
        *,
        page_limit: int = 50,
        max_pages: int = 200,
        http_timeout: float = 30.0,
        sleep_fn: Callable[[float], None] = time.sleep,
        max_retries_on_429: int = 3,
        retry_after_default: float = 5.0,
    ) -> None:
        if page_limit < 1:
            raise ValueError("page_limit must be >= 1")
        if max_pages < 1:
            raise ValueError("max_pages must be >= 1")
        if http_timeout <= 0:
            raise ValueError("http_timeout must be > 0")
        if max_retries_on_429 < 0:
            raise ValueError("max_retries_on_429 must be >= 0")
        if retry_after_default <= 0:
            raise ValueError("retry_after_default must be > 0")
        self._credentials = credentials
        self._transport = transport
        self._settings = settings if settings is not None else BrainSettings()
        self._page_limit = page_limit
        self._max_pages = max_pages
        self._http_timeout = http_timeout
        self._sleep_fn = sleep_fn
        self._max_retries_on_429 = max_retries_on_429
        self._retry_after_default = retry_after_default
        self._authenticated = False

    # --- lifecycle -----------------------------------------------------

    def close(self) -> None:
        self._transport.close()

    # --- auth ----------------------------------------------------------

    def _ensure_authenticated(self) -> None:
        if self._authenticated:
            return
        log.debug("[pool] login: attempting (user=%s)",
                  self._credentials.username)
        try:
            resp = self._transport.request(
                "POST",
                _LOGIN_URL,
                auth=(self._credentials.username, self._credentials.password),
                timeout=self._http_timeout,
            )
        except TransportError as e:
            log.error("[pool] login transport error: %s", e)
            raise PoolBuilderError(f"login transport error: {e}") from e
        if resp.status_code not in (200, 201):
            log.error("[pool] login failed HTTP %d", resp.status_code)
            raise PoolBuilderError(
                f"login failed with HTTP {resp.status_code}"
            )
        self._authenticated = True
        log.debug("[pool] login: OK")

    # --- fetch ---------------------------------------------------------

    def _fetch_paginated(
        self,
        endpoint: str,
        extra_params: Mapping[str, str] | None = None,
    ) -> tuple[list[Any], int]:
        self._ensure_authenticated()
        log.info("[pool] fetching %s", endpoint)

        all_items: list[Any] = []
        offset = 0
        pages = 0
        previous_signature: str | None = None

        sorted_extra: list[tuple[str, str]] = []
        if extra_params:
            sorted_extra = sorted(extra_params.items())

        for _ in range(self._max_pages):
            resp = self._get_page_with_retry(
                endpoint, sorted_extra, offset
            )

            try:
                payload = resp.json()
            except ValueError as e:
                raise PoolBuilderProtocolError(
                    f"{endpoint}: non-JSON body at offset {offset}: {e}"
                ) from e

            items = _extract_items(payload, endpoint)
            pages += 1
            log.debug(
                "[pool] %s page %d: +%d items (total %d)",
                endpoint, pages, len(items), len(all_items) + len(items),
            )

            signature: str | None
            try:
                signature = json.dumps(items, sort_keys=True)
            except (TypeError, ValueError):
                signature = None
            if (
                signature is not None
                and previous_signature is not None
                and signature == previous_signature
            ):
                log.debug(
                    "[pool] %s: repeated page, server ignores pagination",
                    endpoint,
                )
                log.info(
                    "[pool] %s done: %d items from %d pages",
                    endpoint, len(all_items), pages,
                )
                return all_items, pages
            previous_signature = signature

            all_items.extend(items)
            if len(items) < self._page_limit:
                log.info(
                    "[pool] %s done: %d items from %d pages",
                    endpoint, len(all_items), pages,
                )
                return all_items, pages
            offset += len(items)

        raise PoolBuilderError(
            f"{endpoint}: exceeded {self._max_pages} pages at offset "
            f"{offset}. The server is returning distinct full pages "
            f"that ignore limit/offset; check the endpoint's API."
        )

    def _get_page_with_retry(
        self,
        endpoint: str,
        sorted_extra: list[tuple[str, str]],
        offset: int,
    ):
        query_parts = [f"{k}={v}" for k, v in sorted_extra]
        query_parts.append(f"limit={self._page_limit}")
        query_parts.append(f"offset={offset}")
        url = f"{endpoint}?{'&'.join(query_parts)}"

        attempts = 0
        while True:
            try:
                resp = self._transport.request(
                    "GET", url, timeout=self._http_timeout
                )
            except TransportError as e:
                raise PoolBuilderError(
                    f"{endpoint}: transport error at offset {offset}: {e}"
                ) from e

            if resp.status_code == 401:
                self._authenticated = False
                raise PoolBuilderError(
                    f"{endpoint}: 401 at offset {offset}"
                )

            if resp.status_code == 429:
                attempts += 1
                if attempts > self._max_retries_on_429:
                    raise PoolBuilderError(
                        f"{endpoint}: HTTP 429 at offset {offset} "
                        f"after {self._max_retries_on_429} retries"
                    )
                wait = _parse_retry_after(
                    resp, self._retry_after_default
                )
                log.debug(
                    "[pool] %s 429 at offset %d, retry in %.1fs",
                    endpoint, offset, wait,
                )
                self._sleep_fn(wait)
                continue

            if resp.status_code != 200:
                raise PoolBuilderError(
                    f"{endpoint}: HTTP {resp.status_code} at offset {offset}"
                )

            return resp

    # --- field parsing -------------------------------------------------

    def _parse_field(self, raw: Any) -> FieldDef | None:
        if not isinstance(raw, dict):
            return None

        fid = raw.get("id")
        if not isinstance(fid, str) or not fid.strip():
            return None

        raw_type = raw.get("type")
        if not isinstance(raw_type, str):
            return None
        data_type = _BRAIN_TYPE_TO_DATATYPE.get(raw_type.upper())
        if data_type is None:
            return None

        dataset = _extract_dataset_id(raw.get("dataset"))
        if dataset is None:
            return None

        coverage = _coerce_coverage(raw.get("coverage"))
        delay = _coerce_delay(raw.get("delay"))
        unit = raw.get("unit")
        if not isinstance(unit, str):
            unit = None

        try:
            return FieldDef(
                field_id=fid,
                data_type=data_type,
                semantic_kind=_semantic_kind_for(data_type),
                dataset=dataset,
                coverage=coverage,
                delay=delay,
                unit=unit,
            )
        except ValueError:
            return None

    # --- public --------------------------------------------------------

    def fetch_fields(
        self,
    ) -> tuple[tuple[FieldDef, ...], tuple[str, ...], int]:
        extra = _settings_query(self._settings)
        items, pages = self._fetch_paginated(_DATA_FIELDS_URL, extra)
        kept: list[FieldDef] = []
        dropped: list[str] = []
        for raw in items:
            fd = self._parse_field(raw)
            if fd is None:
                raw_id = raw.get("id") if isinstance(raw, dict) else None
                dropped.append(
                    raw_id
                    if isinstance(raw_id, str) and raw_id
                    else "<unknown>"
                )
            else:
                kept.append(fd)
        log.info(
            "[pool] data-fields parsed: %d kept, %d dropped",
            len(kept), len(dropped),
        )
        return tuple(kept), tuple(sorted(dropped)), pages

    def fetch_operator_names(self) -> tuple[tuple[str, ...], int]:
        items, pages = self._fetch_paginated(_OPERATORS_URL)
        names: list[str] = []
        for raw in items:
            if not isinstance(raw, dict):
                continue
            name = raw.get("name") or raw.get("id")
            if isinstance(name, str) and name.strip():
                names.append(name)
        log.info(
            "[pool] operators parsed: %d unique names", len(set(names)),
        )
        return tuple(sorted(set(names))), pages

    def build(
        self,
        signatures: OperatorSignatureTable,
    ) -> PoolBuildResult:
        log.info("[pool] build starting")
        fields, dropped_fields, field_pages = self.fetch_fields()
        names, operator_pages = self.fetch_operator_names()

        kept_ops: list[OperatorDef] = []
        dropped_ops: list[str] = []
        for name in names:
            op = signatures.get(name)
            if op is None:
                dropped_ops.append(name)
            else:
                kept_ops.append(op)

        result = PoolBuildResult(
            pool=Pool(
                fields=FieldPool(fields),
                operators=OperatorPool(kept_ops),
            ),
            dropped_field_ids=dropped_fields,
            dropped_operator_names=tuple(sorted(dropped_ops)),
            fields_fetched=len(fields) + len(dropped_fields),
            operators_fetched=len(names),
            field_pages_fetched=field_pages,
            operator_pages_fetched=operator_pages,
        )
        log.info(
            "[pool] build done: %d fields, %d operators",
            len(result.pool.fields), len(result.pool.operators),
        )
        return result


# --- small coercion helpers -------------------------------------------------

def _extract_dataset_id(raw: Any) -> str | None:
    if isinstance(raw, str) and raw.strip():
        return raw
    if isinstance(raw, dict):
        for key in ("id", "name"):
            v = raw.get(key)
            if isinstance(v, str) and v.strip():
                return v
    return None


def _coerce_coverage(raw: Any) -> float:
    if isinstance(raw, bool):
        return 1.0
    if isinstance(raw, (int, float)):
        value = float(raw)
        if 0.0 <= value <= 1.0:
            return value
    return 1.0


def _coerce_delay(raw: Any) -> int:
    if isinstance(raw, bool):
        return 1
    if isinstance(raw, int):
        return raw if raw >= 0 else 1
    return 1