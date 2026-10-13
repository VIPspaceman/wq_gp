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

BRAIN does not return operator type signatures. The /operators endpoint
gives names and descriptions only. Without signatures, an OperatorDef
would be useless: the AST layer's type checker needs input_types,
output_type, and param_domains. This builder therefore requires a
signature table and drops operators it cannot build.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from wqgp.brain.client import (
    BrainAuthError,
    BrainClientError,
    BrainCredentials,
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


# --- endpoints --------------------------------------------------------------

_DATA_FIELDS_URL = f"{_BASE_URL}/data-fields"
_OPERATORS_URL = f"{_BASE_URL}/operators"


# --- errors -----------------------------------------------------------------

class PoolBuilderError(RuntimeError):
    """Raised on unrecoverable builder failures (auth, transport)."""


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
    """D-083: the only inference we do. GROUP -> CATEGORICAL, else NUMERIC."""
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
    """Mapping from operator name to OperatorDef.

    Loaded from a JSON file shaped like:
        {
          "schema_version": 1,
          "operators": [ { ...operator_to_dict output... }, ... ]
        }
    """
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


# --- pagination helper (D-085, D-106) ---------------------------------------

def _extract_items(payload: Any, endpoint: str) -> list[Any]:
    """BRAIN list endpoints return either a bare list or an object with
    a 'results' / 'data' / 'items' field. Accept both."""
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


# --- builder ----------------------------------------------------------------

class PoolBuilder:
    """Fetches BRAIN's data fields and operators, and builds a Pool.

    Lifecycle:
        builder = PoolBuilder(credentials, transport)
        result = builder.build(signatures)
        # result.pool is ready to hand to Coordinator

    Not thread-safe: login state is mutated on the first call. Use one
    builder per thread, or serialize calls.
    """

    __slots__ = (
        "_credentials",
        "_transport",
        "_page_limit",
        "_max_pages",
        "_http_timeout",
        "_authenticated",
    )

    def __init__(
        self,
        credentials: BrainCredentials,
        transport: HttpTransport,
        *,
        page_limit: int = 50,
        max_pages: int = 200,
        http_timeout: float = 30.0,
    ) -> None:
        if page_limit < 1:
            raise ValueError("page_limit must be >= 1")
        if max_pages < 1:
            raise ValueError("max_pages must be >= 1")
        if http_timeout <= 0:
            raise ValueError("http_timeout must be > 0")
        self._credentials = credentials
        self._transport = transport
        self._page_limit = page_limit
        self._max_pages = max_pages
        self._http_timeout = http_timeout
        self._authenticated = False

    # --- lifecycle -----------------------------------------------------

    def close(self) -> None:
        self._transport.close()

    # --- auth ----------------------------------------------------------

    def _ensure_authenticated(self) -> None:
        if self._authenticated:
            return
        try:
            resp = self._transport.request(
                "POST",
                _LOGIN_URL,
                auth=(self._credentials.username, self._credentials.password),
                timeout=self._http_timeout,
            )
        except TransportError as e:
            raise PoolBuilderError(f"login transport error: {e}") from e
        if resp.status_code not in (200, 201):
            raise PoolBuilderError(
                f"login failed with HTTP {resp.status_code}"
            )
        self._authenticated = True

    # --- fetch ---------------------------------------------------------

    def _fetch_paginated(
        self, endpoint: str
    ) -> tuple[list[Any], int]:
        """Fetch every page from `endpoint`. Returns (items, pages_fetched).

        Loop termination: a page shorter than page_limit marks the end.
        Additionally, if a page equals the previous page byte-for-byte,
        the server is ignoring limit/offset and we treat the first page
        as the whole result. See D-106.
        """
        self._ensure_authenticated()
        all_items: list[Any] = []
        offset = 0
        pages = 0
        previous_signature: str | None = None

        for _ in range(self._max_pages):
            url = (
                f"{endpoint}"
                f"?limit={self._page_limit}&offset={offset}"
            )
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
            if resp.status_code != 200:
                raise PoolBuilderError(
                    f"{endpoint}: HTTP {resp.status_code} at offset {offset}"
                )

            try:
                payload = resp.json()
            except ValueError as e:
                raise PoolBuilderProtocolError(
                    f"{endpoint}: non-JSON body at offset {offset}: {e}"
                ) from e

            items = _extract_items(payload, endpoint)
            pages += 1

            # Detect a repeated page: the server ignored limit/offset
            # and returned the same payload. The first page IS the whole
            # result. See D-106.
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
                return all_items, pages
            previous_signature = signature

            all_items.extend(items)
            if len(items) < self._page_limit:
                return all_items, pages
            offset += len(items)

        raise PoolBuilderError(
            f"{endpoint}: exceeded {self._max_pages} pages at offset "
            f"{offset}. The server is returning distinct full pages "
            f"that ignore limit/offset; check the endpoint's API."
        )

    # --- field parsing -------------------------------------------------

    def _parse_field(self, raw: Any) -> FieldDef | None:
        """Return a FieldDef, or None if the field cannot be built
        (missing required keys or unknown DataType; see D-082)."""
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
        """Fetch every field. Returns (fields, dropped_ids, pages)."""
        items, pages = self._fetch_paginated(_DATA_FIELDS_URL)
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
        return tuple(kept), tuple(sorted(dropped)), pages

    def fetch_operator_names(self) -> tuple[tuple[str, ...], int]:
        """Fetch every operator name. Returns (names, pages)."""
        items, pages = self._fetch_paginated(_OPERATORS_URL)
        names: list[str] = []
        for raw in items:
            if not isinstance(raw, dict):
                continue
            name = raw.get("name") or raw.get("id")
            if isinstance(name, str) and name.strip():
                names.append(name)
        return tuple(sorted(set(names))), pages

    def build(
        self,
        signatures: OperatorSignatureTable,
    ) -> PoolBuildResult:
        """Fetch fields + operator names, combine with `signatures`,
        return the assembled Pool and a report."""
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

        return PoolBuildResult(
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


# --- small coercion helpers -------------------------------------------------

def _extract_dataset_id(raw: Any) -> str | None:
    """`dataset` may be a string id or an object {"id":..., "name":...}."""
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