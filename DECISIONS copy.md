# Decisions & Optimizations Log

Every deviation from the reference project (VIPspaceman/wq) is recorded here
with an ID, the "why", and a "revisit if". If something breaks later and you
can't remember why a choice was made, search this file first.

---

## D-001 — CATEGORICAL is a SemanticKind, not a DataType

**Decision**: `DataType` contains only `MATRIX / VECTOR / GROUP / UNIVERSE /
SCALAR`. `CATEGORICAL` lives in a separate `SemanticKind` enum.

**Why**: BRAIN's API returns `"GROUP"` for sector / industry / country fields.
If we had mapped categorical to its own `DataType`, every API response would
fail type validation at the boundary.

**Revisit if**: BRAIN ever introduces a native `CATEGORICAL` type, or if we
start ingesting fields from a non-BRAIN source with different vocabulary.

---

## D-002 — ParamDomain is a flat frozen dataclass with a discriminant

**Decision**: one `ParamDomain` dataclass, discriminated by `ParamKind`,
instead of subclasses (`IntRange`, `FloatRange`, `Choice`).

**Why**:
- JSON round-trip is trivial (no polymorphic loader / registry).
- Frozen equality works out of the box for operator fingerprints.
- The field set is small; the discriminant keeps it readable.

**Revisit if**: parameter domains multiply beyond ~4 kinds, or if we need
kind-specific behavior that doesn't fit in `contains()`.

---

## D-003 — `DataType(str, Enum)` for JSON-friendliness

**Decision**: enums subclass `str` and use uppercase values matching BRAIN.

**Why**: JSON serialization, log readability, and direct comparison against
the platform vocabulary without a custom encoder.

**Revisit if**: we ever need to version the vocabulary (e.g. BRAIN adds a
type), in which case we may want a separate `KNOWN_TYPES` set for validation.

---

## D-004 — `slots=True` on frozen dataclasses

**Decision**: use `@dataclass(frozen=True, slots=True)` for small value
objects (ParamDomain, and later FieldDef / OperatorDef / ASTNode).

**Why**: lowers per-instance memory; populations can hold 10k+ ASTNodes.

**Revisit if**: we need `__dict__` for pickling across process boundaries
or dynamic attribute injection. (Neither is planned.)

---

## D-005 — No serialization on value objects yet

**Decision**: `FieldDef` (and later `OperatorDef` / `ASTNode`) do NOT get
`to_dict` / `from_dict` in Phase 1.

**Why**: the on-disk schema for fields should be decided by the Phase 2
loader, informed by what BRAIN actually returns, not guessed now. Adding
serialization before the loader exists almost guarantees a rewrite.

**Revisit if**: a Phase 2 component needs persistence before the loader is
written. If so, add a single `serialization.py` module — do NOT scatter
`to_dict` across every dataclass.

---

## D-006 — CATEGORICAL ⟹ GROUP is one-way

**Decision**: `FieldDef` enforces `semantic_kind=CATEGORICAL` ⟹
`data_type=GROUP`. It does NOT enforce the reverse.

**Why**: In BRAIN, all GROUP fields happen to be categorical today. But a
future dataset may expose a GROUP field that is not a categorical bucket
(e.g. an ordinal or a user-defined grouping). Enforcing the reverse now
would make the Phase 2 loader explode on the first such field.

**Revisit if**: we discover a field where `CATEGORICAL` + non-GROUP is
legitimate, or a GROUP field where the semantic kind genuinely matters
for operator dispatch. Document the case here before relaxing the check.

---

## D-007 — `arity` counts DATA arguments only

**Decision**: `OperatorDef.arity == len(input_types)` is the number of data
arguments. Parameters (window, k, lookback, ...) live in `param_domains` and
are NOT counted by `arity`.

**Why**: BRAIN syntax is `op(data_0, ..., data_n, param_0, ..., param_m)`,
where params are things like `ts_mean(close, 5)` — the `5` is not a data
argument, it is a bounded integer parameter. The reference project collapses
both into one arity; every downstream type check then has to guess which
positions are data and which are params, and gets it wrong as soon as a
parametric operator appears.

`total_args = arity + len(param_domains)` gives the flat call arity when
something (parser, pretty-printer) needs it.

**Revisit if**: BRAIN introduces an operator where params may appear
interleaved with data args in the call syntax, e.g. `foo(a, 5, b)`. Then
this model breaks and we need an explicit positional signature.

---

## D-008 — `associative` does NOT imply `commutative`

**Decision**: `OperatorDef` does NOT enforce `associative ⟹ commutative`.
Setting `associative=True, commutative=False` is legal.

**Why**: the implication is mathematically false. Matrix multiplication is
associative and not commutative; string concatenation is associative and
not commutative. In the earlier anchor I had written "associative ⟹
commutative" as an invariant to enforce. That was wrong and is corrected
here. The earlier claim also belonged to the `ASTNode` normalization logic,
not to the operator signature — see D-010 when ASTNode lands.

The only cross-flag invariant enforced is: `commutative | associative |
symmetric` all require `arity >= 2`. Unary operators cannot carry any of
them.

**Revisit if**: we discover an operator in BRAIN whose actual semantics
contradict a declared flag. Fix the flag; do not add a rule to the
validator to compensate.

---

## D-009 — per-position union input types

**Decision**: `input_types` accepts, per argument position, either a single
`DataType` or a tuple of acceptable `DataType`s. Internally normalized to
`tuple[frozenset[DataType], ...]`.

**Why**: real BRAIN operators accept a range of types per position — for
example many `ts_*` operators accept `MATRIX | VECTOR`, and `group_rank`
takes `(MATRIX | VECTOR, GROUP)`. A single `DataType` per position would
force us to either over-approximate (accept anything) or duplicate operator
entries. Neither is acceptable for a type-checked GP.

**Revisit if**: we need disjunctive constraints *across* positions, e.g.
"arg0 and arg1 must have the same type". That is a real case for operators
like `ts_corr` when we want to forbid `MATRIX` + `VECTOR` mixes. If it
appears, add a small `cross_type_constraint` field rather than overloading
`input_types`.

---

## D-010 — No associative flattening in Phase 1

**Decision**: fingerprint normalization sorts children only. It does NOT
flatten associative operators, i.e. `add(add(a, b), c)` stays structurally
distinct from `add(a, add(b, c))`.

**Why**: `OperatorDef` declares a fixed arity, and `ASTNode.__post_init__`
enforces `len(children) == op.arity`. Flattening would produce a CALL with
arity 3 to a binary operator and be rejected at construction. Supporting
flattening requires a variable-arity operator model, which is not worth
building until the GP core (Phase 3) actually needs it.

Consequence: some alpha-equivalent expressions get distinct fingerprints.
That is a small loss in dedup rate, not a correctness bug.

**Revisit if**: Phase 3 diversity metrics show duplicate-equivalent
individuals crowding the population. At that point, either (a) add an
optional `flatten_associative` flag to `canonical()`, or (b) introduce
variadic operators. Do NOT retro-fit flattening into Phase 1.

---

## D-011 — SCALAR broadcast is restricted to numeric positions

**Decision**: a LITERAL node (type SCALAR) may occupy a data-argument
position declared as MATRIX, VECTOR, or SCALAR. It may NOT occupy GROUP or
UNIVERSE positions.

**Why**: BRAIN expressions like `add(close, 1)` are idiomatic and must type
check. But `group_rank(close, 5)` must not — the second arg is a grouping
key, and a scalar constant there is either a user error or a silent type
coercion that will corrupt every downstream grouping operator.

**Revisit if**: BRAIN documents a legitimate scalar-in-group-position use,
e.g. a numeric bucket size in some `bucket()` operator. If so, special-case
by operator name rather than weakening the general rule.

---

## D-012 — Integer-valued float literals normalize to int in fingerprints

**Decision**: `_norm_scalar` maps `5.0` to `"5"`. So `ts_mean(close, 5)` and
`ts_mean(close, 5.0)` share a fingerprint.

**Why**: BRAIN params are usually specified as integers, but the JSON
pipeline can smuggle `5.0` in. Without normalization, the two forms become
distinct individuals in the population, doubling the cost of exploring what
is semantically one point.

**Scope note**: this normalization only fires on parameters where both
`int` and `float` inputs are legal — i.e. `FLOAT_RANGE`. On `INT_RANGE`,
`ParamDomain.contains` rejects `5.0` outright at construction (see
`test_int_range_rejects_bool_and_float`), so the fingerprint layer never
sees a float there. That is intentional: strict at the boundary, lenient
at the fingerprint. Two different layers, two different jobs.

**Revisit if**: we ever support a parameter whose semantics genuinely
distinguish `5` from `5.0` (e.g. a dimensionless exponent). At that point,
make the normalization opt-out per ParamDomain.

---

## D-013 — Layered budget, all-or-nothing consumption

**Decision**: budget has three layers — `total_simulations` (hard),
`per_generation_simulations` (soft, reset each generation), and
`wall_time_seconds` (hard). Consumption is atomic: a failed check leaves
the tracker unchanged. Exhaustion raises `BudgetExhausted`.

**Why**: silent over-consumption is a concurrency bug waiting to happen —
two workers can each see "1 simulation left" and both proceed. Raising on
the very first overrun makes that visible immediately. All-or-nothing
consumption means a partially-debited tracker is never observable.

**Revisit if**: we need soft quota signals for adaptive scheduling (e.g.
"slow down when 80% consumed"). That is a separate observer, not a change
to the hard-check semantics.

---

## D-014 — Checkpoint carries metadata + opaque payload

**Decision**: `Checkpoint` stores schema/run/generation/budget/RNG/time
plus a `payload: dict` that this module never inspects. Phase 3 puts the
population, evaluation cache, etc. inside `payload`.

**Why**: if `checkpoint.py` knew what a population looked like, every
Phase 3 refactor of the population would ripple into the storage layer.
Keeping the payload opaque means one module changes at a time.

**Revisit if**: we need partial reads — e.g. "load just the population,
skip the cache" — at which point `payload` gains a documented schema. Do
NOT split into multiple files until the need is concrete.

---

## D-015 — Checkpoints are single JSON files

**Decision**: one `step_NNNNNN.json` per generation, plus a `latest.txt`
pointer. No SQLite in Phase 2 Step 1.

**Why**: Step 1's goal is "run / stop / resume". JSON files are
inspectable, diffable, trivially backed up, and dependency-free. SQLite
earns its keep only when we have thousands of evaluation records to query,
which is a Step 3 concern.

**Revisit if**: the number of generations per run exceeds a few hundred,
or resume latency becomes noticeable. Migration path: keep the JSON
format as the on-disk source of truth, add a SQLite index over it.

---

## D-016 — RNG state is serialized explicitly

**Decision**: `Checkpoint.capture()` snapshots `random.Random.getstate()`,
and `Checkpoint.restore_rng()` reconstructs a Random from it. Recovery
never re-seeds.

**Why**: re-seeding on resume silently produces a different evolutionary
trajectory, which is the single most common cause of "why did my two runs
diverge?" GP debugging.

**Revisit if**: we switch to a different RNG (numpy PCG64, etc.). Then
this field gains a discriminator (`rng_kind`) so old checkpoints can be
rejected rather than misread.

---

## D-017 — Atomic writes via tmp + os.replace

**Decision**: every file write goes to `<target>.tmp` first, then
`os.replace(tmp, target)`. Both `fsync` and the rename happen before the
call returns.

**Why**: `os.replace` is atomic on POSIX and Windows. A crash mid-write
leaves either the old file or the new one — never a half-written JSON that
`json.load` will choke on.

**Revisit if**: we move to a database with its own durability guarantees.

---

## D-018 — schema_version and run_id validation

**Decision**: every checkpoint file carries `schema_version`. On load, a
mismatch raises `IncompatibleSchemaError`. `run_id` is validated against
`^[A-Za-z0-9._-]+$` and must not contain `..`.

**Why**: two failure modes to prevent. (1) A Phase 3 format change that
silently misreads old files is worse than a hard error. (2) A `run_id`
containing `/` or `..` would let a malicious or buggy caller write outside
the checkpoint root.

**Revisit if**: we need backward compatibility across schema versions —
then add a migration function keyed by `schema_version` rather than
loosening the check.

---

## D-020 — Serialization is a separate module, not a method on value objects

**Decision**: all `to_dict` / `from_dict` logic lives in
`serialization.py`. `FieldDef`, `OperatorDef`, `ParamDomain` do not carry
any serialization code.

**Why**: this is the concrete enforcement of D-005. Two benefits.
(1) Value objects stay pure: their `__post_init__` only knows about
invariants, not about wire formats. (2) Wire formats change; when they do,
only one file changes.

**Revisit if**: we ever need a second wire format (e.g. compact binary).
Then add a `binary_serialization.py` next to this one, do not overload
this module.

---

## D-021 — Schema version lives at the file level

**Decision**: `SERIALIZATION_SCHEMA_VERSION` is checked once, when loading
the top-level pool payload. Individual `FieldDef` / `OperatorDef` dicts
carry no version field.

**Why**: if each object carried its own version, a schema bump would
require editing every producer and every consumer. One version for the
whole file means one place to bump and one place to check.

**Revisit if**: we start embedding fragments of a pool file inside other
files (e.g. a checkpoint carrying a partial pool). At that point those
fragments need their own version, and this decision becomes "one version
per serialized document" rather than "one version per pool".

---

## D-022 — Pool indexes are deterministically ordered

**Decision**: `FieldPool.by_type()` returns fields sorted by `field_id`;
`OperatorPool.by_arity()` and `.by_output_type()` return operators sorted
by `name`. `Pool.to_dict()` emits fields and operators in those same
sorted orders.

**Why**: GP needs reproducibility (see D-016). If population
initialisation iterates a pool in hash-set order, two runs with the same
seed produce different populations. Sorting once at pool-build time makes
downstream code trivially deterministic without sprinkling `sorted()`
everywhere.

**Revisit if**: sorting becomes a measurable cost at scale (unlikely —
pools are hundreds, not millions, of entries). Do NOT switch to
hash-ordered iteration to "save time"; if you need random order, use the
seeded RNG.

---

## D-023 — Pool construction validates uniqueness only

**Decision**: `FieldPool` rejects duplicate `field_id`; `OperatorPool`
rejects duplicate `name`. Nothing else is checked. In particular, an
operator whose `input_types` reference a `DataType` with no corresponding
field is NOT an error at pool-build time.

**Why**: hard errors are reserved for things that make the data
uninterpretable. Semantic gaps ("no VECTOR fields in this dataset")
are legitimate for many runs — an alpha miner that only uses MATRIX
fields is fine. Making that a pool-build error would force every caller
to add a flags-out parameter to disable the check.

Consumers that want the check can ask: `len(pool.fields.by_type(dt)) == 0`.

**Revisit if**: we accumulate enough callers duplicating that check that
it earns a dedicated `Pool.warnings()` method. Return strings, not raise.

---

## D-024 — `input_types` JSON form: string for single, list for union

**Decision**: in the pool file, each position of `input_types` is either
a string (single type) or a list of strings (union). Serializer emits a
string when the position's normalized frozenset has size 1, and a sorted
list otherwise.

**Why**: readability. `["MATRIX", "MATRIX"]` is what most operator
entries look like; forcing `[["MATRIX"], ["MATRIX"]]` everywhere just to
accommodate the rare union case makes the file worse. The parser handles
both, so no information is lost.

**Revisit if**: a downstream tool (schema validator, code generator)
needs a uniform shape. Then either normalize on read or accept the
readability hit. Do not add a third accepted form.

---

## D-025 — Pool classes implement __eq__ / __hash__ explicitly

**Decision**: `FieldPool` and `OperatorPool` define `__eq__` (by their
internal by-id/by-name mapping) and `__hash__` (by the sorted key tuple).
They are NOT dataclasses; they are plain classes with these two dunders
written by hand.

**Why**: `Pool` is a frozen dataclass, so its auto-generated `__eq__`
compares its two members. Without `__eq__` on the pools, that comparison
falls back to object identity and two pools with identical contents
compare unequal — which silently breaks every round-trip test and every
dedup-on-pool check. This surfaced as three failures in Phase 2 Step 2;
the production code was wrong, not the tests.

We hash by `field_id` / `name` only, because duplicate keys are rejected
at construction (D-023), so the key set fully identifies the pool. The
inner `MappingProxyType` is unhashable, hence we cannot hash it directly.

**Revisit if**: `FieldPool` / `OperatorPool` ever gain state that is not
derivable from their by-id/by-name mapping. At that point extend both
dunders to include the new state; do NOT make them dataclasses as a
shortcut — the derived indexes must stay derived.

---

---

## D-026 — run_id validation extracted to _ids.py

**Decision**: `validate_run_id` lives in `src/wqgp/_ids.py`. Both
`checkpoint.py` and `persistence.py` import it. No local copies.

**Why**: two stores sharing the same namespace must agree on what a
legal run_id is. If one rejects `run.1` and the other accepts it, a run
can exist in the DB but not on disk (or vice versa), producing bugs that
only appear during resume.

**Revisit if**: a third store appears with different constraints — e.g.
a remote API that forbids dots. At that point add a second validator
with a name that says what it is (`validate_api_run_id`), rather than
loosening this one.

---

## D-027 — EvaluationRecord: explicit fitness + tuple metrics

**Decision**: `EvaluationRecord` carries `fitness: float | None` as a
top-level field, and `metrics: tuple[tuple[str, float], ...]` for
everything else.

**Why**: two forces. (1) `fitness` is the sort key — making it a first-
class column keeps `top_by_fitness` indexed and O(log n). (2) `metrics`
must stay frozen for the record to be frozen; a `dict` inside a frozen
dataclass is a lie. Tuple-of-pairs is hashable, JSON-friendly, and
round-trips through SQLite without a custom encoder.

A `metrics_dict` property gives consumers a dict view without exposing
the internal tuple.

**Revisit if**: metric values need to be non-numeric (strings, nested
JSON). At that point widen the value type to `Any` and update the
serialization path. Do NOT relax to a mutable dict.

---

## D-028 — SQLite user_version as schema version

**Decision**: the DB's schema version lives in `PRAGMA user_version`,
not in a dedicated `schema_meta` table.

**Why**: `user_version` is a 4-byte integer stored in the DB header,
read with one pragma, and set with one pragma. A dedicated table means
one more round-trip and one more thing to get wrong during bootstrap.

**Revisit if**: we need to version multiple independent parts of the
schema (e.g. evaluations + alphas + runs have different versions). Then
a `schema_meta` table with one row per component becomes justified.

---

## D-029 — Upsert preserves created_at, refreshes updated_at

**Decision**: `INSERT ... ON CONFLICT (run_id, fingerprint) DO UPDATE`
updates every column except `created_at`. `updated_at` is refreshed to
`now()` on every write.

**Why**: `created_at` answers "when did we first see this fingerprint in
this run", which is what resume and lineage checks need. If we
overwrote it, a re-evaluation would erase the original discovery time.

`fitness` and `metrics` are overwritten because a later evaluation with
better parameters is strictly more useful than the earlier one.

**Revisit if**: we need to keep a full history of every evaluation of a
fingerprint. Then the table becomes append-only, `UNIQUE` is dropped,
and `created_at` becomes per-row. Do NOT bolt history onto the current
upsert.

---

## D-030 — One connection, one lock; not a pool

**Decision**: `EvaluationStore` holds a single `sqlite3.Connection`
(created with `check_same_thread=False`) and serializes every method on
a `threading.Lock`.

**Why**: WAL mode allows multiple readers + one writer concurrently, but
that only helps if each thread has its own connection. Opening a pool
now would add connection lifecycle, cleanup, and "which connection am I
on" bugs for a benefit we do not yet measure. The executor (Phase 2
Step 4) will do at most tens of writes per second; a single locked
connection is fine at that rate.

**Revisit if**: profiling shows the lock is a bottleneck. Then introduce
a small pool (2–4 read connections + 1 write connection), keeping the
current API. Do NOT switch to `check_same_thread=True` and hope.

---

## D-031 — Three indexes, chosen for the queries we actually run

**Decision**: indexes on `(run_id, generation)`, `(run_id, fitness DESC)`,
and `(run_id, status)`.

**Why**: each index matches a specific query.
- `list_by_generation` → `(run_id, generation)`
- `top_by_fitness`     → `(run_id, fitness DESC)` (the DESC is load-bearing
  so SQLite can scan forward)
- `count_by_status`    → `(run_id, status)`

No index on `fingerprint` alone: it is never queried without `run_id`,
and `(run_id, fingerprint)` is already the unique key.

**Revisit if**: new query shapes appear in Phase 3. Add an index when a
query is measurably slow, not "just in case". Every extra index slows
writes.

---

## D-032 — Budget is consumed at submit time, not at completion

**Decision**: `Executor.submit()` calls `tracker.consume_simulation(1)`
before enqueuing. If it raises `BudgetExhausted`, `submit` returns
`False` and the task is never queued.

**Why**: the alternative — consuming when the evaluation finishes —
allows two workers to each see "1 slot left" and both proceed, because
neither has completed yet when the other checks. That is a classic TOCTOU
race on a shared counter. Consuming at submit makes the check and the
debit the same critical section.

Consequence: if a submitted task is dropped (worker crash, process
kill), the simulation is still counted. That is correct — the platform
saw the request. Refunding on failure would let a retry loop spin past
the budget by reusing consumed slots.

**Revisit if**: we add a "dry run" mode where no platform call is made.
Then introduce a second executor path that skips `consume_simulation`,
rather than weakening this one.

---

## D-033 — Evaluator is a Protocol, not an ABC

**Decision**: `Evaluator` is a `typing.Protocol` with a single
`__call__(task) -> TaskResult` signature. Callers pass a plain callable
(closure, function, bound method, or a class that defines `__call__`).

**Why**: tests want `lambda t: TaskResult(...)` without subclassing. The
real BRAIN client already exists as a class with its own `__init__`
taking credentials; forcing it to inherit from `Evaluator` means either
multiple inheritance or an adapter wrapper. Protocol + structural typing
removes the ceremony.

**Revisit if**: we need to pass per-evaluator configuration through the
executor (timeouts, retry policy). That is better handled by wrapping
the callable in a configured closure, not by widening this Protocol.

---

## D-034 — Worker swallows all exceptions into FAILED records

**Decision**: `Executor._process` wraps the evaluator call and the store
write in `try/except Exception`. Any exception becomes an
`EvaluationRecord(status=FAILED, error="...")`. Only `KeyboardInterrupt`
and `SystemExit` (which inherit from `BaseException`, not `Exception`)
escape.

**Why**: a GP run evaluates thousands of expressions; a few percent will
be malformed, hit a transient BRAIN error, or trip a bug in the
evaluator. If a worker thread dies on the first one, the run stalls with
no visible cause. Converting to a FAILED record keeps the pipeline
moving and leaves a forensic trail in the store.

The store-write failure path is doubly defensive: if writing the FAILED
record itself raises, we swallow that too and just increment the counter.
A lost record beats a dead thread.

**Revisit if**: we need to distinguish "evaluator bug" from "expression
invalid" for upstream retry logic. Then add a `failure_kind` enum to
`TaskResult` and populate it in the except clauses. Do NOT remove the
catch — always keep the pipeline alive.

---

## D-035 — Task/TaskResult are frozen; TaskStatus is local

**Decision**: `Task` and `TaskResult` are `@dataclass(frozen=True,
slots=True)`. `TaskStatus` is a local enum in `execution.py`, distinct
from `persistence.EvaluationStatus`.

**Why**: frozen value objects make tasks safe to pass across threads and
to hash into sets for dedup. Keeping the enums separate means a change
to storage vocabulary does not silently change the executor's contract,
and vice versa. The mapping between them is a one-line dict in this
module — the coupling is explicit, not implicit.

**Revisit if**: the two enums converge completely. Then merge them, but
do it as a deliberate one-step refactor, not by making one import the
other.

---

## D-036 — queue.Queue + sentinel shutdown

**Decision**: the executor uses `queue.Queue` and N daemon threads. To
shut down, put `None` into the queue `n_workers` times; each worker
returns on `None`. `queue.join()` blocks until all items (including
sentinels) have been `task_done()`'d.

**Why**: stdlib `queue.Queue` is battle-tested, has built-in `task_done`
/ `join` bookkeeping, and does not need a custom condition variable. The
`None` sentinel pattern is idiomatic and works with `join()` because
sentinels also call `task_done()` in the worker's `finally`.

**Revisit if**: we need priority queues or per-worker backpressure.
Then `PriorityQueue` is a drop-in for the same interface; backpressure
would add a bounded queue and a blocking `submit`, which changes the
API and needs its own decision.

---

## D-037 — Stats exposed as frozen snapshots

**Decision**: `Executor.stats()` returns an `ExecutorStats` frozen
dataclass built from a `_MutableStats` internal guarded by a lock.
Callers can hold onto a snapshot without it mutating under them.

**Why**: the alternative — returning a live, mutable counter object —
invites bugs where a caller stores `s = ex.stats()` and later sees
different numbers because the executor kept running. Frozen snapshots
force the caller to be explicit about when they re-read state.

**Revisit if**: snapshot cost matters at high polling frequency. At
current call rates (a few per second at most) it does not.

---

## D-038 — No more "expected N passed" predictions in handoff notes

**Decision**: handoff messages must NOT include a predicted test count.
Instead, list the gate tests by name and instruct the user to run
`pytest -v` and confirm all green.

**Why**: the predicted count was wrong three times in a row (206 vs
predicted 208, then "I carefully counted 19" when it was 22). Each wrong
prediction creates a false signal: the reader either wastes time
hunting for a nonexistent failure, or loses trust in the numbers that
actually matter. Test counts are visible in `pytest` output; guessing
them adds nothing and adds noise.

**Revisit if**: never. This is a no-guess zone. If a handoff note needs
a count, it must be obtained by running the suite, not by counting
functions in a diff.

**Renumbering note**: this entry was originally drafted as D-038 before
Phase 3 started, then Phase 3 decisions were drafted with the same ID.
The conflict is resolved by keeping this entry as D-038 and shifting the
Phase 3 decisions to D-039 onward.

---

## D-038 — No more "expected N passed" predictions in handoff notes

**Decision**: handoff notes must NOT include a predicted test count.
List gate tests by name instead, and instruct the user to run
`pytest -v` and confirm all green.

**Why**: the predicted count was wrong three times in a row (206 vs
predicted 208, then "I carefully counted 19" when it was 22, then the
concurrent executor miscount). Each wrong prediction either sends the
reader hunting for a nonexistent failure or erodes trust in the numbers
that actually matter. Test counts are visible in pytest output; guessing
them adds noise and nothing else.

**Revisit if**: never. This is a no-guess zone. If a handoff note needs
a count, run the suite first.

---

## D-039 — Individual equality is by fingerprint only

**Decision**: `Individual` defines `__eq__` and `__hash__` by hand.
Two Individuals are equal iff their fingerprints are equal. Fitness,
metrics, generation, and parents are ignored for equality.

**Why**: two structural forms of the same expression — e.g.
`add(close, open)` and `add(open, close)`, which share a fingerprint
thanks to commutative normalization — must be one Individual for dedup
purposes. A dataclass-generated `__eq__` would compare `ast` fields
structurally and treat them as distinct, defeating the fingerprint
mechanism.

`hash` is on the fingerprint for the same reason: sets and dicts keyed
by Individual must collapse equivalent expressions.

**Revisit if**: we ever want to distinguish "same expression, different
evaluation history" as separate entities. That is a different concept
(a Trial, not an Individual) and belongs in a separate class.

---

## D-040 — Population is a plain class, not a frozen dataclass

**Decision**: `Population` is a plain class with `__slots__`, not a
`@dataclass(frozen=True)`. It builds its fingerprint index in
`__init__`, wraps the index in `MappingProxyType`, and implements
`__eq__` / `__hash__` by the fingerprint set.

**Why**: the fingerprint index must be derived at construction and
frozen. A dataclass would either store the index as a field (making
equality compare caches, which is wrong) or recompute it on every
lookup (making membership O(n)). A plain class lets the constructor
build the index once and hide the raw list behind a tuple.

`__eq__` compares fingerprint sets, so `Population([a, b])` equals
`Population([b, a])`. `__hash__` hashes the fingerprint frozenset.

**Revisit if**: `Population` needs additional state beyond its members.
At that point either extend both dunders to include the new state, or
split off the state into a wrapper. Do NOT convert to a dataclass as a
shortcut.

---

## D-041 — RandomAstBuilder is stateful and single-use per build

**Decision**: `RandomAstBuilder` stores the RNG, the node budget, and
the pool on the instance. `build()` sets `self._rng` from its argument
and resets `_nodes_left`, then recurses through private methods that
read `self._rng` and `self._nodes_left` directly. It is not safe to
call concurrently, and not safe to interleave two `build()` calls.

**Why**: threading RNG and budget counters as explicit parameters
through five levels of recursion would add the same two arguments to
every method signature. Storing them on the instance keeps signatures
readable at the cost of thread-safety, which is a trade we are happy to
make — population initialisation is a single-threaded phase by design.

A placeholder `random.Random()` is assigned in `__init__` so internal
methods never need a `None` check on `self._rng`.

**Revisit if**: we need parallel population initialisation. Then either
construct one builder per thread (cheapest), or refactor to pass RNG
and budget explicitly. Prefer the per-thread builder.

---

## D-042 — AST generation is bottom-up with three hard limits

**Decision**: `RandomAstBuilder` enforces:
- `max_depth`: recursion depth cap.
- `max_nodes`: total node cap, decremented on every node produced.
- `max_operator_attempts`: retry cap for the whole build, AND for the
  number of alternative operators tried at each call site.

When `_try_operator` fails at any point, it refunds the node slot it
reserved, so a failed attempt costs no budget. When the whole build
fails `max_operator_attempts` times, `build()` raises `GenerationError`.

**Why**: without a node cap, a deep recursion through a pool with high
branching factor can produce astronomically large expressions. Without
a retry cap, a pool with operators that can never be satisfied (e.g.
only `group_rank` but no GROUP field) leads to an infinite loop. The
refund mechanism prevents an unlucky sequence of failures from
starving the remaining budget.

**Revisit if**: profiling shows generation is a bottleneck. Then
increase `max_operator_attempts` before loosening `max_nodes`; large
expressions are the main driver of evaluation cost downstream.

---

## D-043 — Generation failure raises; it does not return None

**Decision**: `RandomAstBuilder.build()` raises `GenerationError` when
it cannot produce an AST. It never returns `None`.

**Why**: the caller must make an explicit decision — retry with a
different RNG, drop the run, or fall back to a simpler builder. A
`None` return invites silent propagation into downstream code that
assumes a valid AST, producing "None is not callable" or a crash far
from the root cause. Making failure loud at the source keeps the
failure mode local.

**Revisit if**: we add a builder that can legitimately produce "no
individual this slot" as a signal — e.g. a type-directed builder that
gives up when a type has no fields. Even then, prefer a distinct
sentinel enum over `None`, and document the contract at the call site.

---

## D-044 — Only SCALAR can be a literal leaf

**Decision**: `RandomAstBuilder._build_leaf` produces a LITERAL node
only when the requested type is `SCALAR`. For any other requested
type (MATRIX, VECTOR, GROUP, UNIVERSE), it produces a FIELD_REF if the
pool has a matching field, otherwise returns `None`.

**Why**: `ASTNode._validate_call` allows a SCALAR literal to occupy a
MATRIX-typed argument position (scalar broadcast, D-011). That is a
type-compatibility rule at CALL sites, not a license to build a
literal-in-MATRIX-slot from scratch. If the builder produced
`add(1, close)`, the AST would be legal, but every generated expression
would risk degenerating into constant folding territory. Starting from
real fields and only using SCALAR where a scalar is genuinely wanted
keeps generated alphas data-driven.

**Revisit if**: we want to explore "constant + field" expressions
explicitly. Then add a dedicated builder mode rather than loosening
this rule; the change should be opt-in and observable in the generated
fingerprints.

---

## D-045 — One mutation per call, no recursion

**Decision**: a single call to a mutation strategy produces exactly one
local edit. If that edit is rejected (fingerprint unchanged, or
`ASTNode.__post_init__` raised), the strategy returns `None` and does
not attempt a second edit internally. The dispatcher retries with a
different strategy, not with the same one.

**Why**: recursive mutation — "mutate, then mutate the result, then
mutate that" — makes the edit distance between parent and child
unbounded. Two consequences. (1) Crossover becomes hard to reason about
because subtrees are effectively uncorrelated. (2) Debugging a run
becomes impossible: you cannot ask "what did this mutation do" and get a
one-line answer.

Keeping each mutation local means the population can maintain a
meaningful edit-distance metric without having to reconstruct it from
fingerprint diffs.

**Revisit if**: we measure that single-step mutations are too weak to
explore the space. Then raise the *number of mutation generations per
offspring* at the pipeline level, not by recursing inside one strategy.

---

## D-046 — Node location is by integer path, not parent pointer

**Decision**: strategies locate the target node by a `tuple[int, ...]`
path from the root (e.g. `(0, 1, 2)` = root's child 0's child 1's child
2). The replacement helpers `_replace_subtree` and `_replace_param`
rebuild the nodes along that path.

**Why**: `ASTNode` is frozen. Giving it a `parent` attribute would
require either a mutable field (breaking `frozen=True` and `__hash__`)
or a back-reference added post-construction (breaking `slots=True` and
`__eq__`). Integer paths are cheap, hashable, and reconstructible.

A `weakref` alternative was considered and rejected: it still needs a
mutable slot, and weakrefs add serialization complexity for checkpoints.

The cost is O(depth) per replacement, which is negligible at our
expression sizes.

**Revisit if**: we add a mutable AST representation (e.g. for a
differentiable or streaming variant). This decision applies to the
frozen representation only.

---

## D-047 — A mutation must change the fingerprint, or it is retried

**Decision**: `Mutator.mutate` rejects any candidate whose fingerprint
equals the input's. It retries with a randomly chosen strategy up to
`max_attempts`, then raises `MutationError`.

**Why**: a mutation that produces an alpha-equivalent AST wastes a
population slot and a simulation budget. This is not hypothetical:
`add(close, open)` and `add(open, close)` share a fingerprint (D-001's
normalization), and a field-swap that just reorders them is a no-op.

Rejecting at the mutation layer is cheaper than rejecting at the
population layer, because the population layer would have to diff
against every existing individual.

**Revisit if**: the retry loop dominates runtime. Then measure the
per-strategy rejection rates and either drop the weak strategies or
improve their target selection. Do NOT remove the check.

---

## D-048 — Operator swap requires matching parameter names

**Decision**: `mutate_operator` only considers candidates where
`new_op.param_names == old_op.param_names`. A same-arity, same-signature
operator with different parameter names (e.g. `ts_mean` with `window`
vs `ts_decay` with `halflife`) is NOT a candidate.

**Why**: the operator's `params` tuple is positional. Swapping to an
operator with different parameter names would silently reinterpret
`window=5` as `halflife=5` — two numbers that mean completely different
things. Catching that at the mutation layer keeps the semantics honest.

The alternative — remap by name and fall back to defaults for missing
params — was rejected because "fall back to defaults" hides a semantic
change behind a numeric coincidence: `ts_mean(close, 5)` becoming
`ts_decay(close, <default halflife>)` is not a small mutation, it is a
rewrite. Rewrites belong in a dedicated transformation, not in a
per-node mutate.

**Revisit if**: we add a "parameter remapping" transform that
explicitly documents the semantic shift. Then that transform, not
`mutate_operator`, is the right place.

---

## D-049 — Subtree swap uses a small-budget builder

**Decision**: `mutate_subtree` constructs a `RandomAstBuilder` with
`max_depth=2, max_nodes=5` for generating replacement subtrees. It does
not reuse the population-init builder's larger budget.

**Why**: subtree mutation replaces an arbitrary subtree, including
internal ones. If the replacement is allowed to be as large as the
original AST, the child's size can grow superlinearly across
generations — a well-known failure mode in tree GP called bloat.

A fixed small budget bounds the per-mutation size delta. It also makes
the mutation more of a local perturbation and less of a wholesale
rewrite, which keeps the lineage traceable.

**Revisit if**: bloat is measured to be negligible with a larger
budget on our pool. Then raise it by an explicit constant and record
the measurement in this file. Do NOT make it proportional to the
parent's size — that reintroduces the feedback loop.

---

## D-050 — "No mutation possible" tests must account for subtree shrink

**Decision**: any test asserting that `Mutator.mutate` raises
`MutationError` must use an AST and a pool where subtree mutation also
cannot change the fingerprint. In practice that means: the AST has no
CALL nodes, and the pool has only one field of that type and no
operators.

**Why**: `mutate_subtree` can *shrink* a tree, not just grow it. Given
`sub(close, close)` and a pool containing `sub`, the builder can
produce `close` as a MATRIX leaf, and replacing the root collapses the
whole tree to `close` — a legitimate, distinct-fingerprint mutation.
The previous test assumed "no alternative operator ⇒ no possible
mutation", which is wrong once subtree replacement is on the table.

This is not a bug in `mutate_subtree`: shrinking is a standard GP
operation and it is what keeps trees from bloating. The test's
premise was wrong, and the fix is in the test setup, not in the
production code.

**Revisit if**: a future strategy makes even a single field-ref
mutable — e.g. a "constant folding" strategy that rewrites `close`
to `close * 1`. Then this minimal pool becomes insufficient again and
needs a different trick (a pool with zero fields would work, at the
cost of being unable to construct the initial AST at all).

---

## D-051 — AST traversal and replacement utilities live in ast.py

**Decision**: `iter_nodes`, `iter_calls`, `iter_field_refs`,
`replace_subtree`, `replace_param`, `tree_depth`, `tree_size`, and the
`Path` type alias live in `ast.py`. `mutation.py` and `crossover.py`
import them from there. The `_`-prefixed private copies in `mutation.py`
are removed.

**Why**: these are pure AST operations with no GP semantics. They were
initially placed in `mutation.py` because that was the first consumer,
but `crossover.py` needs the exact same primitives. Duplicating them
would guarantee divergence; importing private names across modules is
a code smell that hides the real dependency.

Adding them to `ast.py` does not change `ASTNode` itself — the frozen
dataclass, its validators, and its fingerprint are untouched. All
Phase 1 tests continue to pass.

**Revisit if**: the utility set grows large enough that `ast.py`
becomes cluttered. Then split into `ast.py` (the node type) and
`ast_ops.py` (operations). Do NOT move them back into a consumer.

---

## D-052 — Crossover only swaps same-DataType subtrees

**Decision**: `subtree_crossover` collects non-root nodes from both
parents and only considers swap pairs where `node_a.data_type is
node_b.data_type`.

**Why**: the alternative — swapping any two nodes and letting
`ASTNode.__post_init__` reject the bad ones — would discard most
candidate pairs and, worse, could produce a child that type-checks only
by accident (e.g. a MATRIX slot accepting a SCALAR literal via
broadcast). Requiring an exact type match at candidate-collection time
keeps the semantics honest: crossover transplants a subtree into a
position that wanted exactly that kind of subtree.

SCALAR broadcast is deliberately NOT used as a loophole here, even
though it would admit more pairs. Broadcast is for literal leaves
generated by the builder, not for subtree transplantation. See D-011
for the original rule.

**Revisit if**: measurement shows crossover is starved for compatible
pairs. Then consider allowing SCALAR→numeric-position swaps explicitly,
as a separate opt-in flag with its own tests. Do NOT silently widen
the rule.

---

## D-053 — Crossover produces one child, not two

**Decision**: `Crossover.crossover(a, b, rng)` returns a single AST:
a copy of A with one of its subtrees replaced by a subtree of B.

**Why**: textbook GP produces two children per crossover by swapping
in both directions. In practice, the second child (B with a subtree
from A) is redundant for our pipeline: we already generate many
offspring per generation, and the reversed swap can be reproduced by
calling `crossover(b, a, ...)` with a fresh RNG. Producing one child
keeps the dispatcher's API simple and makes the "child must differ
from both parents" check a single comparison against a single result.

If a future design wants symmetric crossover — e.g. to study how the
two directional children differ — the reversed call is one line at the
call site, not a change here.

**Revisit if**: measurement shows the offspring distribution is
biased in a way that a symmetric call would fix. Then add
`crossover_pair(a, b, rng) -> tuple[ASTNode, ASTNode]` as an
additional method. Do NOT change the return type of `crossover`.

---

## D-054 — Depth and size limits enforced after crossover

**Decision**: after every candidate child is built, `subtree_crossover`
checks `tree_depth(child) <= max_depth` and `tree_size(child) <=
max_nodes`. Candidates that exceed either limit are discarded and the
next candidate is tried. If every candidate exceeds the limits,
`subtree_crossover` returns `None` and `Crossover.crossover` raises
`CrossoverError`.

**Why**: a parent that fits within the limits can still produce an
oversized child when a large subtree from B is transplanted into a
position that already held a large subtree. Constructing the child is
always legal at `ASTNode.__post_init__` — the limits are GP-level
policy, not AST-level validity. They must be enforced here.

The two limits are checked against the child as constructed, not
against a running sum. That is the correct semantics: the limits
describe the *shape* of the result, not the cost of producing it.

**Revisit if**: we introduce per-node cost functions (e.g. weights on
expensive operators). Then add a third limit `max_cost` in the same
place. Do NOT overload `max_nodes` with cost semantics.

---

## D-055 — The root node is never a crossover target

**Decision**: `subtree_crossover` filters out `path == ()` from both
parents before collecting candidate pairs. The child always inherits
A's root operator and parameters.

**Why**: replacing A's root with B's root is not crossover, it is
"copy B, discard A". It provides no recombination of genetic material
and, at population scale, effectively reseeds the population from
whichever parent happened to be chosen as the donor. Root replacement
is a *clone* operation, and there are cleaner ways to clone (literally
`Individual.from_ast(b)`).

Filtering out the root also makes the "no possible crossover"
condition well-defined: two leaf-only parents have empty non-root sets,
so `subtree_crossover` returns `None` immediately, and the dispatcher
raises `CrossoverError` rather than silently returning one of the
parents.

**Revisit if**: a future operator makes "root replacement" meaningful,
e.g. a schema-preserving swap at a designated "root-like" position.
That would be a distinct operation; add it by name rather than
relaxing this rule.

---

## D-056 — Crossover-limit tests must use limits below the parents' own size

**Decision**: any test asserting that `Crossover.crossover` raises
`CrossoverError` due to `max_depth` or `max_nodes` must set those
limits strictly below the *parents' own* depth/size, not just below
some hypothetical large child.

**Why**: `subtree_crossover` replaces one non-root subtree of A with
one non-root subtree of B. The minimum possible child is a leaf-for-
leaf replacement, which preserves A's depth and size exactly. So
every child has `depth(child) >= depth(A)` and `size(child) >= size(A)`
for the size that matters here. If `max_depth >= depth(A)` and
`max_nodes >= size(A)`, there is always at least one candidate — the
"swap a leaf for a leaf" case — that passes. The limit check only
starts rejecting candidates when the limit is below A's own footprint.

An earlier version of these tests assumed "a deep subtree in B forces
the child over the limit", which is false: B must contain leaves too,
and a leaf swap keeps the child within A's envelope.

This is not a bug in `subtree_crossover`. The limit check is
correctly rejecting oversized children and correctly allowing
leaf-for-leaf swaps. The tests were pinning the wrong invariant.

**Revisit if**: crossover gains an operation that can *shrink* the
tree below A's size (e.g. a "prune" crossover). Then this reasoning
breaks and the tests need a different setup.

---

## D-057 — Crossover tests must use parents that differ at two positions

**Decision**: any test that expects `Crossover.crossover` to succeed
must use parents whose differing nodes number at least 2, so that a
same-type swap can produce a child distinct from both parents.

**Why**: with parents `A = group_rank(close, sector)` and
`B = group_rank(open, sector)`, the only same-type swap candidate is
"A's `close` for B's `open`", which produces exactly B. The reverse
candidate (A's `sector` for B's `sector`) produces exactly A. Both
are correctly rejected by the dispatcher's fingerprint check, and the
call raises `CrossoverError`. That is not a failure of the crossover
logic — it is the logic working as designed on a pair with too little
genetic diversity.

A minimal productive pair differs at two positions, e.g.
`A = group_rank(close, sector)` and
`B = group_rank(open, industry)`. Then two distinct children are
reachable: `group_rank(open, sector)` and `group_rank(close, industry)`.

**Revisit if**: a future crossover variant is allowed to swap the
root (see D-055) or to substitute an operator in addition to a
subtree. Then a two-position-different pair may no longer be
minimal, and this rule becomes a special case of a more general
"parents must have at least one productive swap" precondition.

---

## D-058 — Handoff edits are full-file or line-precise, never partial

**Decision**: when a fix or addition touches existing code, the handoff
message delivers EITHER (a) the complete file, verbatim, with no
abridgement, OR (b) a line-anchored instruction naming the file, the
line range, and the exact old/new text. Mixed forms ("insert this
snippet somewhere around X") are forbidden.

**Why**: partial snippets force the reader to reconstruct context,
which is exactly the failure mode that makes long refactors error-
prone. Two concrete costs appeared in Phase 3 Step 3: (1) a snippet
that referenced a helper defined in another file but not shown,
(2) three test fixes described in prose that required the reader to
re-derive the surrounding fixtures. Both would have been avoided by
shipping the file.

The rule is asymmetric on purpose: new files are always full files
(they have no "before" state), and edits to existing files are full
files unless the change is small enough that line numbers are
unambiguous.

**Revisit if**: full files exceed practical message sizes for some
module. Then switch to line-anchored patches for that file
specifically, and state the line numbers explicitly. Never fall back
to "find the spot and add this".