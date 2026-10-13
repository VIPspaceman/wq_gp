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

**Scope note**: this normalization only fires on parameters where both
`int` and `float` inputs are legal — i.e. `FLOAT_RANGE`. On `INT_RANGE`,
`ParamDomain.contains` rejects `5.0` outright at construction (see
`test_int_range_rejects_bool_and_float`), so the fingerprint layer never
sees a float there. That is intentional: strict at the boundary, lenient
at the fingerprint. Two different layers, two different jobs.

**Why**: BRAIN params are usually specified as integers, but the JSON
pipeline can smuggle `5.0` in. Without normalization, the two forms become
distinct individuals in the population, doubling the cost of exploring what
is semantically one point.

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

## D-019 — RNG tests must snapshot before advancing

**Decision**: any test that verifies checkpoint RNG reproducibility must
capture the state to be stored via a *separate* `Random` instance (or by
calling `getstate()` explicitly before any further draws). It must NOT
read "expected next" draws from the same live RNG that is about to be
checkpointed.

**Why**: `random.Random.random()` mutates the generator. A test that does
`expected = [rng.random() for _ in range(5)]` and *then* checkpoints
`rng` is capturing the post-advance state, not the state that produced
`expected`. This produced a spurious failure in Phase 2 Step 1 with the
production code entirely correct. The fix is in the test, and the pattern
is easy to repeat, so it is recorded here.

**Revisit if**: we move to a stateless / splittable RNG (e.g. numpy
PCG64 with `SeedSequence.spawn`). Splittable RNGs make this class of
mistake harder, but not impossible; keep the rule.

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
2). The replacement helpers `replace_subtree` and `replace_param`
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

---

## D-059 — Selection is single-objective (fitness only)

**Decision**: `tournament_select` and `select_elites` rank individuals
by `Individual.fitness` only. Multi-objective selection (Pareto,
non-dominated sorting, crowding distance) is deferred.

**Why**: single-objective is what the first working end-to-end run
needs. Introducing Pareto sorting before the loop closes means every
Phase 4 refactor has to maintain two ranking code paths, and the extra
path is dead weight until there is a measured reason to prefer it.

When multi-objective does arrive, it belongs in a separate module
(`pareto.py`) with its own config and its own tests. The selection
API here is written so the caller can swap in a different `_pick_best`
without touching `next_generation`'s structure.

**Revisit if**: single-objective runs plateau with high-variance
offspring — that is the signal that one scalar fitness is collapsing
useful diversity. At that point add `pareto.py` and a
`SelectionConfig.ranking` discriminator.

---

## D-060 — Tournament sampling is without replacement

**Decision**: `tournament_select(pop, k, rng)` uses `rng.sample`, which
draws `min(k, len(pop))` distinct members. No individual appears twice
in the same tournament.

**Why**: with-replacement sampling changes the effective selection
pressure non-linearly. In a population of 10 with k=3, sampling with
replacement gives the best individual a 27% chance of winning;
without replacement it is 30%. The difference is small per draw, but
it compounds over thousands of generations and makes "how strong is
selection" impossible to reason about from the config alone.

`rng.sample` also makes the "k greater than population" case trivial:
`min(k, len(pop))` uses the whole population, which is the natural
fallback.

**Revisit if**: we want explicit selection-pressure control. Then
switch to with-replacement and add a `replacement: bool` field to
`SelectionConfig`, with its own tests. Do NOT silently change the
sampling mode.

---

## D-061 — Elites are carried over unchanged

**Decision**: the top `config.elite_count` individuals are copied
verbatim into the next generation. They are not mutated, not
re-evaluated, and keep their original fingerprint, fitness, and
generation number.

**Why**: elites exist to guarantee monotonic improvement of the best
fitness across generations. Mutating them defeats that purpose — if
the current best ever gets mutated away, the next generation's best
can be worse than the current one, and the run's trajectory becomes
non-monotonic in a way that is hard to diagnose.

Carrying them unchanged also makes the `Individual.generation` field
meaningful: an individual's generation is the generation in which it
was *created*. Elite members visibly keep their old generation number
across the boundary, which is the correct semantics.

**Revisit if**: we want "elite with mutation" as a separate operator
(e.g. to explore the neighborhood of the current best). That is a
distinct strategy; add it as an explicit offspring-producing step
rather than mutating the carried-over elites.

---

## D-062 — next_generation is best-effort on size, hard-fail only if empty

**Decision**: `next_generation` aims for `config.population_size`
individuals. If the breeding-attempt budget runs out before reaching
that size, it returns whatever it produced. It raises `SelectionError`
only if the result would be completely empty — i.e. no elites and no
offspring.

**Why**: in a large search space, a single generation might
legitimately fail to produce a full slate of offspring (an unlucky
run of `CrossoverError` and `MutationError` on a pool with limited
diversity). Making that a hard error would kill long-running jobs
that are otherwise progressing fine. Returning a smaller population
keeps the run alive; the coordinator can top it up later if desired.

Making "empty result" the only hard failure keeps the pipeline from
spinning on a truly dead pool: if we cannot even carry over an elite
(because none are evaluated) and cannot breed a single offspring, the
run is stuck and the caller must know.

**Revisit if**: Phase 4 introduces a "target population size" contract
that downstream stages rely on. Then add a `strict_size: bool` flag
to `SelectionConfig` and raise on shortfall when it is set. Do NOT
change the default behavior — best-effort is the right default for
long-running GP.

---

## D-063 — Crossover requires two parents with distinct fingerprints

**Decision**: `_breed_one` selects `parent_a` via tournament, then
attempts up to 5 times to draw a `parent_b` with a different
fingerprint. If it cannot, it falls back to mutation of `parent_a`.

**Why**: `Crossover.crossover` already rejects identical-fingerprint
parents with `CrossoverError`. But letting the call fail and retrying
the whole breeding step wastes a breeding attempt on a predictable
failure. Catching it here lets the same `parent_a` still be useful
via mutation, which is cheap and preserves the "one offspring per
breeding attempt" throughput guarantee.

The 5-attempt cap is deliberate: in a population where all members
share a fingerprint (should never happen — `Population` enforces
uniqueness — but defensively), we avoid an infinite loop.

**Revisit if**: a future selection mode allows parents with identical
fingerprints (e.g. clonal selection on an explicitly duplicated
population). Then this fallback becomes unnecessary, and the parent-
pair selection can be simplified. Do not remove the guard until that
feature exists.

---

## D-064 — No dead code in delivered test functions

**Decision**: a delivered test function may not contain a
"previous version" of the setup it means to run. If a test author
constructs X, decides X is wrong, and writes Y as the replacement,
X must be deleted from the file before delivery. Comments like
"# rebuild cleanly" or "# actually use the other one" are a signal
that the deletion did not happen and must not appear in shipped code.

**Why**: this pattern caused a real failure in Phase 3 Step 4. The
test `test_tournament_prefers_evaluated_over_unevaluated` opened with
a `Population(...)` call that duplicated a fingerprint, then added a
comment saying "rebuild cleanly", then wrote a second `Population(...)`
call. Python executes top-to-bottom: the first call raised
`DuplicateIndividualError` before the second was reached. The test
looked correct to its author because the second half *was* correct —
the author was reading the file as if the first half were gone.

A delivered file is an artifact, not a scratchpad. Anything visible
in it must be intended.

**Revisit if**: never. This is a code-hygiene floor. If a genuine
"two alternative setups, one commented out" pattern is needed for
documentation purposes, it belongs in a comment block, not as live
code.

---

---

## D-065 — GP-layer serialization lives in gp_serialization.py

**Decision**: `ast_to_dict` / `ast_from_dict` and the Individual /
Population variants live in `src/wqgp/gp_serialization.py`, not in the
existing `serialization.py`. Lookups are passed as callables, not a
`Pool` object.

**Why**: two problems solved at once.

(1) Circular import. `pool.py` imports from `serialization.py`. If
`serialization.py` tried to import `Pool` (via `population.py`) for
lookups, we would have `pool -> serialization -> population -> pool`.
Passing `field_lookup` and `operator_lookup` as plain callables breaks
the cycle without any lazy-import tricks.

(2) Layer separation. `serialization.py` handles Field / Operator /
ParamDomain — the vocabulary of the search space. GP serialization
handles ASTs, Individuals, and Populations — the vocabulary of the
search itself. Keeping these in separate modules means a change to
the AST format cannot ripple into the Field loader.

The dispatch is on `kind` for AST nodes ("LITERAL" / "FIELD_REF" /
"CALL") and on the shape of the payload for the others. No polymorphism,
no registry, no decorators.

**Revisit if**: we add a third serializable GP-layer type (e.g. a
"lineage" object for later analysis). Add it here. Do NOT move any of
this into `serialization.py`.

---

## D-066 — Expression rendering lives in ast.py

**Decision**: `render(node) -> str` and its scalar helper live in
`ast.py`. `coordinator.py` calls `render_ast(ind.ast)` to produce the
string it passes to a `Task`.

**Why**: rendering is a pure function of the AST, with no GP semantics.
Putting it in `ast.py` keeps it next to the type it operates on, and
next to `canonical()` — its closest sibling. The two share a design
constraint: both must be deterministic, and both must handle the same
set of node kinds.

Parameter ordering: data arguments first, then parameters, matching
BRAIN's call syntax (`ts_mean(close, 5)`). Scalar rendering collapses
`5.0` to `"5"` and leaves strings verbatim, because BRAIN does not
quote grouping identifiers like `sector`.

**Revisit if**: the platform introduces a positional-parameter syntax
that interleaves data and parameters, e.g. `foo(a, 5, b)`. Then
`render` needs to walk a joint positional list, and `OperatorDef`
needs a positional signature (see D-007's revisit clause). Both
changes ship together.

---

## D-067 — Checkpoints save the UNEVALUATED next-generation population

**Decision**: a checkpoint's payload contains the population at the
moment it is about to be evaluated. The coordinator saves the initial
population immediately, and after every `next_generation` call. It
does not save the evaluated population.

**Why**: this makes resume trivially consistent. A resumed run reads
an unevaluated population and enters the same loop iteration a
from-scratch run would have entered, in the same RNG state and with
the same evaluation store contents. Any other choice — saving after
evaluation, saving the elites only, saving mid-generation — produces
at least one path where resume diverges from uninterrupted execution
by one RNG draw or one skipped submission.

The consequence is that a crash mid-evaluation loses that generation's
submissions, which is correct: those submissions are recorded in the
evaluation store individually, so nothing is double-counted after
resume. The population payload is the only thing that has to be
re-derived, and it is exactly the pre-evaluation state.

**Revisit if**: we add incremental checkpointing within a generation
(e.g. every N evaluations). Then the payload must record which
individuals have already been submitted, and the resume logic gains a
"resume mid-generation" branch. That branch should be a separate
decision, not an amendment to this one.

---

## D-068 — Absorb existing evaluations before submitting

**Decision**: `Coordinator._evaluate` calls `_merge_evaluations` both
before and after submitting. The first call picks up elites and any
prior evaluations from the store, so those individuals are not
re-submitted. The second call picks up the just-completed submissions.

**Why**: two sources of redundant submission otherwise.

(1) Elites. An elite is copied verbatim into the next generation
(D-061). Its fingerprint is already in the evaluation store. Without
the pre-merge, `_evaluate` would submit it again, wasting a simulation
budget unit and duplicating an evaluation row that the upsert would
overwrite with identical data.

(2) Resume. After resuming, the population payload includes
individuals that the prior run already evaluated (the elites that
survived into the next generation). Same problem.

The pre-merge is cheap: one store read per unevaluated individual, and
the store is indexed by `(run_id, fingerprint)` so each read is
O(log n).

**Revisit if**: the store read becomes a measured bottleneck on large
populations. Then a single bulk read keyed by the population's
fingerprint set replaces the per-individual reads. Do NOT skip the
merge; correctness depends on it.

---

## D-069 — One Executor for the whole run

**Decision**: the coordinator creates one `Executor` in `run()`,
starts it once, and shuts it down in a `finally`. Between generations
it calls `tracker.start_generation()` to reset the per-generation
budget counter, but the executor's worker threads persist.

**Why**: spinning up N threads per generation and tearing them down
immediately costs more than the work they do on small populations.
More importantly, a persistent executor keeps `ExecutorStats`
cumulative across the run, which is what you want when reporting
"total tasks submitted" at the end.

The per-generation reset is a `BudgetTracker` concern, not an
`Executor` concern. The two are decoupled: the tracker counts
budget, the executor counts tasks. They happen to be created in the
same scope because they have the same lifetime, but neither knows
about the other's internals.

**Revisit if**: we add per-generation worker scaling (more workers in
early generations, fewer later). Then the executor gains a
`resize()` method, called at the same place the tracker is reset.

---

## D-070 — Loop terminates on budget, no-progress, or generation cap

**Decision**: the coordinator's main loop ends on the first of these
conditions:
- `tracker.is_exhausted()` → `stopped_reason = "budget_exhausted"`
- `len(pop.evaluated()) == 0` → `stopped_reason = "no_progress"`
- `generation + 1 >= max_generations` → `stopped_reason =
  "max_generations"`

**Why**: three distinct end-of-run states that need to be
distinguishable in the result.

(1) Budget exhaustion is the expected healthy exit for a long run.
The budget check happens right after evaluation, before the next
generation is bred, so an exhausted budget does not waste a
`next_generation` call.

(2) Zero evaluated individuals means the run cannot make progress:
selection has no fitness to rank on, and the next generation would be
arbitrary. Stopping explicitly beats letting `SelectionError` bubble
up from deep in `next_generation`.

(3) Reaching `max_generations` is the normal case for a run with a
loose budget. The check is placed after the budget check and before
`next_generation` so we do not breed a generation we will never
evaluate.

The result dataclass carries `stopped_reason` so callers can log or
branch on it without inspecting counters.

**Revisit if**: we add a "converged" stopping criterion (e.g. no
improvement in N generations). Then add a fourth reason, and consider
making the check the responsibility of a dedicated "stop policy"
object rather than inline code. Do NOT add a fourth `if` to this
function without that refactor.

---

## D-071 — BrainClient depends on HttpTransport, not on requests

**Decision**: `HttpTransport` is a `typing.Protocol` with one
`request()` method plus `close()`. `BrainClient` takes a transport in
its constructor. The production implementation is `RequestsTransport`;
tests inject a `ScriptedTransport` that returns canned responses.

**Why**: an HTTP client that imports `requests` directly is untestable
without monkeypatching. Monkeypatching `requests.Session.request` is
brittle — it couples tests to the version of requests installed and
makes parallel test runs racy. A Protocol + injection keeps every test
pure Python, no I/O, no patching.

The same pattern is used elsewhere in this project (D-033 for
`Evaluator`). Consistency matters: a reader who knows how the evaluator
is tested already knows how the transport is tested.

**Revisit if**: we adopt an async runtime. Then the Protocol gains an
`async def request(...)` and everything above it changes shape.
That is a rewrite, not an extension; do not start it until there is a
measured need.

---

## D-072 — Session cookies are owned by the transport

**Decision**: `RequestsTransport` wraps a `requests.Session`. That
session persists cookies (including BRAIN's JWT) across calls. The
`BrainClient` never reads or writes a cookie directly.

**Why**: cookie handling is a transport concern — it depends on the
specific HTTP library. The client only needs "authenticated or not",
which it tracks as a boolean. Any change in how BRAIN issues or
refreshes tokens is confined to the transport.

This also means `ScriptedTransport` does not need to simulate cookies.
Tests that need to verify "login happens once" count the number of
POSTs to `/authentication`, which is clearer than inspecting cookies.

**Revisit if**: BRAIN moves from cookie-based sessions to bearer tokens
in a header. Then the transport gains token injection and the client
gains token refresh logic. Both changes stay inside the respective
classes.

---

## D-073 — BrainCredentials has from_env()

**Decision**: `BrainCredentials` is a frozen dataclass with
`username` and `password`. `from_env()` reads `WQ_BRAIN_USERNAME` and
`WQ_BRAIN_PASSWORD`. `build_default_client()` uses `from_env()`.

**Why**: environment variables are the standard for secrets in a CLI /
daemon context. Putting the env-var names in one classmethod, rather
than scattering `os.environ.get(...)` calls, makes the contract
explicit and testable via `monkeypatch.setenv`.

The environment-variable prefix `WQ_BRAIN_` matches the reference
project and is unlikely to collide with anything else on a developer's
machine.

**Revisit if**: we need multiple accounts (e.g. one for exploration,
one for production). Then change `BrainClient.__init__` to take a
`credentials` and let the caller decide where they came from. Do NOT
add more env-var names to `from_env`.

---

## D-074 — BrainSettings validation is type-only

**Decision**: `BrainSettings` validates that fields have the right
Python type and that numeric ranges are sane (`delay >= 0`,
`0 <= truncation <= 1`). It does NOT enumerate the allowed values for
`region`, `universe`, `neutralization`, and the like.

**Why**: BRAIN's allowed values for those fields change over time —
regions get added, universes get retired. If we hardcoded the list,
the client would break on every platform update. Type-only validation
catches the mistakes that are ours to catch (a typo that produces an
int where a string is expected) and leaves the platform to reject
values it does not accept. The client already surfaces those
rejections as `TaskResult(FAILED)`.

**Revisit if**: BRAIN publishes a stable, versioned list of allowed
values (e.g. a `/settings` endpoint). At that point we can fetch the
list at startup and validate against it. Do NOT hardcode a copy.

---

## D-075 — Fitness is a pluggable FitnessFunction

**Decision**: `BrainClient` takes a `fitness_fn: MetricsView -> float |
None`. The default returns `metrics["sharpe"]` or `None`. Callers can
supply a composite (e.g. weighted sum of sharpe and returns).

**Why**: "what makes a good alpha" is a research decision, not a
client decision. Baking `sharpe` into the client would force every
researcher who prefers a different metric to fork the client. A
one-argument callable is the smallest interface that supports the
range of choices we expect.

Returning `None` from the callable is a legitimate signal: "these
metrics do not meet the minimum bar for fitness". The client converts
that into a `FAILED` TaskResult with a clear message, rather than
inventing a default. Silent substitution of a default would let
degenerate alphas pass into the population.

**Revisit if**: we want to store multiple fitnesses per evaluation
(e.g. one per objective). Then `FitnessFunction` returns a tuple, and
`EvaluationRecord` gains a second fitness column. Both changes ship
together.

---

## D-076 — Poll interval from Retry-After, backoff on transient errors

**Decision**: when a poll returns 201 or 202, the next sleep duration
comes from the `Retry-After` header (parsed as seconds). If the header
is missing or malformed, fall back to `base_poll_interval`. On a
transport error or 5xx response, sleep for
`base_poll_interval * 2^(attempt-1)`, capped at 30 seconds.

**Why**: BRAIN's poll responses carry a `Retry-After` that reflects
its own internal queue depth. Honoring it is both polite and faster
than a fixed interval — the server tells us when to come back.
Falling back to `base_poll_interval` (default 1.0s) keeps us from
busy-looping on a server that forgot the header.

Exponential backoff on errors is standard for transient failures. The
30-second cap is there to prevent a pathological run of errors from
sleeping for minutes, which would look like a hang.

**Revisit if**: BRAIN changes the header name or starts returning
milliseconds. Then this function needs one more branch. Do NOT add a
second backoff policy — extend this one.

---

## D-077 — Each task is bounded by max_polls AND poll_timeout_seconds

**Decision**: `_poll` gives up after `max_polls` iterations OR when
`time.monotonic()` exceeds a deadline set at the start of the poll.
Both limits are configurable; tests set `max_polls` small and
`poll_timeout_seconds` large.

**Why**: two independent failure modes need two bounds. A fast-but-
infinite loop (server always returns 202 immediately) burns CPU and
API quota; `max_polls` catches that. A slow-but-eventually-doomed
poll (server returns 202 with Retry-After 60 for hours) burns
wall-clock time; the deadline catches that. Neither bound alone is
sufficient.

`sleep_fn` is injectable so tests can pass `lambda _s: None` and
control timing without real waits. This is the standard pattern for
testing code that sleeps; the alternative — patching `time.sleep` —
is fragile under parallel test runners.

**Revisit if**: we add per-simulation scheduling (e.g. batch submit,
then poll all). Then the deadline should be computed once for the
batch, not per simulation. That is a bigger change than it looks;
defer until concrete need.

---

## D-078 — Error taxonomy: task failures return, auth failures raise

**Decision**: two categories of error, two channels.

- **Server-side rejection of one simulation** (bad expression, invalid
  settings, simulation status = ERROR, missing `is` block) →
  `TaskResult(status=FAILED, error=...)`. The pipeline keeps running.
- **Authentication failure** (login fails, a poll returns 401) →
  `BrainAuthError` raised. The Executor's D-034 handler catches it,
  records this task as FAILED, and moves on.
- **Client / protocol failures** (5xx exhausted retries, missing
  Location header, malformed JSON) → return `TaskResult(FAILED)` too,
  with the error message.

**Why**: the pipeline's job is to keep evaluating. Almost no error
should kill a worker thread. The one exception — auth — is still
caught by D-034, but raising it as a distinct type makes the intent
visible in logs: "this is not a bad expression, this is a session
problem, check the credentials".

Distinguishing by return value vs. raise also encodes intent at the
call site. A reader of `BrainClient.__call__` can see at a glance that
server rejections are values, not exceptions.

**Revisit if**: we add retry budgets for auth (e.g. re-login up to 3
times on 401 before giving up). Then `BrainAuthError` becomes
transient in some cases, and the client needs to decide based on its
own state. That is a client change, not a taxonomy change.

---

## D-079 — Test transports match on URL path, not full URL

**Decision**: `ScriptedTransport` (the test double for `HttpTransport`)
extracts the path via `urllib.parse.urlparse(url).path` and matches
registered prefixes against that path. It does NOT call
`url.startswith(prefix)` on the full URL.

**Why**: the test registers prefixes like `/authentication` and
`/simulations/123`, because those are the meaningful parts of a request.
The production client sends absolute URLs like
`https://api.worldquantbrain.com/authentication`, because that is what
`requests` expects. `full_url.startswith("/authentication")` is False
for every request, so every queue lookup missed and all 16 BrainClient
tests failed with "no scripted response".

Path-based matching also avoids a subtler bug that string-prefix
matching would introduce: `/simulations` is a prefix of
`/simulations/123`, so a full-string match on a bare `/simulations`
prefix would incorrectly match a specific-simulation request. Path +
`startswith` has the same textual subtlety, but the caller controls
specificity by registering the more specific path first when both
queues are live. In practice the method discriminator (`GET` vs `POST`)
already separates the two.

**Revisit if**: the client starts sending query strings that
distinguish endpoints (e.g. `/simulations?type=X` vs `?type=Y`). Then
the test transport should match on `parsed.path` AND `parsed.query`,
and the registered keys should become `(method, path, query)` tuples.
Do NOT extend the current prefix scheme to cover query strings.

---

---

## D-080 — BrainAuthError re-raises before the generic handler

**Decision**: `BrainClient.__call__` catches `BrainAuthError` explicitly
and re-raises it, BEFORE the `except BrainClientError` clause that
converts other errors into `TaskResult(FAILED)`.

**Why**: `BrainAuthError` is a subclass of `BrainClientError` (D-078
keeps them in one family), so Python's `except BrainClientError` clause
catches auth errors too if it appears first. The original implementation
put only the generic clause around `_submit_and_poll`, so a 401 during
submit or poll was silently converted into a `TaskResult(FAILED)` and
never reached the caller.

The two-channel design in D-078 is that auth failures go through the
*raise* channel so they are visibly different from per-expression
failures in logs and from any outer-level credential check. The clause
order is what implements that design. Order matters; a future edit
that reorders or merges these clauses would silently break D-078 again.

The alternative — making `BrainAuthError` inherit directly from
`RuntimeError` instead of `BrainClientError` — was considered and
rejected. It would break the semantic "auth error is a client error"
relation, and it would not protect against a future `except` clause
that catches `RuntimeError`. Explicit ordering is the standard Python
idiom for distinguishing a subtype in a try/except chain, and it keeps
the exception hierarchy intuitive.

**Revisit if**: a second "must-raise" subtype appears (e.g.
`BrainRateLimitError` that should also escape the FAILED conversion).
At that point, extract a tuple `_RAISING_ERRORS = (BrainAuthError,
BrainRateLimitError)` and write `except _RAISING_ERRORS: raise`. Do
NOT start a chain of individual clauses.

---

---

## D-081 — PoolBuilder depends on HttpTransport

**Decision**: `PoolBuilder` takes an `HttpTransport` in its
constructor. Production passes `RequestsTransport`; tests pass
`ScriptedTransport`. Same pattern as `BrainClient` (D-071).

**Why**: consistency. A reader who knows how BrainClient is tested
already knows how PoolBuilder is tested. And the same fake works for
both, so `tests/_fakes.py` is the single source of truth for the
transport double.

**Revisit if**: BRAIN introduces an endpoint that needs different HTTP
semantics (e.g. streaming). Then that endpoint gets its own transport
abstraction rather than widening this one.

---

## D-082 — Unknown DataType strings are dropped, not guessed

**Decision**: `PoolBuilder._parse_field` maps BRAIN's `type` string
through a fixed table. A string not in the table causes the field to
be dropped and recorded in `dropped_field_ids`.

**Why**: BRAIN periodically adds new types (or renames existing ones).
If we guessed — e.g. mapped anything unknown to MATRIX — a field that
should have been GROUP would end up MATRIX, and every operator that
expects a grouping key would fail at AST construction with a confusing
error far from the root cause. Silently dropping is louder: the report
tells you which fields vanished, and you can extend the table.

Dropping also gives a clean upgrade path: add the new type to the map,
re-run the builder, done.

**Revisit if**: the dropped count becomes routinely large (>5% of
fields). Then BRAIN has likely added a family of types (e.g. all
"TIME_*"), and the fix is to extend the table, not to loosen the rule.

---

## D-083 — SemanticKind inference is GROUP → CATEGORICAL, else NUMERIC

**Decision**: `_semantic_kind_for(data_type)` returns CATEGORICAL for
GROUP and NUMERIC for everything else. It never returns TIME.

**Why**: BRAIN's `type` field is the only reliable classifier we get.
A GROUP field is by definition a grouping key, and grouping keys are
categorical. Everything else is used numerically in practice. There is
no reliable signal for TIME in the field payload, and a wrong TIME
label would be worse than no label — the fingerprint's scalar
normalization (D-012) and any future time-aware operator would
misbehave.

Doing nothing is the right default for the other cases.

**Revisit if**: BRAIN exposes a `semantic` or `role` field, or if we
find a class of fields that are numerically typed but time-like
(e.g. a "last update timestamp" field). Then extend this function
with an explicit rule per case and a test for each.

---

## D-084 — Operator signatures come from an external table

**Decision**: `PoolBuilder.build(signatures)` takes an
`OperatorSignatureTable`. Operators named by BRAIN but missing from the
table are dropped and reported in `dropped_operator_names`.

**Why**: BRAIN's `/operators` endpoint returns names and descriptions
only. It does not return `input_types`, `output_type`, `param_domains`,
or the commutativity flags. Those are exactly what the AST layer needs
(D-007, D-008, D-009). Constructing a bare `OperatorDef` with a guessed
signature would poison the type checker: every "add(a, b)" would type
check against the wrong arity, and the failures would surface three
layers down.

Requiring an external table makes the missing knowledge explicit and
localized. The table is version-controlled, reviewable, and can be
extended by one person without touching any code.

**Revisit if**: BRAIN adds a `/operator-signatures` endpoint, or
publishes a machine-readable schema. Then `PoolBuilder` gains a second
fetch and the table becomes optional. Do NOT make the table optional
before that — the current requirement is the thing that keeps the
type system honest.

---

## D-085 — Pagination is limit + offset, with a hard page cap

**Decision**: `_fetch_paginated` issues `?limit=N&offset=M` requests
in a loop. It stops when a page returns fewer than `limit` items.
`max_pages` caps the loop; exceeding it raises `PoolBuilderError`.

**Why**: BRAIN's list endpoints return either a bare list or an object
with a `results` / `data` / `items` field. A "short page" is the
universal end-of-data signal, and it works for both shapes without
needing a `total` field. The `max_pages` cap prevents an infinite loop
if the server ever returns full pages forever — a real failure mode
for a buggy endpoint, and one that would silently consume memory.

`_extract_items` accepts all three list-field names because we do not
control which one BRAIN returns, and refusing to parse a valid response
because of a key name is not a useful failure.

**Revisit if**: BRAIN returns a `next` URL or a cursor instead of an
offset. Then `_fetch_paginated` needs a second mode. Do NOT try to
support both transparently; make the caller choose, so the failure
mode is visible.

---

## D-086 — PoolBuildResult is a frozen value object

**Decision**: `build()` returns a frozen `PoolBuildResult` carrying the
`Pool` plus four counters and two tuples of dropped ids/names.

**Why**: the caller needs to know what was dropped. A builder that
returns just a `Pool` forces the caller to diff against the raw
response — impossible, because the caller does not have the raw
response. Making the drops a first-class part of the result keeps the
information visible at the call site.

Frozen because the result must not be mutated after construction; a
caller who wants a modified pool should build a new one.

The `summary()` method is a convenience for logging. It is not the
canonical API; callers who need the raw tuples use the fields.

**Revisit if**: we add a "strict mode" that raises on any drop. Then
`PoolBuildResult` gains a flag or a separate `StrictPoolBuildResult`.
Do NOT change the default behavior — dropping is the correct default
for exploratory runs, where the search space naturally drifts.

---

---

## D-087 — CLI uses argparse subcommands, not typer/click

**Decision**: the CLI is built with `argparse` and one subparser per
command. No third-party CLI library.

**Why**: zero runtime dependency. The project already depends on
`requests` for the BRAIN client; every extra dependency is a supply-
chain surface and a version conflict waiting to happen. `argparse`
covers everything we need: subcommands, typed flags, help text,
`--version`, usage-error exit codes.

The subcommand list is stable (`build-pool`, `run`, `list-runs`) and
small. If the list grows to fifteen commands with nested groups, a
library like `click` would earn its keep. It has not, and I do not
expect it to.

**Revisit if**: the help text starts repeating itself, or we need
shell completion. Both are `click` territory. Migrating later is a
mechanical refactor because the command bodies are plain functions
taking an `argparse.Namespace`.

---

## D-088 — Config is JSON; flag > config > default

**Decision**: `--config PATH` loads a JSON file. Every flag has a
matching config key; if the flag is set it wins, otherwise the config
value, otherwise the built-in default. Merge logic lives in `_pick`.

**Why**: two audiences. An interactive user types flags; a scheduled
job writes a config file once and reuses it. Supporting both with the
same precedence rule means the interactive user can "override one
thing" against a config file without rewriting it.

JSON over TOML because `tomllib` is Python 3.11+, and the project
targets 3.10. Adding `tomli` would be a second dependency for a
cosmetic format preference.

The precedence rule is stated in one helper (`_pick`) rather than
inlined in every resolver. One rule, one place to change.

**Revisit if**: config files become nested enough that JSON becomes
painful (comments, trailing commas). At that point, add `tomli` for
3.10 and keep the same precedence helper. Do NOT introduce a
"config inherits from config" chain; that is a different feature.

---

## D-089 — Secrets only from environment variables

**Decision**: `WQ_BRAIN_USERNAME` and `WQ_BRAIN_PASSWORD` are read by
`BrainCredentials.from_env()`. Neither the config file nor any CLI
flag accepts a credential.

**Why**: a credential in a config file ends up in git; a credential on
a command line ends up in shell history and in `ps` output. Neither is
acceptable. Environment variables are visible only to the process
owner and to root, and are the standard for secret injection in
containers and CI.

The decision is enforced at three layers: the CLI never defines a
credential flag, `RunConfig` has no credential field, and
`BrainCredentials` is constructed only inside `build_default_client()`
and `_cmd_build_pool`. There is no "just this once" path.

**Revisit if**: a hosted deployment needs credential injection from a
vault at runtime. Then a `--credentials-command` flag that *executes*
a command returning JSON would be the pattern — still not accepting
the secret on the CLI, but allowing retrieval. That is a distinct
feature with its own decision.

---

## D-090 — SIGINT propagates; exit code 130

**Decision**: `main()` catches `KeyboardInterrupt`, logs "interrupted
by user", and returns 130. It does NOT install a signal handler. Any
in-progress `Coordinator.run()` finishes its `finally` block before
the exception reaches `main`.

**Why**: 130 is the conventional shell exit code for SIGINT (128 + 2).
Scripts that key off this code behave correctly.

Installing a custom signal handler would be worse: it would either
have to reach into the Coordinator to set a "stop soon" flag (adding
coupling), or it would raise the exception anyway (same as now, with
extra code). Python's default is already correct.

Known limitation: worker threads do not receive signals, so a
`Coordinator.run()` interrupted mid-poll may block in `shutdown()`
until the in-flight poll completes. That is a coordinator-level
concern (see future work), not a CLI concern. The user sees the
`KeyboardInterrupt` logged and the process waits a bounded amount
of time for cleanup.

**Revisit if**: we want "interrupt stops after the current generation,
saving a checkpoint" as a first-class feature. Then `Coordinator`
gains a `stop_requested` callable and the CLI installs a handler
that flips a boolean. That change touches Coordinator, not just CLI,
and gets its own decision.

---

## D-091 — Logging to stderr, level via --log-level

**Decision**: `logging.basicConfig` writes to stderr with a
timestamped format. `--log-level` sets the level (default INFO).
The `list-runs` command prints its table to stdout, not through the
logger.

**Why**: logs and command output are different channels. Logs go to
stderr, machine-readable output goes to stdout, so a shell pipeline
can consume one without the other. `list-runs` is a query command;
its output is the point, not a side effect.

The format includes logger name and level so a multi-module trace is
readable without extra configuration. Timestamps are local; a
production deployment that needs UTC can set `TZ=UTC` in the
environment.

**Revisit if**: we add `--json-logs` for machine consumption. That is
a second formatter selected by a flag, not a change to the default.

---

## D-092 — --dry-run uses a deterministic hash-based evaluator

**Decision**: `--dry-run` replaces the real `BrainClient` with
`_dry_run_evaluator`, a function that computes
`SHA-256(fingerprint)[:8]` as a 64-bit integer and scales it to
`[0, 3.0)` as a fitness.

**Why**: end-to-end CLI tests need to exercise the full pipeline
(build-pool, config loading, coordinator, checkpoints) without
network and without credentials. A random evaluator would break the
reproducibility contract that D-016, D-041, and D-067 rely on;
hashing the fingerprint gives a stable fitness per expression across
processes and runs.

SHA-256 specifically (not Python's `hash()`) because `hash()` for
strings is randomized per-process unless `PYTHONHASHSEED` is set.
A test that passed one process and failed the next is exactly the
kind of flake we want to never introduce.

**Revisit if**: dry runs need to mimic BRAIN's failure modes (e.g. a
percentage of expressions rejected). Then `_dry_run_evaluator` gains
a failure probability driven by the same hash. Keep it deterministic;
do not introduce randomness.

---

---

## D-093 — Notifier is a Protocol; NullNotifier is the default

**Decision**: `Notifier` is a `typing.Protocol` with three methods
(`on_run_start`, `on_generation_end`, `on_run_end`). The default is
`NullNotifier`, which does nothing. The Coordinator's constructor
takes `notifier: Notifier | None = None` and stores a `NullNotifier`
if given `None`.

**Why**: two forces.

(1) Structural typing lets tests pass a plain recording class without
inheriting from an ABC. Same argument as `Evaluator` (D-033) and
`HttpTransport` (D-071). Consistency across the codebase matters.

(2) A default `NullNotifier` means the Coordinator's call sites are
uniform: `self._notifier.on_run_start(event)` with no `if self._notifier
is not None` anywhere. The branch-free code is easier to reason about
and the null case has exactly one implementation, tested once.

**Revisit if**: notifiers need configuration or lifecycle management
(e.g. an async queue with a background drain thread). Then the Protocol
gains a `start()`/`stop()` pair, and `NullNotifier` implements them as
no-ops. Do NOT add those methods until there is a real consumer.

---

## D-094 — TelegramNotifier sends over HttpTransport

**Decision**: `TelegramNotifier` takes an `HttpTransport` in its
constructor. Production passes `RequestsTransport`; tests pass
`ScriptedTransport`. The URL is constructed from the bot token; the
body is JSON.

**Why**: the same decision as `BrainClient` (D-071) and `PoolBuilder`
(D-081). One abstraction, one test double, no monkeypatching. A new
transport-dependent class is a non-event: the pattern is established
and the tests look like every other transport-dependent test.

The message body is plain text, not Markdown or HTML. Markdown escape
rules in Telegram are strict and silently drop messages when unescaped
metacharacters appear. Run ids and expression strings can contain
those characters. Plain text removes a class of "why didn't I get a
message" bugs.

**Revisit if**: we need structured content (buttons, tables). Then
the body becomes MarkdownV2 with a proper escaper, and the escaper
gets its own tests. Do not "just add Markdown" without the escaper.

---

## D-095 — Notifications swallow their own exceptions

**Decision**: every method on a Notifier swallows every exception it
encounters, logs at WARNING level, and returns normally. The Coordinator
adds a second layer of defense (`_safe_notify`), which catches anything
that escapes and logs it.

**Why**: a notification is an observation, not a dependency. If
Telegram is down, or a network flake causes a timeout, the GP run must
not be affected. The alternative — letting a notification error
propagate — turns a cosmetic problem into a data-loss problem (an
uncaught exception in the middle of a generation could lose a
checkpoint write).

The double layer is intentional: the Notifier's own `try/except` is
the primary defense, and the Coordinator's `_safe_notify` catches the
case where a notifier is missing the primary defense (e.g. a custom
notifier written by a user). The two layers cost almost nothing and
remove an entire class of "why did my run die when Telegram hiccupped"
bug reports.

This is the same principle as D-034 (the Executor swallows evaluator
exceptions into FAILED records), applied to a different subsystem.

**Revisit if**: we want to surface notification failures to the user
visibly (e.g. exit non-zero if the final "run complete" message could
not be sent). Then add a `strict: bool` flag to the notifier and a
counter of failed sends. Do NOT remove the swallow; opt in to strict
mode explicitly.

---

## D-096 — Events are frozen dataclasses, not strings

**Decision**: `RunStartEvent`, `GenerationEndEvent`, `RunEndEvent` are
frozen dataclasses with named fields. The Notifier receives the event
object and is responsible for formatting.

**Why**: string-building is a presentation concern; it belongs in the
notifier, not the coordinator. A coordinator that builds "run alpha
finished with sharpe 1.5..." has embedded Telegram-shaped text into
the GP loop; swapping notifiers means editing the coordinator. Passing
structured events keeps the coordinator ignorant of formatting.

Frozen because a notifier must not be able to mutate the coordinator's
view of the run. A notifier that could change an event's fields would
be a back-channel mutation API, which is exactly what we do not want.

The events deliberately exclude the population itself. A notifier
that receives the full population could start making decisions based
on it, or worse, mutate it. For debugging, the checkpoint and the
evaluation store are the sources of truth. Events are a signal, not a
snapshot.

**Revisit if**: we add an event that carries the current best
individual (e.g. for a "post the best alpha to a channel" feature).
Then the event carries the individual's fingerprint and expression
string, still not the Individual object. Keep the boundary: events
are data about the run, not references into it.

---

## D-097 — Telegram credentials from environment only

**Decision**: `TelegramCredentials.from_env()` reads
`TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`. `TelegramCredentials`
itself only takes them via its constructor; the CLI does not accept
them as flags or config values.

**Why**: same as D-089 for BRAIN credentials. A secret in a config
file or a shell command is a secret in git or a shell history. The
env-var path is the only one that a reasonable deployment can secure.

The env-var names match the convention used by most Telegram bot
libraries, so a user who already configured a bot for another tool
does not need to learn a second set of names.

**Revisit if**: multi-chat routing is needed (different runs to
different chats). Then `TELEGRAM_CHAT_ID` becomes a per-run setting,
which puts it in `RunConfig` — still read from env at the CLI layer,
never accepted as a raw flag. That is a distinct feature; do not
weaken this rule to support it.

---

## D-098 — Coordinator takes an optional notifier

**Decision**: `Coordinator.__init__(config, evaluator, notifier=None)`.
The CLI creates the notifier from `--telegram`; programmatic callers
pass one explicitly or omit it.

**Why**: the coordinator should not know what a Telegram is, and it
should not read environment variables. Making the notifier an
injected dependency keeps the coordinator testable without touching
`os.environ` and keeps the notification transport decisions in the
CLI (or in whatever program is embedding the coordinator).

The CLI is the right place for `--telegram` because it is the only
layer that already reads env vars for credentials (D-089) and already
owns the argument-parsing surface. A future GUI or scheduler that
embeds the Coordinator can pick its own notifier without a CLI flag.

**Revisit if**: the coordinator needs to emit events to *multiple*
notifiers (e.g. Telegram and a log file). Then the constructor takes
`notifiers: Sequence[Notifier]`, and a `FanoutNotifier` composes them.
Do NOT add a `notifier` field to `RunConfig`; config describes the
run, not the observation of it.

---

---

## D-099 — ScriptedTransport matches URLs by path suffix, not prefix

**Decision**: `ScriptedTransport.request` matches the registered key
against `urlparse(url).path` using `path.endswith(suffix)`. It does
NOT use `startswith`.

**Why**: two URL shapes are in play.

- BRAIN: `https://api.worldquantbrain.com/authentication` → path
  `/authentication`. Both `startswith` and `endswith` work.
- Telegram: `https://api.telegram.org/bot<TOKEN>/sendMessage` → path
  `/bot<TOKEN>/sendMessage`. Only `endswith` works; `startswith` fails
  because the identifying segment is at the end.

Switching to `endswith` is a superset for our cases: every BRAIN
registration still matches (the registered prefix is the whole path
there). It is also the right choice in general for REST-ish APIs:
the resource identifier is almost always at the end of the path.

The suffix rule does not introduce ambiguity when both a short and a
long suffix are registered. `"/simulations/1".endswith("/simulations")`
is `False`, so a specific registration cannot be shadowed by a shorter
one. The method discriminator (`GET` vs `POST`) further separates the
two endpoint families.

**Revisit if**: an API introduces a query-string discriminator, e.g.
`/simulations?type=REGULAR` vs `?type=BATCH`. Then registration keys
become `(method, path_suffix, query_key, query_value)` tuples. Do NOT
extend the current suffix scheme to cover query strings; the matching
logic would become unreadable.

---

## D-100 — Resuming requires a prior next_generation, not just a checkpoint

**Decision**: any test that asserts `RunStartEvent.resuming is True`
must first run a prior Coordinator with `max_generations >= 2`. A
`max_generations=1` prior run is not sufficient.

**Why**: D-067 says a checkpoint captures the population the moment it
is about to be evaluated, i.e. after a `next_generation` call. The
initial setup writes a generation-0 checkpoint; every subsequent
checkpoint is written after a `next_generation`. When
`max_generations=1`, the loop evaluates generation 0 and then exits
*before* calling `next_generation`, so no generation-1 checkpoint is
written.

The second run then loads the generation-0 checkpoint and reports
`resuming=False`, `generation=0`, which is semantically correct: no
progress was lost. The earlier test assumed "any prior run leaves a
resumable checkpoint", which is not what the coordinator promises.

This is not a bug in the coordinator. It is a property of the
"checkpoint = pre-evaluation population" rule, and the test's premise
was wrong about the boundary case.

**Revisit if**: we change D-067 so that every generation boundary
writes a checkpoint regardless of whether next_generation ran. Then
this decision reverses. Do not reverse it without also revisiting
D-067, because the two are the same invariant viewed from different
angles.

---

---

## D-101 — The signatures auditor is a function, not a class

**Decision**: `audit_signatures(credentials, transport, table)` is a
module-level function in `signatures_audit.py`. There is no
`SignaturesBuilder` class. The result is a frozen `SignatureAudit`
value object.

**Why**: the audit has no state to maintain. It reads BRAIN's operator
names, compares them to a table, returns a report, and exits. A class
would need an `__init__`, a `close`, and a lifecycle; all of that
would be overhead without a benefit. The function form makes the call
site read as one operation: "audit this table against BRAIN".

The result is a separate type because the CLI needs to format it and
tests need to assert on its fields. A raw dict would be less checkable
(no invariants) and less readable (string keys).

If a second operation appears — e.g. "fill missing entries from a
template file" — that becomes another function in the same module.
Only if multiple operations need to share login state does a class
become justified.

**Revisit if**: we add stateful operations (e.g. a cursor that walks
BRAIN's operator list page by page and pauses). Then a small
`SignaturesSession` class wrapping the auth flag is justified. Do NOT
create it preemptively.

---

## D-102 — The auditor never writes to the signature file

**Decision**: `audit_signatures` reads BRAIN, reads the table (which
the caller loaded), and returns a report. It does not open, modify, or
overwrite the signature file. The CLI prints the report and exits.

**Why**: an audit that silently modifies the file it audits is a
footgun. Two failure modes would appear immediately.

(1) A partially-filled table becomes a trap. If the auditor appends
placeholder entries for missing operators, those placeholders flow
into the next `build-pool` call, and `OperatorSignatureTable.from_dict`
either rejects them (breaking the pipeline) or parses them (silently
producing wrong signatures). Neither is acceptable.

(2) The user loses track of what they wrote. A signature table is a
hand-curated asset. Its diff should be under the user's control; a
tool that rewrites it turns "git diff" from a reviewable record into
a fight.

The output includes paste-ready skeletons (D-103) so the user can
add entries with minimal friction. The choice to add them stays with
the user.

**Revisit if**: we add a `--write` flag that the user opts into. Then
the write path is explicit and the diff is still reviewable. The
default must remain read-only.

---

## D-103 — Skeletons use MATRIX as an obvious placeholder, not a guess

**Decision**: `format_skeletons` emits
`{"name": "X", "input_types": ["MATRIX"], "output_type": "MATRIX"}`
for each missing operator. It does not attempt to infer the actual
types from BRAIN's description or from operator naming conventions.

**Why**: two forces.

(1) The standing rule (D-084) is that we do not guess signatures. A
guessed MATRIX-typed `ts_mean` would type check against `add(ts_mean
(...), 1)` and silently break the meaning of the expression. Once a
guess is written into the file, it looks like a decision, not a
placeholder.

(2) A skeleton with a wrong placeholder is still useful because the
placeholder is visibly wrong. If a user pastes the skeleton and
forgets to change `"MATRIX"`, `operator_from_dict` will accept it
(the string is a valid DataType), but the human review of the diff
will catch it — the type is wrong for most operators and that is
obvious at a glance.

The safer alternative — emitting a deliberately invalid placeholder
like `"TODO"` — was rejected because it makes the table unusable until
every entry is edited. A user who wants to run a 20-operator subset
should be able to paste 20 skeletons, edit the 3 they care about, and
delete the rest. The current placeholder supports that workflow; the
"TODO" placeholder does not.

**Revisit if**: the placeholder is empirically mistaken for a real
signature in practice. Then switch to a discriminated `{"kind":
"TODO"}` marker that `operator_from_dict` rejects explicitly. Do NOT
make the placeholder silent — a wrongly-typed operator that passes
the loader is worse than one that fails.

---

## D-104 — The auditor does not close the caller's transport

**Decision**: `audit_signatures` constructs a `PoolBuilder` to borrow
its authenticated pagination, but does NOT call `builder.close()`. The
caller owns the transport and is responsible for closing it.

**Why**: `PoolBuilder.close()` closes the underlying transport. When
the caller passes a shared transport (the CLI's `_make_transport`
result, or a test's `ScriptedTransport`), closing it inside the audit
would break any subsequent use in the same process. The audit is a
read-only, short-lived operation; leaving the transport open is the
right default.

The alternative — auditing without a `PoolBuilder`, by duplicating the
pagination logic — was rejected because the duplication is nontrivial
(authentication caching, page-loop termination, response-shape
tolerance) and this is the third consumer of that logic (after
`PoolBuilder` itself and the CLI). Duplicating a 40-line function
three times is worse than one caller paying attention to lifecycle.

The consequence is that a caller who audits repeatedly with the same
transport re-authenticates on each call. That is cheap (one POST) and
correct. A future optimization, if it becomes necessary, would be a
public `PoolBuilder.fetch_operator_names()` that reuses an existing
`PoolBuilder` instance passed by the caller.

**Revisit if**: the re-authentication cost becomes measurable (e.g.
auditing dozens of signature variants in a loop). Then the API
changes to accept an already-authenticated `PoolBuilder`, and the
caller manages its lifetime explicitly. The current signature stays
for the common one-shot case.

---

---

## D-105 — Diagnostics verify values, not just presence

**Decision**: any diagnostic that reports whether an environment
variable is correctly set must check its length or content, not only
`${VAR:+...}`. The `${VAR:+<set>}` form proves only non-emptiness.

**Why**: during Phase 4 Step 4 a user confirmed "password=[<set>]"
and only discovered the variable held a single character after running
`curl` and receiving a 401. The check reported presence; the user read
it as correctness; two rounds were spent debugging the client before
the real problem was visible.

Canonical check for a password-like variable:

    echo -n "$WQ_BRAIN_PASSWORD" | wc -c

Show the length, never the value. A non-zero length that is obviously
wrong (e.g. 1 for a real password) is the signal that the variable was
mangled during assignment, typically by double-quoted shell expansion
of `$`, `!`, or backticks.

**Revisit if**: never. This is a debugging-hygiene rule.

---

---

## D-106 — Repeated pages terminate pagination

**Decision**: `PoolBuilder._fetch_paginated` computes a JSON signature
of each page's items. If a page's signature equals the previous page's,
the loop stops and the accumulated items (from the first page) are
returned. No error is raised; this is a legitimate end-of-data signal
for endpoints that ignore `limit`/`offset`.

**Why**: BRAIN's `/operators` endpoint returns the full operator list
on every request, regardless of the `limit` and `offset` query
parameters. The `/data-fields` endpoint respects them. With the
previous implementation (end only on a short page), `/operators` would
return a full page forever, exhaust `max_pages`, and fail the whole
build with a confusing "exceeded 200 pages" error. Users saw this as
a bug in wqgp; it was actually the endpoint ignoring our pagination
request.

Detecting the repeat is the minimal fix. It preserves correct behavior
on `/data-fields` (distinct pages keep advancing) and handles
`/operators` without special-casing the endpoint name. It also covers
future endpoints that might change their pagination behavior.

The signature uses `json.dumps(items, sort_keys=True)`. Items are
dicts with stable key ordering after `sort_keys`; identical responses
serialize to identical strings. If an item contains a non-JSON-able
value, the signature is `None` and the repeat check is skipped —
degrading gracefully to the old behavior.

The `max_pages` cap remains. If a hypothetical endpoint returned
*full but distinct* pages forever, we would still stop at `max_pages`
with an error. That case is not observed today, but keeping the cap
costs nothing and prevents an unbounded loop.

**Revisit if**: we observe an endpoint that legitimately returns two
identical consecutive pages as part of a longer result. In that case
the signature check needs to look at page index rather than content,
or a new termination hint (e.g. a `Link: rel=next` header) needs to
be used. Do NOT weaken the check to "only the first two pages".

---

---

## D-107 — PoolBuilder takes settings; /data-fields needs them

**Decision**: `PoolBuilder.__init__` accepts `settings: BrainSettings |
None = None`. When None, it uses `BrainSettings()` (USA / TOP3000 /
delay 1). `fetch_fields()` appends `instrumentType`, `region`, `delay`,
`universe` to the query string of every `/data-fields` request.
`fetch_operator_names()` sends none of these.

**Why**: BRAIN's `/data-fields` endpoint returns HTTP 400 with
`["Invalid query"]` when called without simulation settings. The
endpoint is not "list all fields"; it is "list fields available in
this specific simulation configuration". The same field id can exist
in one region and be absent in another, and the `delay` parameter
selects a different variant of the dataset.

`/operators` does not have this requirement; the operator list is
global. Sending settings to it would be harmless (extra query
parameters are ignored) but we deliberately don't, so the request
URLs stay minimal and the behavior is testable.

The `settings` parameter is optional so `SignaturesAudit` (which only
calls `/operators`) does not need to construct one. The default is
`BrainSettings()` rather than a hard-coded dict because `BrainSettings`
is already the single source of truth for what the defaults mean, and
keeping them in one place avoids drift.

The CLI exposes four flags on `build-pool`: `--instrument-type`,
`--region`, `--universe`, `--delay`. Each defaults to None, and the
resolver fills in from `BrainSettings()`. This keeps the flag help
text and the dataclass defaults in sync automatically.

Query parameters are sorted alphabetically before being joined, so
the URL is deterministic. This makes the request reproducible across
runs and the tests straightforward to assert against.

**Revisit if**: BRAIN adds a second settings-bearing endpoint. Then
the settings serialization (`_settings_query`) moves into a shared
helper, or `BrainSettings` gains a `to_query_dict()` method next to
its existing `to_api_dict()`. Do NOT duplicate the four-key dict in
each endpoint's fetch method.

---

---

## D-108 — PoolBuilder retries 429 with header-driven backoff

**Decision**: when `_fetch_paginated` receives an HTTP 429, it sleeps
for the interval advertised in the response's `Retry-After` header,
falling back to `ratelimit-reset`, falling back to a configurable
`retry_after_default` (5.0s). It retries up to `max_retries_on_429`
times (default 3) before raising `PoolBuilderError`. The `sleep_fn`
is injectable for tests. The default `page_limit` is raised from 50
to 500, cutting the number of requests for a full `/data-fields`
fetch from ~88 to ~9.

**Why**: BRAIN caps requests at 50/minute (`x-ratelimit-limit-minute:
50` in every response). Fetching all ~4400 USA/TOP3000/delay=1 fields
at `page_limit=50` requires 88 requests, well over the quota. The
first `build-pool` attempt failed at offset 200 with 429.

Two changes together. Raising `page_limit` avoids the limit in the
common case; retrying handles the case where the limit is still hit
(e.g. a first fetch followed immediately by a second, or multiple
parallel CLIs). Neither change alone is sufficient — a large page
size can still blow through the quota if the user runs back-to-back
fetches, and the retry alone would make the default fetch take
minutes due to repeated sleeps.

BRAIN advertises two headers: `Retry-After` (HTTP standard) and
`ratelimit-reset` (BRAIN-specific seconds-until-reset). We check
Retry-After first, then ratelimit-reset, so a standards-compliant
proxy in front of BRAIN also works. The default is 5.0 seconds, long
enough to be useful and short enough to not feel like a hang.

The retry counter is per-request, not per-fetch. A page that succeeds
on the third attempt resets nothing; the next page starts with a
fresh counter. This matches the semantics of "each request is
independently rate-limited".

`sleep_fn` is injectable for the same reason as D-077 (in the
BrainClient): tests that assert on sleeps must not actually sleep.

**Revisit if**: BRAIN's rate limit changes to a sliding window that a
fixed 5-second default does not fit. Then the default becomes a
configurable derived value (e.g. `60 / limit + 1`). Do NOT remove the
retry — a hard 429 that kills a multi-minute fetch is worse than a
few seconds of sleeping.

---

---

## D-109 — BRAIN /data-fields rejects large page sizes with 400

**Decision**: `PoolBuilder`'s default `page_limit` is 50. D-108's
attempt to raise it to 500 was reverted after BRAIN returned HTTP 400
at offset 0 on the first request with `limit=500`.

**Why**: the reasoning behind D-108 was that fewer requests would
avoid hitting the rate limit. That is true in principle but does not
work in practice: BRAIN imposes a maximum page size below 500 and
rejects larger values with a bare `400 Bad Request` (no `detail`
message telling us the cap). Probing for the exact maximum would burn
request quota on a value that changes without notice.

50 is known to work — every successful fetch in this project has used
it, including the earlier partial run that reached offset 200 before
hitting the rate limit. Combined with D-108's retry-on-429 logic, the
default handles both failure modes:

  - Too many requests per minute → 429, retried after Retry-After.
  - Too-large page → not sent; we stay at a page size BRAIN accepts.

The retry logic from D-108 remains in place and is exercised by tests
independently of the default page size.

**Revisit if**: BRAIN publishes a maximum page size in its docs, or if
we discover empirically that a larger value (e.g. 100, 200) is also
accepted. Even then, the retry logic must stay: any page size can
still hit the per-minute quota during a long fetch.

---

---

## D-110 — Poll responses are two-step; metrics need a second GET

**Decision**: the poll loop interprets four response shapes, only one
of which yields metrics — and even that one requires a second request.

- `{"progress": float}` (no status) → keep polling.
- `{"status": "COMPLETE", "alpha": "<id>"}` → GET
  `/alphas/<id>`, extract the `is` block, done.
- `{"status": "ERROR", ...}` → terminal failure.
- `{"status": "WARNING", "message": ...}` → terminal failure (D-111).

The polling response NEVER contains the `is` metrics block. Metrics
live only on the alpha resource.

**Why**: the previous implementation expected `is` in the poll body.
That expectation was wrong and produced `poll response missing 'is'
block` on every successful simulation. A direct probe of BRAIN (a
`close` alpha) confirmed the real shapes:

    POLL 1-7:  HTTP 200  {"progress": 0.1} ... {"progress": 0.35}
    POLL 8:    HTTP 200  {"id": "...", "status": "COMPLETE",
                          "alpha": "9qEqww9V", "regular": "close", ...}

No `is`. No `sharpe`. Just an alpha id.

The two-step flow mirrors how the reference project's
`SimulationPoller.wait_for_simulation` works: on COMPLETE it does
`GET /alphas/{alpha_id}` to obtain the metrics. The extra request per
simulation is negligible compared to the minutes the simulation
itself takes.

The alternative — asking BRAIN for a "return metrics inline" mode —
does not exist. The platform's contract is fixed.

**Revisit if**: BRAIN adds an `is` block to the simulation poll
response (e.g. for REGULAR-type simulations). Then the second GET can
be skipped when `is` is present. Add the branch; do not remove the
`/alphas/{id}` fetch, because it will still be needed for other
simulation types.

---

## D-111 — WARNING with a message is terminal

**Decision**: when the poll body has `status == "WARNING"` and a
non-empty `message`, the client returns a FAILED `SimulationResult`
immediately. It does not sleep, retry, or keep polling. Only a
WARNING with no message is treated as "keep waiting".

**Why**: the reference project documented this exact hazard:

> BRAIN's WARNING with a message is a **terminal** state (computation
> finished but with an error), not "in progress". Previously, a
> warning was treated as ongoing, causing WARNING → wait 60s →
> timeout → retry → WARNING → ... Each incompatible-unit alpha
> wasted three simulations.

A simulation that has emitted a WARNING with a message will not
later emit COMPLETE. Treating it as ongoing wastes wall-clock time
and, worse, simulation budget on retries that cannot succeed. The
correct response is to record the failure and move on.

The "WARNING with no message" branch is kept as a continuing state
because we have not observed a message-less WARNING in practice, and
the reference project's own heuristic (wait up to
`min(warning_timeout, 3 * poll_interval)`) suggests the state can be
transient in rare cases. When in doubt, waiting is safer than
failing — but only when the server has not told us the reason.

**Revisit if**: BRAIN starts emitting non-terminal WARNINGs with
messages. Then this rule needs a discriminator (e.g. a specific
message prefix). Do NOT relax it on the assumption that a message
might be informational.

---

## D-112 — A semaphore bounds simultaneously active simulations

**Decision**: `BrainClient` takes a `max_concurrent` parameter
(default 2) and holds a `BoundedSemaphore` with that many permits.
`__call__` acquires a permit before the submit and releases it after
the poll loop returns, so a permit is held for the whole
submit+poll cycle, not just the POST.

**Why**: BRAIN enforces a per-account concurrency cap. Exceeding it
returns HTTP 429 with `{"detail": "CONCURRENT_SIMULATION_LIMIT_EXCEEDED"}`.
This is a distinct failure mode from the per-minute request-rate
limit; it depends on how many simulations the account is currently
running, not how many requests have been issued.

The observed behavior during Phase 4 with a TUTORIAL account:
submitting 8 tasks with `n_workers=4` produced 5 concurrency 429s
out of 8. The cap for this account tier is around 2-3.

Holding the permit across the poll loop is the correct semantics: a
simulation occupies a BRAIN-side slot for its entire runtime
(minutes), not just for the submit POST (milliseconds). If we
released after submit, the semaphore would not bound anything
useful.

Default 2 matches the low end of the observed cap. Users with a
higher account tier can raise it; users on TUTORIAL should not need
to. The parameter is exposed on `build_default_client` so the CLI
can add a flag later if requested.

`BoundedSemaphore` (not `Semaphore`) so an accidental double-release
raises immediately rather than silently inflating the permit count.

**Revisit if**: BRAIN starts advertising the concurrency cap in a
response header (e.g. `x-concurrency-limit`). Then the client can
read it at login and set `max_concurrent` automatically. Do NOT
hard-code a number above the documented minimum.

---

## D-113 — BrainSettings defaults match BRAIN-accepted values

**Decision**: `BrainSettings` now defaults to `decay=4` and
`neutralization="INDUSTRY"`. Previously it was `decay=0` and
`neutralization="SUBINDUSTRY"`.

**Why**: the earlier defaults were plausible guesses that were never
verified. The reference project's `config/settings.py` uses
`decay=4` and `neutralization="INDUSTRY"`, and the Phase 4 probe
submitted a simulation with exactly these values and received a
clean `COMPLETE` from BRAIN. A second data point (this project's own
successful probe) confirms they are accepted.

`decay=0` and `SUBINDUSTRY` may also be legal, but we have no
positive evidence for either. Changing the defaults to a known-good
pair removes one source of "why did every simulation fail"
ambiguity without costing anything in expressiveness: the user can
still override every field.

The test `test_settings_defaults_match_brain_accepted_values` pins
both fields. If a future update changes them, the test fails with a
clear message pointing at this decision.

**Revisit if**: BRAIN documents that these values are account-tier
dependent. Then the defaults stay at the most permissive pair the
lowest tier accepts; the TUTORIAL tier currently accepts
`INDUSTRY` and `decay=4`.

---

---

## D-114 — HTTP 202 is accepted without body parsing

**Decision**: `_poll` treats HTTP 202 as "accepted, not done": it
sleeps per the Retry-After header and continues the loop, without
attempting to parse the response body. Only HTTP 200 goes through
`resp.json()`.

**Why**: an earlier edit collapsed the `not in (200, 202)` branch
into a `!= 200` check, which sent 202 responses into the JSON parse
path. Since 202 responses from our test transport carry no body,
`json.loads("")` raised and every simulated 202 became a
`BrainProtocolError`. Three tests failed on this (`waits_when_poll_
returns_202`, `honors_retry_after`, `exceeds_max_polls`).

The fix is a two-branch structure: 202 handled first (sleep +
continue), then everything-other-than-200 handled as error/retry.
The 202 case is structurally different from a 200: it means the
server has not yet finished and has no data to send. A body-parsing
step before that branch is a mistake regardless of whether the body
happens to be JSON.

BRAIN's own contract, as observed by the probe, never returns 202 —
it returns 200 with `{"progress": x}` while running. But the
reference project accepts 202, the HTTP standard defines it as
"Accepted", and the cost of accepting it is one extra branch. When
the server behavior is uncertain, staying on the permissive side of
the HTTP standard is the safer default.

**Revisit if**: BRAIN starts returning 202 with a meaningful body
(e.g. a queue position). Then that body is parsed in the 202 branch;
do not move the branch order back.

---

## D-115 — Progress logging at INFO in Coordinator and BrainClient

**Decision**: the Coordinator logs one INFO line per generation
(submissions count and post-evaluation summary). BrainClient logs at
DEBUG on submit, on poll completion, and on alpha metric fetch.

**Why**: a user watching a real run asked "why is nothing happening"
because the only output between "starting new run" and "run finished"
was silence for 15 minutes. Real simulations take 2-3 minutes each;
without progress logs, the process looks hung.

INFO for the Coordinator is the right level: one or two lines per
generation is a manageable summary. DEBUG for BrainClient keeps the
per-simulation noise off unless someone is debugging a specific
failure; a 30-generation run with 20 individuals each would otherwise
print 600+ lines at INFO.

**Revisit if**: INFO becomes too sparse for long generations, or DEBUG
becomes too noisy even for debugging. Then split into a `wqgp.progress`
logger that stays at INFO, and move everything else to DEBUG.

---

## D-116 — LoggingSettings is structured, not a single level

**Decision**: `LoggingSettings` carries `level`, `style`,
`components` (per-logger overrides), and `file`. Style is a name from
a fixed set (`default`, `compact`, `minimal`), not a raw Python
format string.

**Why**: a single `level` field is what the reference project has,
and it was the immediate cause of a user request for more control.
Two concrete needs appeared:

1. "Show me what BrainClient is doing, but not the executor's retry
   noise" — solved by per-component levels, using Python's logging
   hierarchy.
2. "Different output when I'm watching versus logging to a file" —
   solved by named styles instead of format strings.

Exposing Python's `%(...)s` format syntax in a JSON file is a footgun:
a typo silently produces broken output, and the user has to learn a
Python-internal DSL. Named styles give three useful shapes and take
one line to extend later.

**Revisit if**: someone needs a style we don't ship. Add it to
`_STYLES` and document it. Do NOT open up raw format strings in the
config.

---

## D-117 — One JSON file with a `logging` block and a `defaults` block

**Decision**: `Settings` reads a single JSON file. The file has two
accepted top-level keys: `logging` (a `LoggingSettings` dict) and
`defaults` (a `DefaultsSettings` dict). Unknown top-level keys are
rejected with a clear error.

**Why**: the user asked for "one settings file controlling
everything". A single file with two well-scoped blocks is the minimal
answer. It does not try to absorb every existing config object
(`RunConfig`, `Budget`, `SelectionConfig` stay where they are) —
those have different lifetimes and different owners.

The `defaults` block is deliberately narrow. It carries values that
are usually the same across runs: BRAIN region/universe/delay,
concurrency, seed, generation cap, checkpoint root. It does NOT carry
`run_id`, `pool_path`, or budget limits — those change per run and
must be supplied on the command line or in a per-run config.

Rejecting unknown keys is a deliberate strictness. A typo'd
`"loging"` block would otherwise be silently ignored, and the user
would wonder why their log level had no effect.

**Revisit if**: the file needs a third block (e.g. a `notifications`
block for Telegram). Add the key to the known set and to `Settings`.
Do NOT switch to "accept any key and warn"; the strict version has
already prevented a typo class.

---

## D-118 — CLI flag beats file; file beats code default

**Decision**: the `--log-level` flag, when present, overrides
`logging.level` from the config file. When `--log-level` is absent,
the file's value is used. When the file is absent, `LoggingSettings()`
defaults apply. The same rule applies to every `defaults` field: a
CLI flag overrides the file, the file overrides the code default.

**Why**: this is the same precedence rule the CLI already uses for
per-run flags (D-088), extended to the new config blocks. One rule,
one place to reason about it.

Implementation detail: `--log-level` has `default=None` rather than
`default="INFO"`. `None` means "not given", so the resolver can tell
"user explicitly asked for INFO" apart from "user said nothing". If
the flag were `default="INFO"`, the file's `"level": "DEBUG"` would
never take effect.

**Revisit if**: we add environment variables as a third layer.
Precedence would become flag > env > file > default. That is a
reasonable extension; add it here and in one resolver.

---

## D-119 — Component log levels use Python's logger hierarchy

**Decision**: `logging.components` is a mapping from logger name to
level. Names may be full (`wqgp.brain.client`) or prefixes (`wqgp`).
The setup function calls `logging.getLogger(name).setLevel(...)` for
each entry, after resetting any previously set overrides to NOTSET.

**Why**: Python's logging module already implements hierarchical
level resolution: a child logger inherits its parent's effective level
unless it has its own. Delegating to that mechanism means we do not
have to write a matcher, a fallback chain, or any custom dispatch —
one `setLevel` call per entry and the standard library does the rest.

The reset step is important. Without it, removing an override from
the config (e.g. changing `wqgp.brain.client` from DEBUG back to
unset) has no effect on a long-running process that already called
`setup_logging` once. The `_LAST_OVERRIDES` set tracks which loggers
we touched, so the next call can reset exactly those and only those.

**Revisit if**: we need per-handler filtering (different levels for
stderr vs file). Then `setup_logging` gains two lists, one per
handler, and each handler gets `logging.Filter` instances. Do NOT
invent a custom filter language on top of the config; filters are
already a Python concept.

---

---

## D-120 — _load_settings uses getattr for the optional --config flag

**Decision**: `_load_settings` reads `--config` (and `--log-level`)
via `getattr(args, name, None)`. Subcommands that do not define
`--config` (`build-pool`, `list-runs`, `signatures`) work unchanged.

**Why**: `--config` is defined only on the `run` subcommand. After
D-115..D-119 introduced a global settings load in `main`, reading
`args.config` directly raised `AttributeError` for every subcommand
that lacked the flag. Six CLI tests failed with exit code 1 because
`main`'s catch-all handler converted the `AttributeError` to
`EXIT_ERROR`.

The alternative — adding `--config` to every subcommand — was
rejected for now because it changes the CLI surface for subcommands
that have no use for the `defaults` block. `list-runs` reads its
checkpoint root from a flag; the settings file would add nothing.

**Revisit if**: we decide all subcommands should honor the settings
file's `defaults` block (e.g. `build-pool` picking up
region/universe from the file). Then add `--config` to the shared
`common` parent parser and remove the `getattr` fallback. That is a
deliberate CLI expansion, not a bug fix.

---

---

## D-121 — Settings ignores unknown top-level keys

**Decision**: `Settings.from_dict` reads the `logging` and `defaults`
blocks and ignores every other top-level key. Unknown keys INSIDE
`logging` or `defaults` are still rejected.

**Why**: the same JSON file is read by two loaders:

1. `Settings` (via `main` → `_load_settings`) for `logging` and
   `defaults`.
2. `_load_config` (in `_cmd_run`) for run-level keys: `run_id`,
   `pool_path`, `budget`, `selection`, etc.

D-117 made `Settings` reject unknown top-level keys, which caused
both loaders to fight over the file: `_cmd_run` wanted
`run_id` / `pool_path` at the top level, and `Settings` rejected them
with "unknown top-level settings keys: [...]". Two CLI tests failed
immediately, and every user who put run-level keys in `--config`
would have hit the same error.

Relaxing the top level is the minimal fix that keeps the useful part
of D-117's strictness. A typo'd `"loging"` block is still caught
because the unknown key would appear... wait, no. If the top level is
lenient, `"loging": {"level": "DEBUG"}` is silently ignored.

This is a real downside. The trade-off:

- Strict top level: catches "loging" typos, but forces users to split
  their config into two files.
- Lenient top level: single file works, but a typo'd block is silent.

We chose lenient because splitting the file is a bigger usability
cost than a silent typo that is easy to notice (the log level simply
does not change; the user investigates; the typo is found in five
seconds). The `logging` block's own keys are still strict, which
catches `"levl"` once the user gets into the block.

**Revisit if**: users repeatedly hit the silent-typo case. Then
`Settings.load` can log a warning for any top-level key that is not
in the recognized set and not in the run-config's known set. That
keeps one file while restoring a signal for typos. Do NOT re-tighten
the top level without also splitting the CLI flag into `--settings`
and `--run-config`.

---