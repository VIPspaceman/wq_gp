# wqgp

WorldQuant BRAIN alpha mining via genetic programming.

## Status

Phase 1 (foundation) — COMPLETE.
Phase 2 (skeleton)   — COMPLETE.
Phase 3 (GP core)    — COMPLETE.

Phase 4 (polish) — IN PROGRESS.

- [x] `brain/pool_builder.py`  — fetch fields + operator names from BRAIN
- [x] `cli.py`                 — command-line entry point
- [x] `notification/`          — run lifecycle notifications (Telegram)
- [x] `signatures_audit.py`    — audit a signature table against BRAIN
- [ ] `pareto.py`              — multi-objective selection
- [ ] `tools/`                 — analysis scripts

See `DECISIONS.md` for every deviation from the reference project.

## Install

    pip install -e ".[dev]"

## Usage

Secrets are read only from the environment:

    export WQ_BRAIN_USERNAME=...
    export WQ_BRAIN_PASSWORD=...

### 1. Audit / build the signature table

BRAIN's `/operators` endpoint returns names and descriptions only, not
type signatures. The signature table is a hand-curated JSON file that
supplies the rest. To see which operators BRAIN exposes that are
missing from your table:

    wqgp signatures --signatures config/signatures.json

Output lists missing operators and prints paste-ready JSON skeletons.
Add the ones you need to `config/signatures.json`, then verify each signature
against BRAIN's docs — the `"MATRIX"` placeholder is a guess.

Use `--strict` in CI to fail when the table is incomplete.

### 2. Build the pool

    wqgp build-pool \
        --signatures config/signatures.json \
        --output pool.json

Fetches BRAIN's data fields and operator names, joins them with the
signature table, and writes the result. Operators named by BRAIN but
not present in the table are dropped and reported.

### 3. Run the GP

    wqgp run \
        --pool pool.json \
        --run-id my_first_run \
        --total-simulations 1000 \
        --population-size 20 \
        --max-generations 10 \
        --seed 42 \
        --checkpoint-root ./checkpoints

Re-running with the same `--run-id` resumes from the last checkpoint.

### Multi-objective selection

By default, selection ranks by sharpe only. To select on a Pareto
front over multiple metrics:

    wqgp run \
        --pool pool.json \
        --run-id pareto_run \
        --total-simulations 1000 \
        --mode pareto

Or in a config file:

    {
      "selection": {
        "population_size": 20,
        "elite_count": 4,
        "mode": "pareto",
        "pareto_objectives": [
          ["sharpe", "MAXIMIZE"],
          ["turnover", "MINIMIZE"]
        ]
      }
    }

The default objectives, when `pareto_objectives` is omitted, are
`sharpe` (maximize) and `turnover` (minimize). Both metrics are stored
in the evaluation record by the default BrainClient.

### 4. Telegram notifications

    export TELEGRAM_BOT_TOKEN=...
    export TELEGRAM_CHAT_ID=...

    wqgp run ... --telegram

Lifecycle events (run start, each generation's best fitness, run end)
are pushed to the configured chat. If Telegram is unreachable, the
run continues; only a warning is logged.

### 5. Smoke-test without network

    wqgp run \
        --pool pool.json \
        --run-id smoke \
        --total-simulations 100 \
        --max-generations 2 \
        --dry-run

`--dry-run` uses a deterministic fake evaluator. No credentials needed.

### 6. Inspect runs

    wqgp list-runs --checkpoint-root ./checkpoints

### Config file

Any `run` flag can also be set in a JSON config passed via `--config`.
Flags override config values.

    {
      "run_id": "my_run",
      "pool_path": "./pool.json",
      "checkpoint_root": "./checkpoints",
      "seed": 42,
      "max_generations": 10,
      "n_workers": 4,
      "budget": { "total_simulations": 1000 },
      "selection": { "population_size": 20, "elite_count": 2 }
    }

Credentials are NOT accepted in the config file.

## Layout

    src/wqgp/types.py             # DataType, SemanticKind, ParamDomain
    src/wqgp/field.py             # FieldDef
    src/wqgp/operator.py          # OperatorDef
    src/wqgp/ast.py               # ASTNode, tree utils, render
    src/wqgp/budget.py            # Budget, BudgetTracker, BudgetSnapshot
    src/wqgp/checkpoint.py        # Checkpoint, CheckpointStore
    src/wqgp/serialization.py     # Field / Operator / ParamDomain JSON
    src/wqgp/pool.py              # FieldPool, OperatorPool, Pool
    src/wqgp/persistence.py       # EvaluationRecord, EvaluationStore
    src/wqgp/_ids.py              # run_id validation
    src/wqgp/execution.py         # Task, TaskResult, Executor
    src/wqgp/population.py        # Individual, Population, RandomAstBuilder
    src/wqgp/mutation.py          # Mutator + 4 strategies
    src/wqgp/crossover.py         # Crossover, subtree_crossover
    src/wqgp/selection.py         # SelectionConfig, tournament, elitism
    src/wqgp/gp_serialization.py  # AST / Individual / Population JSON
    src/wqgp/coordinator.py       # RunConfig, Coordinator, RunResult
    src/wqgp/signatures_audit.py  # audit_signatures, SignatureAudit
    src/wqgp/cli.py               # command-line entry point
    src/wqgp/__main__.py          # python -m wqgp
    src/wqgp/brain/
      __init__.py
      transport.py                # HttpTransport, HttpResponse, RequestsTransport
      client.py                   # BrainCredentials, BrainSettings, BrainClient
      pool_builder.py             # PoolBuilder, OperatorSignatureTable
    src/wqgp/notification/
      __init__.py
      events.py                   # RunStartEvent, GenerationEndEvent, RunEndEvent
      protocol.py                 # Notifier Protocol
      null.py                     # NullNotifier
      telegram.py                 # TelegramNotifier, TelegramCredentials
    tests/
      _fakes.py                   # ScriptedTransport (shared test double)
      ...

## Run tests

    pytest