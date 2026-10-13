# wqgp

WorldQuant BRAIN alpha mining via genetic programming.

## Status

Phase 1 (foundation) — COMPLETE.
Phase 2 (skeleton)   — COMPLETE.

Phase 3 (GP core) — IN PROGRESS.

- [x] `population.py`       — Individual, Population, RandomAstBuilder
- [x] `mutation.py`         — window / field / operator / subtree mutations
- [x] `crossover.py`        — subtree crossover with type and size guards
- [x] `selection.py`        — tournament, elitism, next_generation
- [x] `gp_serialization.py` — AST / Individual / Population round-trip
- [x] `coordinator.py`      — end-to-end GP loop with checkpoints
- [x] `brain/transport.py`  — HttpTransport protocol + requests backend
- [x] `brain/client.py`     — BrainClient (real Evaluator against BRAIN)
- [ ] `pareto.py`           — multi-objective (deferred)

See `DECISIONS.md` for every deviation from the reference project.

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
    src/wqgp/brain/
      __init__.py
      transport.py                # HttpTransport, HttpResponse, RequestsTransport
      client.py                   # BrainCredentials, BrainSettings, BrainClient
    tests/                        # Phase 1 + Phase 2 + Phase 3 gates

## Usage

    export WQ_BRAIN_USERNAME=...
    export WQ_BRAIN_PASSWORD=...

    from wqgp.brain.client import build_default_client
    from wqgp.coordinator import Coordinator, RunConfig
    from wqgp.pool import Pool
    from wqgp.budget import Budget
    from wqgp.selection import SelectionConfig

    client = build_default_client()
    config = RunConfig(
        run_id="my_first_run",
        pool=Pool.from_file("pool.json"),
        budget=Budget(total_simulations=1000),
        selection=SelectionConfig(population_size=20),
        seed=42,
        max_generations=10,
    )
    result = Coordinator(config, client).run()
    print(result.stopped_reason, result.total_simulations_used)

## Run tests

    pip install -e ".[dev]"
    pytest