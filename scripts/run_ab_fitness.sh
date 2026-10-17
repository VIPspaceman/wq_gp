#!/bin/bash
set -u

PY="$HOME/python/.venv/bin/python"

if [ ! -x "$PY" ]; then
    echo "ERROR: venv python not found at $PY" >&2
    exit 1
fi

echo "using python: $PY"
"$PY" -c "import wqgp; print('wqgp import OK')" || exit 1

cd "$HOME/python/wq_gp"
mkdir -p logs

echo "=== A/B started at $(date) ==="

echo ""
echo "=== RUN 1/2: sharpe (legacy) ==="
"$PY" -m wqgp run --settings config/wqgp.json \
    --pool pool.json --run-id fitness_ab_sharpe \
    --total-simulations 300 --population-size 8 --max-generations 15 \
    --seed 2026 --checkpoint-root ./checkpoints \
    --fitness-strategy sharpe 2>&1 | tee logs/fitness_ab_sharpe.log

echo ""
echo "=== RUN 2/2: feasibility (new) ==="
"$PY" -m wqgp run --settings config/wqgp.json \
    --pool pool.json --run-id fitness_ab_feasibility \
    --total-simulations 300 --population-size 8 --max-generations 15 \
    --seed 2026 --checkpoint-root ./checkpoints \
    --fitness-strategy feasibility 2>&1 | tee logs/fitness_ab_feasibility.log

echo ""
echo "=== A/B complete at $(date) ==="