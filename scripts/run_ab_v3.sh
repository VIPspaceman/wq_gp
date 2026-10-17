#!/bin/bash
set -u
PY="$HOME/python/.venv/bin/python"
if [ ! -x "$PY" ]; then
  echo "ERROR: venv python not found at $PY" >&2
  exit 1
fi
"$PY" -c "import wqgp; print('wqgp import OK')" || exit 1

cd "$HOME/python/wq_gp"
mkdir -p logs

echo "=== A/B v3 started at $(date) ==="

echo ""
echo "=== RUN 1/2: sharpe (baseline) ==="
"$PY" -m wqgp run --settings config/wqgp.json \
    --pool pool.json --run-id ab_v3_sharpe \
    --total-simulations 200 --population-size 8 --max-generations 12 \
    --seed 2026 --checkpoint-root ./checkpoints \
    --fitness-strategy sharpe 2>&1 | tee logs/ab_v3_sharpe.log

echo ""
echo "=== RUN 2/2: feasibility (v3) ==="
"$PY" -m wqgp run --settings config/wqgp.json \
    --pool pool.json --run-id ab_v3_feasibility \
    --total-simulations 200 --population-size 8 --max-generations 12 \
    --seed 2026 --checkpoint-root ./checkpoints \
    --fitness-strategy feasibility 2>&1 | tee logs/ab_v3_feasibility.log

echo ""
echo "=== A/B v3 complete at $(date) ==="
