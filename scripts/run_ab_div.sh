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

echo "=== A/B diversity started at $(date) ==="

echo ""
echo "=== RUN 1/2: diversity OFF (baseline) ==="
"$PY" -m wqgp run --settings config/wqgp.json \
    --pool pool.json --run-id ab_div_off \
    --total-simulations 200 --population-size 8 --max-generations 12 \
    --seed 2026 --checkpoint-root ./checkpoints \
    --no-diversity 2>&1 | tee logs/ab_div_off.log

echo ""
echo "=== RUN 2/2: diversity ON (A + A') ==="
"$PY" -m wqgp run --settings config/wqgp.json \
    --pool pool.json --run-id ab_div_on \
    --total-simulations 200 --population-size 8 --max-generations 12 \
    --seed 2026 --checkpoint-root ./checkpoints \
    2>&1 | tee logs/ab_div_on.log

echo ""
echo "=== A/B diversity complete at $(date) ==="
