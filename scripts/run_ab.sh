#!/bin/bash
set -u
cd ~/python/wq_gp
mkdir -p logs

echo "=== AB test started at $(date) ==="

echo ""
echo "=== RUN 1/2: diversity_off (no fitness sharing) ==="
python -m wqgp run \
    --pool pool.json \
    --run-id diversity_off \
    --total-simulations 300 \
    --population-size 8 \
    --max-generations 15 \
    --seed 42 \
    --checkpoint-root ./checkpoints \
    --settings config/wqgp.json \
    --no-diversity 2>&1 | tee logs/diversity_off.log

echo ""
echo "=== diversity_off finished at $(date) ==="
echo ""
echo "=== RUN 2/2: diversity_on (fitness sharing enabled) ==="

python -m wqgp run \
    --pool pool.json \
    --run-id diversity_on \
    --total-simulations 300 \
    --population-size 8 \
    --max-generations 15 \
    --seed 42 \
    --checkpoint-root ./checkpoints \
    --settings config/wqgp.json 2>&1 | tee logs/diversity_on.log

echo ""
echo "=== diversity_on finished at $(date) ==="
echo "=== AB test complete ==="
