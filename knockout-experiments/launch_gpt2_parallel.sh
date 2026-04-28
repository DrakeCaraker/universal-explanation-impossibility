#!/bin/bash
# Launch GPT-2 IOI experiment across 4 GPUs in parallel
# For ml.g5.12xlarge (4x A10G)
#
# Usage: bash launch_gpt2_parallel.sh
#
# Trains 10 models in 3 rounds (4+4+2), then runs full analysis.

set -e
cd "$(dirname "$0")"

pip install torch transformers datasets scipy 2>/dev/null

N_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
echo "=== GPT-2 IOI Circuit Stability — ${N_GPUS}-GPU Parallel ==="
echo ""

# Train in rounds of N_GPUS
SEEDS=(0 1 2 3 4 5 6 7 8 9)
ROUND=0

for ((i=0; i<${#SEEDS[@]}; i+=N_GPUS)); do
    ROUND=$((ROUND+1))
    BATCH=("${SEEDS[@]:i:N_GPUS}")
    echo "=== ROUND $ROUND: Seeds ${BATCH[*]} ==="

    PIDS=()
    for j in "${!BATCH[@]}"; do
        seed=${BATCH[$j]}
        gpu=$j
        echo "  Starting seed $seed on GPU $gpu"
        python3 train_single_seed.py --seed $seed --gpu $gpu \
            > gpt2_seed${seed}.log 2>&1 &
        PIDS+=($!)
    done

    # Wait for this round
    for pid in "${PIDS[@]}"; do
        wait $pid
    done

    echo "  Round $ROUND complete:"
    for seed in "${BATCH[@]}"; do
        tail -1 gpt2_seed${seed}.log
    done
    echo ""
done

# Verify all models exist
echo "=== Checking trained models ==="
COUNT=$(ls gpt2_ioi_models/model_seed*_final.pt 2>/dev/null | wc -l)
echo "$COUNT / 10 models trained"

if [ "$COUNT" -lt 10 ]; then
    echo "ERROR: Not all models trained. Check gpt2_seed*.log files."
    exit 1
fi

# Run full analysis on GPU 0
echo ""
echo "=== Running full analysis ==="
CUDA_VISIBLE_DEVICES=0 python3 gpt2_ioi_circuit_stability.py \
    2>&1 | tee gpt2_ioi_full_log.txt

echo ""
echo "=== DONE ==="
