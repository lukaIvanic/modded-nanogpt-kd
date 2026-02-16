#!/bin/bash
# Teacher pretraining sweep: 3 batch sizes × 3 step counts = 9 runs
# All runs: 3L / 6H / 128D / 768dim, warmup+cosine LR
# Estimated total time: ~5 min

set -e

echo "=== Teacher Pretraining Sweep ==="
echo "Start: $(date)"
echo ""

CASES=(
    # BATCH_SIZE  NUM_ITERATIONS  LABEL
    "131072       250             bs100_s250"
    "131072       400             bs100_s400"
    "131072       600             bs100_s600"
    "98304        250             bs75_s250"
    "98304        400             bs75_s400"
    "98304        600             bs75_s600"
    "65536        250             bs50_s250"
    "65536        400             bs50_s400"
    "65536        600             bs50_s600"
)

for case in "${CASES[@]}"; do
    read -r BS STEPS LABEL <<< "$case"
    echo "--- [$LABEL] batch_size=$BS steps=$STEPS ---"
    BATCH_SIZE=$BS NUM_ITERATIONS=$STEPS SAVE_CHECKPOINT=0 \
        torchrun --standalone --nproc_per_node=1 train_gpt_teacher.py
    echo ""
done

echo "=== Sweep complete: $(date) ==="
echo ""
echo "Results summary:"
echo "----------------"
for f in logs/dev/*.txt; do
    BEST=$(grep "Best val_loss" "$f" 2>/dev/null | tail -1)
    TIME=$(grep "step:.*val_loss.*train_time" "$f" 2>/dev/null | tail -1 | grep -o 'train_time:[0-9]*ms' | head -1)
    CONFIG=$(grep "Teacher config" "$f" 2>/dev/null | tail -1)
    if [ -n "$BEST" ]; then
        echo "$f: $CONFIG | $TIME | $BEST"
    fi
done
