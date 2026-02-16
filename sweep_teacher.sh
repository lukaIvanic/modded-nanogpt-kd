#!/bin/bash
# Teacher pretraining sweep: 3 layer counts × 3 batch sizes × 3 step counts = 27 runs
# All runs: 6H / 128D / 768dim, warmup+cosine LR

set -e

LOG_DIR="logs/dev/pretrain-teacher/full-27-sweep"

echo "=== Teacher Pretraining Sweep (27 runs) ==="
echo "Start: $(date)"
echo ""

CASES=(
    # LAYERS  BATCH_SIZE  NUM_ITERATIONS  LABEL
    "3  131072  250  3L_bs128k_s250"
    "3  131072  400  3L_bs128k_s400"
    "3  131072  600  3L_bs128k_s600"
    "3  98304   250  3L_bs96k_s250"
    "3  98304   400  3L_bs96k_s400"
    "3  98304   600  3L_bs96k_s600"
    "3  65536   250  3L_bs64k_s250"
    "3  65536   400  3L_bs64k_s400"
    "3  65536   600  3L_bs64k_s600"
    "6  131072  250  6L_bs128k_s250"
    "6  131072  400  6L_bs128k_s400"
    "6  131072  600  6L_bs128k_s600"
    "6  98304   250  6L_bs96k_s250"
    "6  98304   400  6L_bs96k_s400"
    "6  98304   600  6L_bs96k_s600"
    "6  65536   250  6L_bs64k_s250"
    "6  65536   400  6L_bs64k_s400"
    "6  65536   600  6L_bs64k_s600"
    "9  131072  250  9L_bs128k_s250"
    "9  131072  400  9L_bs128k_s400"
    "9  131072  600  9L_bs128k_s600"
    "9  98304   250  9L_bs96k_s250"
    "9  98304   400  9L_bs96k_s400"
    "9  98304   600  9L_bs96k_s600"
    "9  65536   250  9L_bs64k_s250"
    "9  65536   400  9L_bs64k_s400"
    "9  65536   600  9L_bs64k_s600"
)

for case in "${CASES[@]}"; do
    read -r LAYERS BS STEPS LABEL <<< "$case"
    echo "--- [$LABEL] layers=$LAYERS batch_size=$BS steps=$STEPS ---"
    NUM_LAYERS=$LAYERS BATCH_SIZE=$BS NUM_ITERATIONS=$STEPS SAVE_CHECKPOINT=0 LOG_DIR=$LOG_DIR \
        torchrun --standalone --nproc_per_node=1 train_gpt_teacher.py
    echo ""
done

echo "=== Sweep complete: $(date) ==="
echo ""
echo "Results summary:"
echo "----------------"
for f in $LOG_DIR/*.txt; do
    BEST=$(grep "Best val_loss" "$f" 2>/dev/null | tail -1)
    TIME=$(grep "step:.*val_loss.*train_time" "$f" 2>/dev/null | tail -1 | grep -o 'train_time:[0-9]*ms' | head -1)
    CONFIG=$(grep "Teacher config" "$f" 2>/dev/null | tail -1)
    if [ -n "$BEST" ]; then
        echo "$f: $CONFIG | $TIME | $BEST"
    fi
done
