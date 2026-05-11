#!/bin/bash
# Run directly — no SBATCH needed.
# Usage:
#   bash smoketest_local.sh                   # auto-detects GPU
#   bash smoketest_local.sh --data_dir /path  # override data dir
set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
CHALLENGE_DIR=/mnt/f/challenge
DATA_DIR=/mnt/f/challenge/fomo_test_preprocessed 
OUTPUT_DIR=/mnt/f/challenge/outputs/local_smoketest

export PYTHONPATH=${CHALLENGE_DIR}:${CHALLENGE_DIR}/dinov3:$PYTHONPATH

mkdir -p "${OUTPUT_DIR}"

LAST_CKPT="${OUTPUT_DIR}/last.ckpt"
echo "=========================================="
echo "Smoketest start: $(date)"
echo "Data:   $DATA_DIR"
echo "Output: $OUTPUT_DIR"
if [ -f "$LAST_CKPT" ]; then
    echo "Resume: $LAST_CKPT"
else
    echo "Resume: none (fresh start)"
fi
echo "=========================================="

python ${CHALLENGE_DIR}/train_dino3d.py \
    --data_dir          ${DATA_DIR} \
    --output_dir        ${OUTPUT_DIR} \
    \
    --epochs            2 \
    --warmup_epochs     1 \
    --batch_size        16 \
    --grad_accum        1 \
    --num_workers       8 \
    \
    --global_crop_size  96 \
    --local_crop_size   48 \
    --local_crops_n     4 \
    --ibot_mask_ratio   0.3 \
    \
    --lr                1e-3 \
    --min_lr            1e-6 \
    --weight_decay      0.04 \
    --clip_grad         3.0 \
    --freeze_last_layer 1 \
    \
    --dino_prototypes   65536 \
    --ibot_prototypes   65536 \
    --teacher_momentum  0.992 \
    --teacher_temp      0.07 \
    --warmup_temp       0.04 \
    --warmup_temp_epochs 1 \
    --no_wandb \
    --save_every        999 \
    --precision         bf16-mixed \
    --experiment        smoketest

echo "=========================================="
echo "Smoketest done: $(date)"
echo "=========================================="
echo "Check for:"
echo "  - 'Dataset: N scans' printed"
echo "  - step prints every 50 steps (may not hit with max_steps=20)"
echo "  - loss / dino / ibot / koleo all finite (not NaN)"
echo "  - No CUDA OOM"
