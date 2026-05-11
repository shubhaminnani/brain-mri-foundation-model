#!/bin/bash
#SBATCH --job-name=dino3d_v3
#SBATCH -p gpu
#SBATCH -A r00917
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1          # one torchrun launcher per node
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32           # 8 CPU workers × 4 GPUs
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=logs/dino3d_v3_%j.out
#SBATCH --error=logs/dino3d_v3_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=sinnani@iu.edu
set -euo pipefail

# ── Environment ────────────────────────────────────────────────────────────────
module load conda
conda activate /N/project/histo-exp/challenge/envs_fomo

# NCCL tuning for A100 NVLink / InfiniBand
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=^lo,docker
export NCCL_DEBUG=WARN

# ── Paths ──────────────────────────────────────────────────────────────────────
CHALLENGE_DIR=/N/project/histo-exp/challenge/fomo
FOMO300K_DIR=${CHALLENGE_DIR}/fomo_preprocessed
OUTPUT_DIR=${CHALLENGE_DIR}/outputs/dino3d_v3

# v3 uses dinov3 augmentations — no 3DINO dependency at all
export PYTHONPATH=${CHALLENGE_DIR}:${CHALLENGE_DIR}/dinov3:$PYTHONPATH

mkdir -p ${OUTPUT_DIR} logs

# ── Multi-node rendezvous ──────────────────────────────────────────────────────
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29501          # different port from v2 in case both run simultaneously

LAST_CKPT="${OUTPUT_DIR}/last.ckpt"

echo "=========================================="
echo "Job:    $SLURM_JOB_ID"
echo "Nodes:  $SLURM_JOB_NODELIST"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "GPUs:   $SLURM_NNODES nodes × $SLURM_GPUS_PER_NODE GPUs = $((SLURM_NNODES * SLURM_GPUS_PER_NODE)) total"
echo "Start:  $(date)"
echo "Output: $OUTPUT_DIR"
if [ -f "$LAST_CKPT" ]; then
    echo "Resume: $LAST_CKPT"
else
    echo "Resume: none (fresh start)"
fi
echo "=========================================="

# ── Distributed launch (one torchrun per node via srun) ───────────────────────
srun --nodes="$SLURM_NNODES" --ntasks-per-node=1 bash -c "
    torchrun \
        --nnodes=${SLURM_NNODES} \
        --nproc_per_node=4 \
        --rdzv_backend=c10d \
        --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
        --rdzv_id=job_${SLURM_JOB_ID} \
        ${CHALLENGE_DIR}/train_dino3d_v3.py \
        \
        --data_dir          ${FOMO300K_DIR} \
        --output_dir        ${OUTPUT_DIR} \
        \
        --epochs            100 \
        --warmup_epochs     10 \
        --batch_size        2 \
        --grad_accum        4 \
        --num_workers       8 \
        \
        --global_crop_size  112 \
        --local_crop_size   48 \
        --local_crops_n     6 \
        --ibot_mask_ratio   0.3 \
        --drop_path_rate    0.1 \
        --rope_base         10000.0 \
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
        --warmup_temp_epochs 30 \
        --no_wandb \
        \
        --save_every        10 \
        --precision         bf16-mixed \
        --experiment        dino3d_v3_rope3d_fomo300k_8gpu \
    2>&1 | tee ${OUTPUT_DIR}/train_${SLURM_JOB_ID}_\$(hostname).log
"

echo "=========================================="
echo "Done: $(date)"
echo "Checkpoints: $(find ${OUTPUT_DIR} -name '*.ckpt' | wc -l)"
echo "=========================================="

# ── Compute note ───────────────────────────────────────────────────────────────
# Backbone:        DINOv3-style ViT, embed_dim=1152, depth=24, heads=16
#                  head_dim=72 (72 % 6 = 0 → 3D RoPE compatible)
#                  4 storage tokens, no pos_embed, trains from scratch
# Effective batch: 2 × 8 GPUs × 4 grad_accum = 64
# Steps/epoch:     306,207 / 64 = 4,785
# Memory/GPU:      ~20 GB / 40 GB at batch=2 (confirmed by local smoke test)
# Time estimate:   ~35–40 min/epoch → 100 epochs ≈ 58–67 hours
#                  → plan for two 48 h jobs with checkpoint resume
# No pretrained:   v3 trains from scratch (RoPE ≠ absolute pos_embed)
