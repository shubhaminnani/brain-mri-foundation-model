#!/bin/bash
#SBATCH --job-name=fomo26-dino3d
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2          # 1 task per GPU
#SBATCH --gres=gpu:a6000:2
#SBATCH --cpus-per-task=12           # A6000 nodes often have more CPUs
#SBATCH --mem=128G
#SBATCH --time=14-00:00:00           # 14 days; adjust if your cluster has a shorter max
#SBATCH --partition=gpu
#SBATCH --output=logs/dino3d_%j.out
#SBATCH --error=logs/dino3d_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
# #SBATCH --mail-user=your@email.com

# ── Environment ────────────────────────────────────────────────────────────────
conda activate fomo26

export NCCL_IB_DISABLE=1            # A6000 uses PCIe, not InfiniBand
export NCCL_SOCKET_IFNAME=^lo,docker
export NCCL_DEBUG=WARN

# ── Paths ──────────────────────────────────────────────────────────────────────
CHALLENGE_DIR=/mnt/f/challenge
FOMO300K_DIR=${CHALLENGE_DIR}/fomo_test_preprocessed
OUTPUT_DIR=${CHALLENGE_DIR}/outputs/dino3d_a6000
BRAINIAC_CKPT=${CHALLENGE_DIR}/BrainIAC/src/checkpoints/brainiac.ckpt

export PYTHONPATH=${CHALLENGE_DIR}:${CHALLENGE_DIR}/dinov3:$PYTHONPATH

mkdir -p ${OUTPUT_DIR} logs

# ── Distributed launch ─────────────────────────────────────────────────────────
torchrun \
    --standalone \
    --nproc_per_node=2 \
    --nnodes=1 \
    ${CHALLENGE_DIR}/train_dino3d.py \
    \
    --data_dir          ${FOMO300K_DIR} \
    --output_dir        ${OUTPUT_DIR} \
    --brainiac_ckpt     ${BRAINIAC_CKPT} \
    \
    --epochs            150 \
    --warmup_epochs     10 \
    --batch_size        12 \
    --grad_accum        3 \
    --num_workers       12 \
    \
    --global_crop_size  96 \
    --local_crop_size   48 \
    --local_crops_n     8 \
    --ibot_mask_ratio   0.3 \
    \
    --lr                1e-3 \
    --min_lr            1e-6 \
    --weight_decay      0.04 \
    --weight_decay_end  0.4 \
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
    --experiment        dino3d_vitb_fomo300k_a6000 \
    2>&1 | tee ${OUTPUT_DIR}/train_${SLURM_JOB_ID}.log

# ── Compute note ───────────────────────────────────────────────────────────────
# A6000 has 48 GB VRAM (vs A100-40GB's 40 GB) → larger batch per GPU
# A6000 FP32/BF16 peak: ~38.7 TFLOPS tensor (vs A100's ~77.6 TFLOPS)
# → A6000 is ~50% slower per GPU but more memory, enabling larger batch
#
# Per GPU (A6000-48GB):
#   - Student + teacher + optimizer + activations: ~1.8 GB (batch=12)
#   - Headroom for larger batch or local_crops_n=8
#
# Throughput estimate:
#   - ~150-200 ms per raw step on 2× A6000
#   - 306 000 / (12 × 2) = 12 750 raw steps/epoch
#   - ≈ 35-42 min/epoch
#   - 150 epochs ≈ 88-105 hours ≈ 4-4.5 days  ← well within no-time-limit
#
# Effective batch: 12 × 2 GPUs × 3 grad_accum = 72
# Use more local crops (8 vs 6) since extra VRAM allows it
# Extra epochs (150 vs 100) since time is not a constraint
