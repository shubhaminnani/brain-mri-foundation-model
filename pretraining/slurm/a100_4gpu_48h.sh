#!/bin/bash
#SBATCH --job-name=fomo26-dino3d
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4          # 1 task per GPU
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=8            # 8 CPU workers per GPU for data loading
#SBATCH --mem=256G                   # 64G per GPU is plenty for ViT-B 3D
#SBATCH --time=48:00:00
#SBATCH --partition=gpu              # adjust to your cluster's partition name
#SBATCH --output=logs/dino3d_%j.out
#SBATCH --error=logs/dino3d_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
# #SBATCH --mail-user=your@email.com  # uncomment and fill in

# ── Environment ────────────────────────────────────────────────────────────────
module purge
module load cuda/12.1          # adjust to your cluster's CUDA version
module load python/3.11

# Activate your venv / conda env:
# conda activate fomo26
# OR:
# source /path/to/venv/bin/activate

# NCCL tuning for A100 NVLink
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=^lo,docker
export NCCL_DEBUG=WARN

# Ensure dinov3 loss modules are importable
export PYTHONPATH=/mnt/f/challenge:/mnt/f/challenge/dinov3:$PYTHONPATH

# ── Paths ──────────────────────────────────────────────────────────────────────
CHALLENGE_DIR=/mnt/f/challenge
FOMO300K_DIR=/path/to/fomo300k_preprocessed   # ← set this
OUTPUT_DIR=${CHALLENGE_DIR}/outputs/dino3d_a100
BRAINIAC_CKPT=${CHALLENGE_DIR}/BrainIAC/checkpoints/brainiac.ckpt  # SimCLR init

mkdir -p ${OUTPUT_DIR} logs

# ── Distributed launch ─────────────────────────────────────────────────────────
# Lightning auto-spawns 4 DDP processes for --gpus-per-node=4
python ${CHALLENGE_DIR}/train_dino3d.py \
    \
    --data_dir          ${FOMO300K_DIR} \
    --output_dir        ${OUTPUT_DIR} \
    --brainiac_ckpt     ${BRAINIAC_CKPT} \
    \
    --epochs            100 \
    --warmup_epochs     10 \
    --batch_size        8 \
    --grad_accum        2 \
    --num_workers       8 \
    \
    --global_crop_size  96 \
    --local_crop_size   48 \
    --local_crops_n     6 \
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
    \
    --save_every        10 \
    --precision         bf16-mixed \
    --experiment        dino3d_vitb_fomo300k_a100 \
    2>&1 | tee --line-buffered ${OUTPUT_DIR}/train_${SLURM_JOB_ID}.log

# ── Compute note ───────────────────────────────────────────────────────────────
# Token count: 96³ / 16³ = 216 patch tokens + 1 CLS = 217 total
# Same order as 2D DINOv2 (224/16)² = 196 tokens → similar memory footprint
#
# Per GPU (A100-40GB):
#   - Student weights (ViT-B bf16):  ~176 MB
#   - Teacher weights (no grad):     ~176 MB
#   - Optimizer (AdamW fp32):        ~688 MB
#   - Activations (batch=8, 2g+6l):  ~350 MB
#   - Total est.:                   ~1.4 GB  (of 40 GB available)
#
# Throughput estimate (bf16, DDP):
#   - ~80-120 ms per raw step on 4× A100-40GB
#   - 306 000 samples / (8 batch × 4 GPUs) = 9 563 raw steps/epoch
#   - ≈ 16-19 min/epoch
#   - 100 epochs ≈ 27-32 hours  ← fits comfortably in 48 h
#
# Effective batch: 8 × 4 GPUs × 2 grad_accum = 64
