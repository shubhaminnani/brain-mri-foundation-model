#!/bin/bash
#SBATCH --job-name=fomo26-smoketest
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --output=logs/smoketest_%j.out
#SBATCH --error=logs/smoketest_%j.err

# ── Environment ────────────────────────────────────────────────────────────────
module purge
module load cuda/12.1
module load python/3.11

# conda activate fomo26
# source /path/to/venv/bin/activate

export PYTHONPATH=/mnt/f/challenge:/mnt/f/challenge/dinov3:$PYTHONPATH

# ── Paths ──────────────────────────────────────────────────────────────────────
CHALLENGE_DIR=/mnt/f/challenge
FOMO300K_DIR=/path/to/fomo300k_preprocessed   # ← same as full runs
OUTPUT_DIR=${CHALLENGE_DIR}/outputs/smoketest
BRAINIAC_CKPT=${CHALLENGE_DIR}/BrainIAC/checkpoints/brainiac.ckpt

mkdir -p ${OUTPUT_DIR} logs

# ── Single-GPU launch (no torchrun needed) ─────────────────────────────────────
python ${CHALLENGE_DIR}/train_dino3d.py \
    --data_dir          ${FOMO300K_DIR} \
    --output_dir        ${OUTPUT_DIR} \
    --brainiac_ckpt     ${BRAINIAC_CKPT} \
    \
    --epochs            2 \
    --warmup_epochs     1 \
    --batch_size        4 \
    --grad_accum        1 \
    --num_workers       4 \
    \
    --global_crop_size  96 \
    --local_crop_size   48 \
    --local_crops_n     4 \
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
    --warmup_temp_epochs 2 \
    \
    --max_steps         100 \
    --no_wandb \
    --save_every        999 \
    --precision         bf16-mixed \
    --experiment        smoketest \
    2>&1 | tee ${OUTPUT_DIR}/smoketest_${SLURM_JOB_ID}.log

# ── What to check in the log ──────────────────────────────────────────────────
# 1. "BrainIAC weights loaded" — warm-start OK
# 2. "Dataset: N scans" — data loading OK
# 3. step 1: train/loss shows finite values (not NaN/inf)
# 4. train/dino, train/ibot, train/koleo all finite
# 5. No CUDA OOM (4.8 GB should be used of 48 GB)
# 6. 100 steps completes in < 10 min
