#!/bin/bash
# Local version of setup_env_v2.sh — same packages, local env path.
# Run from WSL:  bash /mnt/f/challenge/slurm/setup_env_v2_local.sh
set -euo pipefail

ENV_PATH=/mnt/f/challenge/envs/fomo_v2

conda create -p ${ENV_PATH} python=3.11 -y
conda activate ${ENV_PATH}

# ── PyTorch 2.6 + CUDA 12.4 (works with RTX 6000 Ada, driver 572) ────────────
pip install torch==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124

# ── xFormers (matched to torch 2.6.0 + cu124) ────────────────────────────────
pip install xformers==0.0.29.post3 \
    --index-url https://download.pytorch.org/whl/cu124

# ── Medical imaging ───────────────────────────────────────────────────────────
pip install "monai[all]==1.5.2"
pip install nibabel SimpleITK pydicom

# ── Training framework ────────────────────────────────────────────────────────
pip install lightning==2.6.0 pytorch-lightning==2.6.0
pip install wandb

# ── DINOv3 / 3DINO dependencies ───────────────────────────────────────────────
pip install omegaconf ftfy iopath regex pandas \
            scikit-learn termcolor submitit torchmetrics fvcore

# ── Install dinov3 losses as editable package ─────────────────────────────────
pip install -e /mnt/f/challenge/dinov3

# ── Sanity check ──────────────────────────────────────────────────────────────
python - <<'EOF'
import torch, monai, xformers, lightning
print(f"torch:      {torch.__version__}")
print(f"cuda:       {torch.version.cuda}")
print(f"xformers:   {xformers.__version__}")
print(f"monai:      {monai.__version__}")
print(f"lightning:  {lightning.__version__}")
print(f"GPUs:       {torch.cuda.device_count()} × {torch.cuda.get_device_name(0)}")

from monai.utils import set_determinism
set_determinism(42)
from monai.transforms import Compose, RandFlip
Compose([RandFlip(prob=0.5), RandFlip(prob=0.5)])
print("MONAI Compose seed: OK")

from dinov3.loss import DINOLoss, iBOTPatchLoss, KoLeoLoss
from dinov3.layers.dino_head import DINOHead
print("dinov3 imports: OK")
EOF
