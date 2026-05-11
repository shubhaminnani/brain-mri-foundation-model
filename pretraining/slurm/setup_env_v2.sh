#!/bin/bash
# Creates a new conda environment: fomo26_v2
# Key upgrades vs fomo26:
#   - PyTorch 2.6.0 + CUDA 12.4 (works on A100 w/ CUDA 12.7 driver)
#   - xFormers 0.0.29.post3 → memory-efficient attention (faster, less VRAM)
#   - MONAI 1.5.2 → fixes uint32 overflow bug in Compose seed init
#   - lightning 2.6.0 (matches pytorch-lightning version)
#
# Run once on the HPC login node:
#   bash setup_env_v2.sh
# ---------------------------------------------------------------------------

ENV_PATH=/N/project/histo-exp/challenge/envs_fomo_v2

conda create -p ${ENV_PATH} python=3.11 -y
conda activate ${ENV_PATH}
#/mnt/e/challenge/doserad/fomo26_v2
# ── PyTorch 2.6 + CUDA 12.4 ──────────────────────────────────────────────────
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124


# ── xFormers (matched to torch 2.6.0 + cu124) ────────────────────────────────
pip install xformers==0.0.29.post3 --index-url https://download.pytorch.org/whl/cu124

# ── Medical imaging ───────────────────────────────────────────────────────────
pip install "monai[all]==1.5.2"
pip install nibabel SimpleITK pydicom

# ── Training framework ────────────────────────────────────────────────────────
pip install lightning==2.6.0 pytorch-lightning==2.6.0
pip install wandb

# ── DINOv3 / 3DINO dependencies ───────────────────────────────────────────────
pip install omegaconf ftfy iopath regex pandas \
            scikit-learn termcolor submitit torchmetrics fvcore

# ── Install dinov3 losses as editable package ────────────────────────────────
pip install -e dinov3

# ── Sanity check ──────────────────────────────────────────────────────────────
python - <<'EOF'
import torch, monai, xformers, lightning
print(f"torch:      {torch.__version__}")
print(f"cuda:       {torch.version.cuda}")
print(f"xformers:   {xformers.__version__}")
print(f"monai:      {monai.__version__}")
print(f"lightning:  {lightning.__version__}")
print(f"GPU avail:  {torch.cuda.is_available()}")

# Verify the MONAI Compose seed bug is fixed
import monai.utils
monai.utils.set_determinism(42)
from monai.transforms import Compose, RandFlip
c = Compose([RandFlip(prob=0.5), RandFlip(prob=0.5)])
print("MONAI Compose seed: OK (no uint32 overflow)")

from dinov3.loss import DINOLoss, iBOTPatchLoss, KoLeoLoss
from dinov3.layers.dino_head import DINOHead
print("dinov3 imports: OK")
EOF
