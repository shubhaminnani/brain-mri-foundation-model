#!/bin/bash
# Run once on the cluster login node to create the environment.
# Tested with CUDA 12.1, Python 3.11.

conda create -p /N/project/histo-exp/challenge/envs_fomo python=3.11 -y
conda activate /N/project/histo-exp/challenge/envs_fomo

# Core deep learning
pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121

# Medical imaging + training framework
pip install monai==1.3.0
pip install lightning==2.3.0
pip install wandb

# DINOv3 dependencies (from dinov3/conda.yaml)
pip install omegaconf ftfy iopath regex pandas scikit-learn termcolor submitit torchmetrics

# FOMO2JOMO / baseline dependencies
pip install batchgenerators yucca

# Install dinov3 as editable package (so its losses are importable)
git clone https://github.com/facebookresearch/dinov3.git
pip install -e dinov3
ip install "setuptools==68.2.2"
# Verify key imports
python -c "
from dinov3.loss import DINOLoss, iBOTPatchLoss, KoLeoLoss
from dinov3.layers.dino_head import DINOHead
from monai.networks.nets import ViT
import lightning as L
print('All imports OK')
"
