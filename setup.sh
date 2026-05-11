#!/usr/bin/env bash
# Clone external dependency repos into the challenge directory.
# Run once after cloning this repository.
#
# Usage:
#   bash setup.sh [CHALLENGE_DIR]
#
# CHALLENGE_DIR defaults to the directory containing this script.

set -euo pipefail

CHALLENGE="${1:-$(cd "$(dirname "$0")" && pwd)}"
cd "$CHALLENGE"

echo "==> Setting up FOMO26 dependencies in: $CHALLENGE"

# ── 3DINO: 3D DINOv2 ViT-L architecture ───────────────────────────────────────
if [ ! -d "3DINO/.git" ]; then
    echo "--> Cloning 3DINO..."
    git clone https://github.com/AICONSlab/3DINO.git 3DINO
else
    echo "    3DINO already present"
fi

# ── BrainIAC: checkpoints + downstream reference ───────────────────────────────
if [ ! -d "BrainIAC/.git" ]; then
    echo "--> Cloning BrainIAC..."
    git clone https://github.com/AIM-KannLab/BrainIAC.git BrainIAC
    echo "    NOTE: download BrainIAC checkpoints separately into BrainIAC/src/checkpoints/"
    echo "    (brainiac.ckpt is the one used for v1 warm-start)"
else
    echo "    BrainIAC already present"
fi

# ── dinov3: DINOv3 losses, heads, RoPE layers ──────────────────────────────────
if [ ! -d "dinov3/.git" ]; then
    echo "--> Cloning dinov3..."
    git clone https://github.com/facebookresearch/dinov3.git dinov3
else
    echo "    dinov3 already present"
fi

echo ""
echo "==> External repos ready."
echo ""
echo "Next steps:"
echo "  1. Install pip package:  pip install -e dinov3/"
echo "  2. Activate env:         conda activate envs_fomo_v2  (or envs_fomo for v1)"
echo "  3. Export CHALLENGE:     export CHALLENGE=$CHALLENGE"
echo "  4. For v1/v3:            export PYTHONPATH=\$CHALLENGE/pretraining:\$CHALLENGE/dinov3:\$PYTHONPATH"
echo "  5. For v2:               export PYTHONPATH=\$CHALLENGE/pretraining:\$CHALLENGE/dinov3:\$CHALLENGE/3DINO:\$PYTHONPATH"
