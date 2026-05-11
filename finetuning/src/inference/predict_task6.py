"""
Inference for Task 6: Linear Probing (OvR AUROC + F1, no finetuning).
Output: <output_dir>/<subject_id>.txt  — class probabilities (space-separated)
"""

from __future__ import annotations
from pathlib import Path

import torch
import torch.nn.functional as F
import nibabel as nib
import numpy as np

from backbone.vit_wrapper import ViTWrapper, CONFIGS
from models.lit_linear_probe import LinearProbeHead
from data.preprocess.fomo1 import _znorm, _resample_to_1mm


def predict(args):
    device   = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg      = CONFIGS[args.model]
    img_size = cfg["img_size"]
    D        = cfg["embed_dim"]

    backbone = ViTWrapper(args.model, args.checkpoint, freeze=True).to(device).eval()

    # Load linear probe head
    state   = torch.load(args.finetune_ckpt, map_location=device)
    n_cls   = state["head.fc.weight"].shape[0]
    probe   = LinearProbeHead(D, n_cls).to(device)
    probe.load_state_dict({k.replace("head.", "", 1): v for k, v in state.items()
                           if k.startswith("head.")})
    probe.eval()

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    subjects   = [d for d in input_dir.iterdir() if d.is_dir()]

    print(f"[Task6] Predicting {len(subjects)} subjects...")
    with torch.no_grad():
        for subj in subjects:
            nii_files = list(subj.glob("**/*.nii.gz"))
            if not nii_files:
                continue
            arr = _znorm(_resample_to_1mm(nib.load(nii_files[0])).get_fdata(dtype=np.float32))
            t   = torch.from_numpy(arr)[None, None]
            x   = F.interpolate(t, size=(img_size,)*3, mode="trilinear",
                                  align_corners=False).to(device)

            emb   = backbone.encode(x)["cls"]           # (1, D)
            logits = probe(emb)
            probs  = torch.sigmoid(logits).squeeze().cpu().numpy()

            line = " ".join(f"{p:.6f}" for p in probs)
            (output_dir / f"{subj.name}.txt").write_text(line + "\n")

    print(f"[Task6] Done → {output_dir}")
