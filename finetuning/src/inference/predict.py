"""
Container inference entry point for FOMO26.

Called by the Apptainer container during challenge evaluation:
  python predict.py --task <N> --model <v1|v2|v3> \
      --input_dir /input --output_dir /output

Dispatches to the per-task predictor.
"""

import argparse
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task",       type=int, required=True, choices=list(range(1, 8)))
    p.add_argument("--model",      type=str, required=True, choices=["v1", "v2", "v3"])
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Pretraining backbone checkpoint (last.ckpt)")
    p.add_argument("--finetune_ckpt", type=str, default=None,
                   help="Task-specific finetuning checkpoint or linear probe .pt")
    p.add_argument("--input_dir",  type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--device",     type=str, default="cuda")
    args = p.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Dynamically import per-task predictor
    mod = importlib.import_module(f"inference.predict_task{args.task}")
    mod.predict(args)


if __name__ == "__main__":
    main()
