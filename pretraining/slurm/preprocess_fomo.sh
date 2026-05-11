#!/bin/bash

#SBATCH -J preprocess_fomo
#SBATCH -p general
#SBATCH -A r00917
#SBATCH -o %x_%j.txt
#SBATCH -e %x_%j.err
#SBATCH --mail-type=fail
#SBATCH --mail-user=sinnani@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00

set -euo pipefail

module load conda
conda activate /N/project/histo-exp/challenge/envs_fomo

CHALLENGE_DIR=/N/project/histo-exp/challenge
IN_PATH=/N/project/histopath/challenge/fomo
OUT_PATH=${CHALLENGE_DIR}/fomo_preprocessed
SCRIPT=${CHALLENGE_DIR}/preprocess_from_zips.py

mkdir -p "${OUT_PATH}"

echo "=========================================="
echo "Job:         $SLURM_JOB_ID"
echo "Node:        $SLURMD_NODENAME"
echo "CPUs:        $SLURM_CPUS_PER_TASK"
echo "Input:       $IN_PATH"
echo "Output:      $OUT_PATH"
echo "Start:       $(date)"
echo "=========================================="

python "${SCRIPT}" \
    --in_path       "${IN_PATH}" \
    --out_path      "${OUT_PATH}" \
    --num_workers   16 \

echo "=========================================="
echo "Done: $(date)"
DONE=$(find "${OUT_PATH}" -name "*.npy" | wc -l)
echo "Files saved: ${DONE}"
echo "=========================================="
