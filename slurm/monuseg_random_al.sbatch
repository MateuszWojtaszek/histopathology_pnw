#!/bin/bash
# Random-selection AL on MoNuSeg, one seed per array task (task index = seed).
# Submit from the repository root on the WCSS login node, after `uv sync` and `uv run dvc pull`:
#   sbatch -o "$PD/mateusz/logs/%x-%A_%a.log" slurm/monuseg_random_al.sbatch
#   MODE=smoke sbatch --array=0 --time=00:30:00 -o "$PD/mateusz/logs/%x-%A_%a.log" slurm/monuseg_random_al.sbatch
# Results go to $PNW_OUTPUTS_DIR/al_random/<mode>/seed<k> (set in .env). A job killed at the time
# limit resumes from the last finished round when the same task is submitted again.
#SBATCH --job-name=monuseg-al
#SBATCH --account=hpc-martinta-1783643755
#SBATCH --partition=lem-gpu-short
#SBATCH --gres=gpu:hopper:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-5
#SBATCH --output=%x-%A_%a.log

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1  # write the log as it goes, so `tail -f` shows progress
MODE="${MODE:-full}"

echo "job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID on $(hostname), mode $MODE, commit $(git rev-parse --short HEAD)"
# --no-sync: compute nodes may have no network; the environment is synced on the login node
# 80 GB of an H100 fits batch 4 without gradient checkpointing (same gradients, faster)
uv run --no-sync python -m histopathology_pnw.al.random_loop \
    --mode "$MODE" \
    --seed "$SLURM_ARRAY_TASK_ID" \
    --no-gradient-checkpointing \
    --keep-checkpoints best
