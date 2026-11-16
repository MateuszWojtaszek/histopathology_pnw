#!/bin/bash
# TTA uncertainty of the saved students of an AL run, one seed per array task (inference only).
# Submit from the repository root on the WCSS login node:
#   sbatch -o "$PD/mateusz/logs/%x-%A_%a.log" slurm/monuseg_tta_scores.sbatch
#   MAX_IMAGES=3 ROUNDS=4 sbatch --array=0 --time=00:15:00 -o "$PD/mateusz/logs/%x-%A_%a.log" slurm/monuseg_tta_scores.sbatch
# Reads $PNW_OUTPUTS_DIR/al_<selector>/<mode>/seed<k>/round_XX/models/student and writes
# tta_nuclei.csv and tta_images.csv next to it. Finished rounds are skipped when resubmitted.
#SBATCH --job-name=monuseg-tta
#SBATCH --account=hpc-martinta-1783643755
#SBATCH --partition=lem-gpu-short
#SBATCH --gres=gpu:hopper:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-5
#SBATCH --output=%x-%A_%a.log

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

export PYTHONUNBUFFERED=1  # write the log as it goes, so `tail -f` shows progress
SELECTOR="${SELECTOR:-oracle_peek}"
MODE="${MODE:-full}"
EXTRA=()
if [ -n "${MAX_IMAGES:-}" ]; then EXTRA+=(--max-images "$MAX_IMAGES"); fi
if [ -n "${ROUNDS:-}" ]; then EXTRA+=(--rounds $ROUNDS); fi

echo "job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID on $(hostname), selector $SELECTOR, mode $MODE, commit $(git rev-parse --short HEAD)"
# --no-sync: compute nodes may have no network; the environment is synced on the login node
uv run --no-sync python -m histopathology_pnw.uncertainty.pool_scores \
    --selector "$SELECTOR" \
    --mode "$MODE" \
    --seed "$SLURM_ARRAY_TASK_ID" \
    ${EXTRA[@]+"${EXTRA[@]}"}
