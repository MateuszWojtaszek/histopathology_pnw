#!/bin/bash
# CAMELYON17: train a few random-AL students (default 4 and 10 labelled tiles, seed 0), then
# the tile-level TTA uncertainty of each student on the unlabelled training tiles and the
# validation tiles. Test tiles are never used. Both steps resume when the job is resubmitted.
# Submit from the repository root on the WCSS login node, after `uv sync` and
#   uv run dvc pull data/processed/camelyon17.dvc models/oracle/cpsam.dvc \
#       models/student_init/sam_vit_l_0b3195.pth.dvc
#   sbatch -o "$PD/barpro2920/logs/%x-%j.log" slurm/camelyon17_tile_scores.sbatch
#   SEED=1 ROUNDS="4" sbatch -o "$PD/barpro2920/logs/%x-%j.log" slurm/camelyon17_tile_scores.sbatch
# Results: $PNW_OUTPUTS_DIR/al_random_c17/seed<k>/ and $PNW_OUTPUTS_DIR/tile_scores/al_random_c17/.
#SBATCH --job-name=c17-tiles
#SBATCH --account=hpc-martinta-1783643755
#SBATCH --partition=lem-gpu-short
#SBATCH --gres=gpu:hopper:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.log

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

export PYTHONUNBUFFERED=1  # write the log as it goes, so `tail -f` shows progress
SEED="${SEED:-0}"
ROUNDS="${ROUNDS:-4 10}"

echo "job $SLURM_JOB_ID on $(hostname), seed $SEED, rounds $ROUNDS, commit $(git rev-parse --short HEAD)"
# --no-sync: compute nodes may have no network; the environment is synced on the login node
uv run --no-sync python -m histopathology_pnw.al.camelyon17_students --seed "$SEED" --rounds $ROUNDS
uv run --no-sync python -m histopathology_pnw.uncertainty.tile_scores \
    --dataset camelyon17 --seeds "$SEED" --rounds $ROUNDS
echo "done $(date '+%F %T')"
