#!/bin/bash
#SBATCH --job-name=attr_drift
#SBATCH --account=torch_pr_36_mren
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/wc3013/video-attractors-runs/slurm/%x_%j.out
#SBATCH --error=/scratch/wc3013/video-attractors-runs/slurm/%x_%j.err
# Drift-correction experiment (user request 2026-10-06). Third-party repos read-only.
set -euo pipefail
module purge; module load anaconda3/2025.06
unset PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1 HF_HOME=/scratch/wc3013/.cache/huggingface TMPDIR=/scratch/wc3013/tmp/ai-attractors
mkdir -p "$TMPDIR"
R=/scratch/wc3013/video-attractors-runs
PY=/scratch/wc3013/conda-envs/self_forcing/bin/python
for ARM in ${ARMS}; do
  MODE=${ARM%%:*}; ALPHA=${ARM##*:}
  $PY -u $R/long_rollout.py --model sf \
    --prompts /scratch/wc3013/third_party/Self-Forcing/prompts/MovieGenVideoBench_extended.txt \
    --prompt-offset ${OFFSET:-8} --n-prompts ${NPROMPTS:-8} --seeds ${SEEDS:-0 1} --latents ${LATENTS:-720} \
    --drift $MODE --alpha $ALPHA --attractor $R/attractor_sf_p0-7.pt \
    --out $R/long/${TAG}/sf --rundir $R/rundir ${EXTRA:-}
done
echo ALL_ARMS_DONE
