#!/bin/bash
#SBATCH --job-name=attr_long
#SBATCH --account=torch_pr_36_mren
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=04:00:00
#SBATCH --comment="preemption=yes;requeue=true"
#SBATCH --output=/scratch/wc3013/ai-attractors/slurm/%x_%j.out
#SBATCH --error=/scratch/wc3013/ai-attractors/slurm/%x_%j.err
# Attractor study phase 0: multi-minute T2V rollouts (approved 2026-10-03, ~10 H200-h budget).
# Imports third_party repos read-only; writes only under /scratch/wc3013/ai-attractors.
set -euo pipefail
module purge; module load anaconda3/2025.06
unset PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1 HF_HOME=/scratch/wc3013/.cache/huggingface TMPDIR=/scratch/wc3013/tmp/ai-attractors
mkdir -p "$TMPDIR"
R=/scratch/wc3013/ai-attractors
for M in ${MODELS}; do
  /scratch/wc3013/conda-envs/self_forcing/bin/python -u $R/long_rollout.py --model $M \
    --prompts /scratch/wc3013/third_party/Self-Forcing/prompts/MovieGenVideoBench_extended.txt \
    --n-prompts ${NPROMPTS:-8} --seeds ${SEEDS:-0 1} --latents ${LATENTS:-720} \
    --out $R/long/${TAG:-main}/$M --rundir $R/rundir ${EXTRA:-}
done
echo ALL_JOBS_DONE
