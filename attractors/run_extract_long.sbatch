#!/bin/bash
#SBATCH --job-name=attr_lobs
#SBATCH --account=torch_pr_36_mren
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --time=01:55:00
#SBATCH --output=/scratch/wc3013/ai-attractors/slurm/%x_%j.out
#SBATCH --error=/scratch/wc3013/ai-attractors/slurm/%x_%j.err
# Attractor study, phase -1: observables for existing cite-128 rollouts. Read-only on inputs.
set -euo pipefail
module purge; module load anaconda3/2025.06
unset PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1 TORCH_HOME=/scratch/wc3013/.cache/torch TMPDIR=/scratch/wc3013/tmp/ai-attractors
mkdir -p "$TMPDIR"
cd /scratch/wc3013/ai-attractors
/scratch/wc3013/conda-envs/self_forcing/bin/python -u extract_long.py \
  --videos /scratch/wc3013/ai-attractors/long/${TAG:-main} --out /scratch/wc3013/ai-attractors/lobs/${TAG:-main} --workers 4
