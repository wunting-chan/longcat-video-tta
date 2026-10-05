#!/bin/bash
#SBATCH --job-name=oracle_fvd
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:h200:1
#SBATCH --output=sweep_experiment/slurm_log/oracle_fvd_%j.out
#SBATCH --error=sweep_experiment/slurm_log/oracle_fvd_%j.err

# Phase-1 oracle upper-bound FVD only (no TTA re-inference).
#
# Submit:
#   cd /scratch/wc3013/longcat-video-tta
#   sbatch --account=torch_pr_36_mren sweep_experiment/sbatch/run_phase1_oracle_fvd.sbatch
#
# Or rebuild dirs only (no GPU):
#   SKIP_FVD=1 sbatch --account=torch_pr_36_mren sweep_experiment/sbatch/run_phase1_oracle_fvd.sbatch

set -euo pipefail
export PYTHONNOUSERSITE=1

SCRATCH_BASE="/scratch/wc3013"
PROJECT_ROOT="${SCRATCH_BASE}/longcat-video-tta"
GT_CACHE="${GT_CACHE:-${PROJECT_ROOT}/gt_caches/panda_1000_longcat.npz}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/sweep_experiment/reports/phase1_oracle_fvd}"
GAINS_CSV="${GAINS_CSV:-${PROJECT_ROOT}/sweep_experiment/reports/per_video_analysis/2026-06-09/per_video_gains.csv}"
SERIES_ROOT="${SERIES_ROOT:-${PROJECT_ROOT}/sweep_experiment/results/panda_1000v_standard}"

module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
CONDA_ENV="${SCRATCH_BASE}/conda-envs/longcat"
conda activate "$CONDA_ENV"
export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="${SCRATCH_BASE}/.cache/huggingface"

cd "$PROJECT_ROOT"
git pull origin main

mkdir -p sweep_experiment/slurm_log

if [[ "${SKIP_FVD:-0}" == "1" ]]; then
  python sweep_experiment/scripts/build_oracle_policy_dirs.py \
    --gains-csv "$GAINS_CSV" \
    --series-root "$SERIES_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --clean
  echo "SKIP_FVD=1: policy dirs built; FVD eval skipped."
  exit 0
fi

EXTRA_FLAGS=()
if [[ "${SKIP_BUILD:-0}" == "1" ]]; then
  EXTRA_FLAGS+=(--skip-build)
fi

python sweep_experiment/scripts/run_phase1_oracle_fvd.py \
  --gains-csv "$GAINS_CSV" \
  --series-root "$SERIES_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --gt-cache "$GT_CACHE" \
  "${EXTRA_FLAGS[@]}"

echo ""
echo "Done. Inspect:"
echo "  for p in always_notta always_ada always_lora oracle_best_psnr oracle_skip_ada_nonpos oracle_skip_both_nonpos oracle_top50_ada_dpsnr; do"
echo "    jq -r '.fvd' ${OUTPUT_ROOT}/\$p/fvd.json"
echo "  done"
