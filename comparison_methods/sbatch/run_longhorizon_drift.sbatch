#!/bin/bash
#SBATCH --job-name=lh_drift
# Single H200 (same rationale as run_savi_dno_longcat.sbatch: naive multi-GPU
# split idles GPUs and trips Torch's <60% util auto-cancel). One AR rollout =
# num_chunks sequential 50-step CFG samples of a few latent frames each; a
# handful of videos fits well inside the wall clock.
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256GB
#SBATCH --time=08:00:00
#SBATCH --account=torch_pr_36_mren
#SBATCH --requeue
#SBATCH --comment="preemption=yes;requeue=true"
#SBATCH --output=comparison_methods/slurm_log/lh_drift_%j.out
#SBATCH --error=comparison_methods/slurm_log/lh_drift_%j.err

set -euo pipefail
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "=============================================================================="
echo "Long-horizon DRIFT rollout (TRUE autoregressive) on LongCat"
echo "Job ID: $SLURM_JOB_ID   Node: $SLURM_NODELIST   Start: $(date)"
echo "  cond=${DRIFT_NUM_COND:-13} chunk_gen=${DRIFT_CHUNK_GEN:-17} num_chunks=${DRIFT_NUM_CHUNKS:-8}"
echo "  steps=${DRIFT_STEPS:-50} guidance=${DRIFT_GUIDANCE:-4.0} N=${DRIFT_MAX_VIDEOS:-8}"
echo "=============================================================================="

SCRATCH_BASE="/scratch/wc3013"
PROJECT_ROOT="${SCRATCH_BASE}/longcat-video-tta"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${SCRATCH_BASE}/longcat-video-checkpoints}"
DATA_DIR="${DRIFT_DATA_DIR:-${PROJECT_ROOT}/datasets/panda_ood_budget_1000v_preview_480p}"
OUTPUT_DIR="${DRIFT_OUTPUT_DIR:-${PROJECT_ROOT}/comparison_methods/results/longhorizon_drift_c${DRIFT_NUM_CHUNKS:-8}x${DRIFT_CHUNK_GEN:-17}}"

module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
CONDA_ENV="${SCRATCH_BASE}/conda-envs/longcat"
if [ -d "$CONDA_ENV" ]; then
    conda activate "$CONDA_ENV"
    echo "Activated conda: $CONDA_ENV"
else
    echo "ERROR: Conda environment not found at $CONDA_ENV" >&2
    exit 1
fi
unset PYTHONHOME
unset PYTHONPATH
PYTHON="${CONDA_ENV}/bin/python"
if [ ! -x "$PYTHON" ]; then
    echo "ERROR: Python not found at $PYTHON" >&2
    exit 1
fi
echo "Using: $PYTHON ($($PYTHON --version 2>&1))"
"$PYTHON" -c "import torch; print(f'torch {torch.__version__}, CUDA {torch.cuda.is_available()}')"

cd "${PROJECT_ROOT}"

EVAL_GT_FLAG=""
if [ "${DRIFT_EVAL_GT:-1}" = "1" ]; then
    EVAL_GT_FLAG="--eval-gt"
fi
SAVE_FLAG=""
if [ "${DRIFT_SAVE_VIDEOS:-1}" = "1" ]; then
    SAVE_FLAG="--save-videos"
fi
CFG_FLAG=""
if [ "${DRIFT_NO_CFG:-0}" = "1" ]; then
    CFG_FLAG="--no-cfg"
fi
FIXED_CAP_FLAG=""
if [ -n "${DRIFT_FIXED_CAPTION:-}" ]; then
    FIXED_CAP_FLAG="--fixed-caption ${DRIFT_FIXED_CAPTION}"
fi

"$PYTHON" comparison_methods/scripts/longhorizon_drift_rollout.py \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --max-videos "${DRIFT_MAX_VIDEOS:-8}" \
    --start-video-idx "${DRIFT_START_IDX:-0}" \
    --chunk-size "${DRIFT_CHUNK_SIZE:-0}" \
    --num-cond-frames "${DRIFT_NUM_COND:-13}" \
    --chunk-gen "${DRIFT_CHUNK_GEN:-17}" \
    --num-chunks "${DRIFT_NUM_CHUNKS:-8}" \
    --num-inference-steps "${DRIFT_STEPS:-50}" \
    --guidance-scale "${DRIFT_GUIDANCE:-4.0}" \
    --resolution "${DRIFT_RESOLUTION:-480p}" \
    --seed "${DRIFT_SEED:-42}" \
    ${EVAL_GT_FLAG} ${SAVE_FLAG} ${CFG_FLAG} ${FIXED_CAP_FLAG}

echo ""
echo "Rollout done. Plot the drift curves (login node is fine):"
echo "  \"$PYTHON\" comparison_methods/scripts/plot_drift_curves.py \\"
echo "     --stats ${OUTPUT_DIR}/drift_stats.json \\"
echo "     --out ${OUTPUT_DIR}/drift_curves.png \\"
echo "     --md-out ${OUTPUT_DIR}/drift_slopes.md"
echo "=============================================================================="
echo "End: $(date)"
