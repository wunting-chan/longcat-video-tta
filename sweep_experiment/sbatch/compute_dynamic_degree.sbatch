#!/bin/bash
#SBATCH --job-name=compute_dynamic_degree
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24GB
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=sweep_experiment/slurm_log/compute_dynamic_degree_%j.out
#SBATCH --error=sweep_experiment/slurm_log/compute_dynamic_degree_%j.err

# ============================================================================
# Compute per-video dynamicness (VBench-style RAFT mean optical flow) for an
# evaluation video directory. Output JSON is consumed by
# scripts/plot_dynamicness_correlation.py for the per-method correlation
# figure requested by the meeting partner (Jun 1, 2026).
#
# Required env (pass via --export=ALL):
#   VIDEOS_DIR  : eval video directory (e.g. datasets/panda_1000_480p)
#   OUTPUT_JSON : destination JSON path (typically VIDEOS_DIR/dynamic_degree.json)
#
# Optional env:
#   MAX_FRAMES  : default 28 (full standard-horizon clip). Use 76 for
#                 long-horizon analysis to match generation length.
#   TARGET_SIZE : HxW (default 256x320, both divisible by 8 as RAFT requires)
#   METHOD      : raft (default) or farneback
#
# Submit (Panda standard horizon):
#   sbatch --account=torch_pr_36_mren \
#       --export=ALL,\
# VIDEOS_DIR=/scratch/wc3013/longcat-video-tta/datasets/panda_1000_480p,\
# OUTPUT_JSON=/scratch/wc3013/longcat-video-tta/datasets/panda_1000_480p/dynamic_degree.json \
#       sweep_experiment/sbatch/compute_dynamic_degree.sbatch
#
# Submit (UCF standard horizon):
#   sbatch --account=torch_pr_36_mren \
#       --export=ALL,\
# VIDEOS_DIR=/scratch/wc3013/longcat-video-tta/datasets/ucf101_1000_480p,\
# OUTPUT_JSON=/scratch/wc3013/longcat-video-tta/datasets/ucf101_1000_480p/dynamic_degree.json \
#       sweep_experiment/sbatch/compute_dynamic_degree.sbatch
#
# Expected wall time: ~5-10 min on an H200 for 1000 videos x 28 frames at
# 256x320. RAFT-small inference at 28 frame pairs/video, batched.
# ============================================================================

set -euo pipefail
export PYTHONNOUSERSITE=1

SCRATCH_BASE="/scratch/wc3013"
PROJECT_ROOT="${SCRATCH_BASE}/longcat-video-tta"
ENV_PATH="${SCRATCH_BASE}/conda-envs/longcat"

VIDEOS_DIR="${VIDEOS_DIR:?ERROR: VIDEOS_DIR must be set (--export=ALL,VIDEOS_DIR=...)}"
OUTPUT_JSON="${OUTPUT_JSON:?ERROR: OUTPUT_JSON must be set}"
MAX_FRAMES="${MAX_FRAMES:-28}"
TARGET_SIZE="${TARGET_SIZE:-256x320}"
METHOD="${METHOD:-raft}"

export TMPDIR="${SCRATCH_BASE}/tmp"
export PIP_CACHE_DIR="${SCRATCH_BASE}/pip-cache"
export HF_HOME="${HF_HOME:-${SCRATCH_BASE}/hf-cache}"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TORCH_HOME="${TORCH_HOME:-${SCRATCH_BASE}/torch-hub-cache}"
mkdir -p "${TMPDIR}" "${PIP_CACHE_DIR}" "${HF_HOME}" "${HF_HUB_CACHE}" \
    "${TORCH_HOME}" "${PROJECT_ROOT}/sweep_experiment/slurm_log"

echo "=============================================================================="
echo "Compute per-video dynamicness (RAFT mean optical flow)"
echo "=============================================================================="
echo "  Videos dir : ${VIDEOS_DIR}"
echo "  Output     : ${OUTPUT_JSON}"
echo "  Max frames : ${MAX_FRAMES}"
echo "  Target size: ${TARGET_SIZE}"
echo "  Method     : ${METHOD}"
echo "  TORCH_HOME : ${TORCH_HOME}"
echo "  Job ID     : ${SLURM_JOB_ID:-(not slurm)}"
echo "=============================================================================="

if [ ! -d "${VIDEOS_DIR}" ]; then
    echo "ERROR: ${VIDEOS_DIR} does not exist."
    exit 2
fi

module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate "${ENV_PATH}"
unset PYTHONHOME
unset PYTHONPATH
export PATH="${ENV_PATH}/bin:${PATH}"
PYTHON="${ENV_PATH}/bin/python"

cd "${PROJECT_ROOT}"

"${PYTHON}" scripts/compute_dynamic_degree.py \
    --videos-dir "${VIDEOS_DIR}" \
    --output-json "${OUTPUT_JSON}" \
    --max-frames "${MAX_FRAMES}" \
    --target-size "${TARGET_SIZE}" \
    --method "${METHOD}"

echo ""
echo "=============================================================================="
echo "Done. Dynamicness JSON: ${OUTPUT_JSON}"
echo ""
echo "Next: produce the correlation figure, e.g."
echo ""
echo "  python scripts/plot_dynamicness_correlation.py \\"
echo "      --series-root sweep_experiment/results/panda_1000v_standard \\"
echo "      --methods NOTTA ADA LORA_R8_TTA \\"
echo "      --extra-method-root delta_experiment/results/tinylora_panda_1000v_standard \\"
echo "      --extra-methods TL_BARE_R2 \\"
echo "      --dynamic-degree-json ${OUTPUT_JSON} \\"
echo "      --output-png sweep_experiment/reports/figures/dynamicness_panda_1000v_std.png \\"
echo "      --title 'Panda-70M N=999 standard horizon'"
echo "=============================================================================="
