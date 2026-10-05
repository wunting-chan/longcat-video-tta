#!/bin/bash
#SBATCH --job-name=precompute_pool_emb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24GB
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=delta_experiment/slurm_log/precompute_pool_emb_%j.out
#SBATCH --error=delta_experiment/slurm_log/precompute_pool_emb_%j.err

# ============================================================================
# Pre-compute caption embeddings for a retrieval pool.
#
# Produces:
#   <POOL_DIR>/caption_embeddings.npy   (N x D L2-normalised float32)
#   <POOL_DIR>/caption_embeddings.json  (sidecar: model, dim, count)
#
# delta_experiment/scripts/common.py::build_retrieval_pool (post v3 deploy)
# loads this cache instead of re-encoding on every TTA job.
#
# Required env (pass via --export=ALL):
#   POOL_DIR   : full path to the pool directory (must contain metadata.csv)
#
# Optional env:
#   MODEL      : sentence-transformers model name
#                (default all-MiniLM-L6-v2; matches build_retrieval_pool's
#                default so cached embeddings are picked up automatically)
#   BATCH_SIZE : encoding batch size (default 256)
#
# Submit:
#   sbatch --account=torch_pr_36_mren \
#       --export=ALL,POOL_DIR=/scratch/wc3013/longcat-video-tta/datasets/ucf101_pool_max \
#       delta_experiment/sbatch/precompute_pool_embeddings.sbatch
#
# Wall-time: ~30 min for 30K captions on a single H200 (most of the time
# is HF model download + sentence-transformers init; encoding itself is
# ~1-2 minutes).
# ============================================================================

set -euo pipefail
export PYTHONNOUSERSITE=1

SCRATCH_BASE="/scratch/wc3013"
PROJECT_ROOT="${SCRATCH_BASE}/longcat-video-tta"
ENV_PATH="${SCRATCH_BASE}/conda-envs/longcat"

POOL_DIR="${POOL_DIR:?ERROR: POOL_DIR must be set (--export=ALL,POOL_DIR=...)}"
MODEL="${MODEL:-all-MiniLM-L6-v2}"
BATCH_SIZE="${BATCH_SIZE:-256}"

export TMPDIR="${SCRATCH_BASE}/tmp"
export PIP_CACHE_DIR="${SCRATCH_BASE}/pip-cache"
export HF_HOME="${HF_HOME:-${SCRATCH_BASE}/hf-cache}"
export HF_HUB_CACHE="${HF_HOME}/hub"
export SENTENCE_TRANSFORMERS_HOME="${HF_HOME}/sentence-transformers"
mkdir -p "${TMPDIR}" "${PIP_CACHE_DIR}" "${HF_HOME}" \
    "${HF_HUB_CACHE}" "${SENTENCE_TRANSFORMERS_HOME}" \
    "${PROJECT_ROOT}/delta_experiment/slurm_log"

echo "=============================================================================="
echo "Pre-compute pool caption embeddings"
echo "=============================================================================="
echo "  Pool dir   : ${POOL_DIR}"
echo "  Model      : ${MODEL}"
echo "  Batch size : ${BATCH_SIZE}"
echo "  HF_HOME    : ${HF_HOME}"
echo "  Job ID     : ${SLURM_JOB_ID:-(not slurm)}"
echo "=============================================================================="

if [ ! -f "${POOL_DIR}/metadata.csv" ]; then
    echo "ERROR: ${POOL_DIR}/metadata.csv does not exist."
    echo "       Has the pool been built yet? See e.g."
    echo "         sbatch --account=torch_pr_36_mren \\"
    echo "             datasets/build_ucf101_retrieval_pool.sbatch"
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

# Sanity-check sentence-transformers presence.
"${PYTHON}" -c "import sentence_transformers; \
    print('sentence_transformers', sentence_transformers.__version__)"

cd "${PROJECT_ROOT}"

"${PYTHON}" scripts/precompute_pool_embeddings.py \
    --pool-dir "${POOL_DIR}" \
    --model "${MODEL}" \
    --batch-size "${BATCH_SIZE}"

echo ""
echo "=============================================================================="
echo "Pre-compute job finished"
echo "  Cache   : ${POOL_DIR}/caption_embeddings.npy"
echo "  Sidecar : ${POOL_DIR}/caption_embeddings.json"
echo "=============================================================================="
echo ""
echo "Next step: submit the retrieval-augmented TTA sweep that points at"
echo "this pool via the ``retrieval_pool_dir`` config field. The TTA"
echo "build_retrieval_pool function will auto-detect this cache and skip"
echo "the on-the-fly encode step."
echo "=============================================================================="
