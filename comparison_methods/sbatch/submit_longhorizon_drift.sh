#!/bin/bash
# ============================================================================
# Long-horizon DRIFT diagnostic launcher.
#
# Answers ONE question: does LongCat's quality DRIFT under TRUE autoregressive
# rollout (feeding its own frames back as conditioning)? This is the regime the
# long-video literature (Rolling Forcing / BAgger / Self-Forcing / TTC) gets its
# headroom from -- and the regime our single-shot pipeline never enters.
#
# Pilot (default): N=8 videos, 8 chunks x 17 gen frames = 136-frame rollout
# (~9s @ 15fps), CFG on, 50 steps. Cheap (~8 x 8 = 64 samples). If the drift
# slopes are significant -> headroom found. If flat -> LongCat too strong here;
# harden further (more chunks / weaker base / OOD) or switch base model.
#
# Usage:
#   bash comparison_methods/sbatch/submit_longhorizon_drift.sh
#   # scale the horizon:
#   DRIFT_NUM_CHUNKS=16 DRIFT_MAX_VIDEOS=32 bash comparison_methods/sbatch/submit_longhorizon_drift.sh
# ============================================================================
set -euo pipefail

SCRATCH_BASE="/scratch/${USER}"
PROJECT_ROOT="${PROJECT_ROOT:-${SCRATCH_BASE}/longcat-video-tta}"
SBATCH="${PROJECT_ROOT}/comparison_methods/sbatch/run_longhorizon_drift.sbatch"
ACCOUNT="${ACCOUNT:-torch_pr_36_mren}"

DATA_DIR="${DRIFT_DATA_DIR:-${PROJECT_ROOT}/datasets/panda_ood_budget_1000v_preview_480p}"
N="${DRIFT_MAX_VIDEOS:-8}"
NUM_CHUNKS="${DRIFT_NUM_CHUNKS:-8}"
CHUNK_GEN="${DRIFT_CHUNK_GEN:-17}"
NUM_COND="${DRIFT_NUM_COND:-13}"
STEPS="${DRIFT_STEPS:-50}"
OUTPUT_DIR="${DRIFT_OUTPUT_DIR:-${PROJECT_ROOT}/comparison_methods/results/longhorizon_drift_c${NUM_CHUNKS}x${CHUNK_GEN}}"

echo "============================================================"
echo "Long-horizon DRIFT diagnostic (NOTTA baseline)"
echo "  account   : ${ACCOUNT}"
echo "  data_dir  : ${DATA_DIR}"
echo "  N         : ${N}   horizon = ${NUM_CHUNKS} chunks x ${CHUNK_GEN} = $((NUM_CHUNKS*CHUNK_GEN)) frames"
echo "  cond=${NUM_COND} steps=${STEPS} cfg=on guidance=4.0"
echo "  out       : ${OUTPUT_DIR}"
echo "============================================================"

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "DRY: sbatch --account=${ACCOUNT} --export=ALL,DRIFT_MAX_VIDEOS=${N},DRIFT_NUM_CHUNKS=${NUM_CHUNKS},DRIFT_CHUNK_GEN=${CHUNK_GEN},DRIFT_NUM_COND=${NUM_COND},DRIFT_STEPS=${STEPS},DRIFT_DATA_DIR=${DATA_DIR},DRIFT_OUTPUT_DIR=${OUTPUT_DIR} ${SBATCH}"
  exit 0
fi

jid=$(sbatch --parsable --account="${ACCOUNT}" \
    --export="ALL,DRIFT_MAX_VIDEOS=${N},DRIFT_NUM_CHUNKS=${NUM_CHUNKS},DRIFT_CHUNK_GEN=${CHUNK_GEN},DRIFT_NUM_COND=${NUM_COND},DRIFT_STEPS=${STEPS},DRIFT_DATA_DIR=${DATA_DIR},DRIFT_OUTPUT_DIR=${OUTPUT_DIR}" \
    "${SBATCH}")
echo "submitted drift rollout job ${jid}"
echo ""
echo "After it finishes, render the drift curves + slope table:"
echo "  PY=/scratch/${USER}/conda-envs/longcat/bin/python"
echo "  \$PY comparison_methods/scripts/plot_drift_curves.py \\"
echo "     --stats ${OUTPUT_DIR}/drift_stats.json \\"
echo "     --out ${OUTPUT_DIR}/drift_curves.png --md-out ${OUTPUT_DIR}/drift_slopes.md"
echo ""
echo "Then paste drift_slopes.md back here (I'll log it + decide the pivot)."
