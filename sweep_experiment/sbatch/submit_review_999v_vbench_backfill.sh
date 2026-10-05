#!/usr/bin/env bash
# VBench backfill for Phase A + B review runs @ 999v Panda.
#
# Inline COMPUTE_VBENCH=1 during sweep writes 3 dims (subject/background/aesthetic)
# into chunk summary.json. The other 4 dims need the vbench-backfill conda env
# (numpy 1.x + cv2) and write per-chunk JSON under chunk_*/vbench_results/.
#
# Prerequisites:
#   - Mp4s saved: chunk_*/videos/*.mp4  (NO_SAVE_VIDEOS=0 sweeps)
#   - bash scripts/setup_vbench_backfill_env.sh  (once; cv2 must import)
#
# Submit:
#   bash sweep_experiment/sbatch/submit_review_999v_vbench_backfill.sh
#
# After jobs finish (~30–90 min each):
#   bash scripts/run_review_post_vbench_backfill.sh
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/${USER}/longcat-video-tta}"
ACCOUNT="${ACCOUNT:-torch_pr_36_mren}"
SBATCH_SCRIPT="${SBATCH_SCRIPT:-sweep_experiment/sbatch/run_vbench_backfill.sbatch}"
DRY_RUN="${DRY_RUN:-0}"
ONLY_RUNS="${ONLY_RUNS:-}"

# Default backfill dims (same as run_vbench_backfill.sbatch).
MISSING_DIMS="${MISSING_DIMS:-motion_smoothness dynamic_degree imaging_quality temporal_flickering}"

REVIEW_RUNS=(
    "sweep_experiment/results/panda_1000v_standard/LORA_R1_TTA"
    "sweep_experiment/results/panda_1000v_adasteer_budget_vbench/S2_LR1e3"
    "sweep_experiment/results/panda_1000v_adasteer_budget_vbench/S10_LR5e3"
    "sweep_experiment/results/panda_1000v_adasteer_budget_vbench/S5_LR1e3"
)

_in_filter() {
    local needle="$1"
    [ -z "${ONLY_RUNS}" ] && return 0
    for m in ${ONLY_RUNS}; do
        [ "${m}" = "${needle}" ] && return 0
    done
    return 1
}

_exec() {
    if [ "${DRY_RUN}" = "1" ]; then
        echo "[DRY] $*"
    else
        "$@"
    fi
}

cd "${PROJECT_ROOT}"
mkdir -p sweep_experiment/slurm_log

count=0
skipped=0
job_ids=()

for rel in "${REVIEW_RUNS[@]}"; do
    run_id="$(basename "${rel}")"
    _in_filter "${run_id}" || continue

    METHOD_DIR="${PROJECT_ROOT}/${rel}"
    if [ ! -d "${METHOD_DIR}" ]; then
        echo "WARN: missing ${METHOD_DIR}" >&2
        skipped=$((skipped + 1))
        continue
    fi

    n_mp4=$(find "${METHOD_DIR}" -path '*/videos/*.mp4' 2>/dev/null | wc -l | tr -d ' ')
    if [ "${n_mp4}" = "0" ]; then
        echo "WARN: no mp4s under ${METHOD_DIR} — skip" >&2
        skipped=$((skipped + 1))
        continue
    fi

    # Skip if all chunks already have imaging_quality JSON (proxy for full backfill).
    n_chunks=$(find "${METHOD_DIR}" -mindepth 1 -maxdepth 1 -type d -name 'chunk_*' 2>/dev/null | wc -l | tr -d ' ')
    n_iq=$(find "${METHOD_DIR}" -path '*/vbench_results/vbench_imaging_quality_eval_results.json' 2>/dev/null | wc -l | tr -d ' ')
    if [ "${n_chunks}" -gt 0 ] && [ "${n_iq}" -ge "${n_chunks}" ]; then
        echo "skip (backfill complete): ${run_id}"
        skipped=$((skipped + 1))
        continue
    fi

    job_name="vb_rev_${run_id}"
    job_name="${job_name:0:30}"
  if [ "${DRY_RUN}" = "1" ]; then
      echo "[DRY] sbatch --account=${ACCOUNT} --job-name=${job_name} METHOD_DIR=${METHOD_DIR}"
  else
      jid=$(_exec sbatch --parsable \
          --account="${ACCOUNT}" \
          --job-name="${job_name}" \
          --export="ALL,METHOD_DIR=${METHOD_DIR},DIMS=${MISSING_DIMS},PROJECT_ROOT=${PROJECT_ROOT}" \
          "${SBATCH_SCRIPT}")
      echo "  submitted ${run_id}: job ${jid}  (${n_mp4} mp4s)"
      job_ids+=("${jid}")
  fi
    count=$((count + 1))
done

echo ""
echo "Submitted ${count} VBench backfill jobs (${skipped} skipped)."
if [ ${#job_ids[@]} -gt 0 ]; then
    echo "Job IDs: ${job_ids[*]}"
    echo ""
    echo "Monitor:"
    echo "  squeue -u \$USER | grep vb_rev"
    echo "  tail -f sweep_experiment/slurm_log/vbench_backfill_*.out"
fi
echo ""
echo "After completion:"
echo "  DATE_TAG=2026-07-03 bash scripts/run_review_post_vbench_backfill.sh"
