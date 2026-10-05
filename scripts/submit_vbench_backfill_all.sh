#!/bin/bash
# =============================================================================
# Submit one VBench backfill sbatch per method dir that needs it.
#
# Reads the discovery TSV (or JSON) and submits one job per "needs_backfill"
# row, with concurrency limit so we don't drown the queue.
#
# Run after:
#   1. bash scripts/setup_vbench_backfill_env.sh   (one-time env + cache setup)
#   2. python3 scripts/discover_vbench_backfill_targets.py \
#        --output sweep_experiment/reports/vbench_backfill_targets.tsv \
#        --only-needs-backfill
#
# Then:
#   bash scripts/submit_vbench_backfill_all.sh
#
# Optional env:
#   TARGETS_FILE : TSV path (default: sweep_experiment/reports/vbench_backfill_targets.tsv)
#   ACCOUNT      : SLURM account (default: torch_pr_36_mren)
#   PARTITION    : leave unset to use sbatch script default
#   MAX_PARALLEL : max concurrent submitted jobs (default: 8)
#   DRY_RUN      : 1 to print sbatch commands without submitting (default: 0)
# =============================================================================

set -euo pipefail

TARGETS_FILE="${TARGETS_FILE:-sweep_experiment/reports/vbench_backfill_targets.tsv}"
ACCOUNT="${ACCOUNT:-torch_pr_36_mren}"
PARTITION="${PARTITION:-}"
MAX_PARALLEL="${MAX_PARALLEL:-8}"
DRY_RUN="${DRY_RUN:-0}"
SBATCH_SCRIPT="sweep_experiment/sbatch/run_vbench_backfill.sbatch"
PROJECT_ROOT="${PROJECT_ROOT:-/scratch/${USER}/longcat-video-tta}"
VBENCH_PYTHON="${VBENCH_PYTHON:-/scratch/${USER}/conda-envs/vbench-backfill/bin/python}"

if [ ! -f "${TARGETS_FILE}" ]; then
    echo "ERROR: ${TARGETS_FILE} not found." >&2
    echo "Run discovery first:" >&2
    echo "  python3 scripts/discover_vbench_backfill_targets.py \\" >&2
    echo "    --output ${TARGETS_FILE} --only-needs-backfill" >&2
    exit 1
fi

if [ ! -f "${SBATCH_SCRIPT}" ]; then
    echo "ERROR: ${SBATCH_SCRIPT} not found." >&2
    exit 1
fi

echo "=============================================================================="
echo "VBench backfill mass-submission"
echo "=============================================================================="
echo "  Targets file : ${TARGETS_FILE}"
echo "  Project root : ${PROJECT_ROOT}"
echo "  Account      : ${ACCOUNT}"
echo "  Partition    : ${PARTITION:-<sbatch default — do not pass h200_cour>}"
echo "  Max parallel : ${MAX_PARALLEL}"
echo "  Dry run      : ${DRY_RUN}"
echo "=============================================================================="
echo ""

# Skip header line
n_total=0
n_submitted=0
n_skipped=0
job_ids=()

# Read TSV: method_dir  n_chunks  n_chunks_with_videos  total_videos  existing_dims_all  missing_dims  needs_backfill
while IFS=$'\t' read -r method_dir n_chunks n_with_videos total_videos existing missing needs; do
    # Skip header
    if [ "${method_dir}" = "method_dir" ]; then continue; fi
    n_total=$((n_total + 1))

    if [ "${needs}" != "True" ]; then
        n_skipped=$((n_skipped + 1))
        continue
    fi

    # missing comes back as comma-separated; convert to space-separated for sbatch env
    dims_space=$(echo "${missing}" | tr ',' ' ')

    job_name="vb_$(basename "$(dirname "${method_dir}")")_$(basename "${method_dir}")"
    job_name="${job_name:0:30}"  # SLURM 30-char job name limit

    # Throttle on max parallel submitted
    while true; do
        # Count currently active jobs that match our naming
        n_active=$(squeue -u "${USER}" --noheader -o "%j" 2>/dev/null | grep -c '^vb_' || true)
        if [ "${n_active}" -lt "${MAX_PARALLEL}" ]; then
            break
        fi
        echo "  [throttle] ${n_active}/${MAX_PARALLEL} VBench jobs active; sleeping 60s ..."
        sleep 60
    done

  method_dir_abs="${method_dir}"
  if [[ "${method_dir_abs}" != /* ]]; then
      method_dir_abs="${PROJECT_ROOT}/${method_dir_abs}"
  fi

    cmd=(sbatch
        --chdir="${PROJECT_ROOT}"
        --account="${ACCOUNT}"
        --job-name="${job_name}"
        --export="ALL,METHOD_DIR=${method_dir_abs},DIMS=${dims_space},VBENCH_PYTHON=${VBENCH_PYTHON}"
    )
    if [ -n "${PARTITION}" ]; then
        cmd+=(--partition="${PARTITION}")
    fi
    cmd+=("${SBATCH_SCRIPT}")

    echo "  [submit] ${method_dir}"
    echo "           dims_to_backfill=${dims_space}"
    echo "           total_videos=${total_videos}"
    if [ "${DRY_RUN}" = "1" ]; then
        echo "           DRY-RUN: ${cmd[*]}"
    else
        out=$("${cmd[@]}" 2>&1) || {
            echo "  [error] sbatch failed for ${method_dir}: ${out}" >&2
            continue
        }
        # capture jobid
        jid=$(echo "${out}" | grep -oP 'Submitted batch job \K[0-9]+' || echo "")
        if [ -n "${jid}" ]; then
            echo "           job=${jid}"
            job_ids+=("${jid}")
        else
            echo "           ${out}"
        fi
    fi
    n_submitted=$((n_submitted + 1))
    echo ""

done < "${TARGETS_FILE}"

echo "=============================================================================="
echo "Mass-submission summary"
echo "=============================================================================="
echo "  Rows processed : ${n_total}"
echo "  Submitted      : ${n_submitted}"
echo "  Skipped        : ${n_skipped}"
if [ ${#job_ids[@]} -gt 0 ]; then
    echo "  Job IDs        : ${job_ids[*]}"
fi
echo ""
echo "Monitor:"
echo "  squeue -u \$USER --format='%.10i %.30j %.10P %.2t %.10M %R' | grep '^.*vb_'"
echo ""
echo "After all complete, fold dims into merged_summary.json for each method dir:"
echo "  for d in \$(cut -f1 ${TARGETS_FILE} | tail -n +2); do"
echo "    python3 scripts/update_merged_with_vbench.py --method-dir \"\$d\""
echo "  done"
echo "=============================================================================="
