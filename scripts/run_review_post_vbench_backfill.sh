#!/usr/bin/env bash
# Fold review-series VBench backfill into merged summaries + refresh analysis.
set -euo pipefail

REPO="${REPO:-/scratch/${USER}/longcat-video-tta}"
DATE_TAG="${DATE_TAG:-$(date +%Y-%m-%d)}"
BASE="${REPO}/sweep_experiment/reports/per_video_analysis/${DATE_TAG}"

cd "${REPO}"

REVIEW_DIRS=(
    sweep_experiment/results/panda_1000v_standard/LORA_R1_TTA
    sweep_experiment/results/panda_1000v_adasteer_budget_vbench/S2_LR1e3
    sweep_experiment/results/panda_1000v_adasteer_budget_vbench/S10_LR5e3
    sweep_experiment/results/panda_1000v_adasteer_budget_vbench/S5_LR1e3
)

echo "=== Verify backfill JSON (want 10 chunks × 4 dims each) ==="
for rel in "${REVIEW_DIRS[@]}"; do
    d="${REPO}/${rel}"
    [ -d "${d}" ] || { echo "MISSING ${rel}"; continue; }
    n_chunks=$(find "${d}" -mindepth 1 -maxdepth 1 -type d -name 'chunk_*' | wc -l | tr -d ' ')
    for dim in motion_smoothness dynamic_degree imaging_quality temporal_flickering; do
        n=$(find "${d}" -path "*/vbench_results/vbench_${dim}_eval_results.json" 2>/dev/null | wc -l | tr -d ' ')
        echo "  $(basename "$(dirname "${rel}")")/$(basename "${rel}"): ${dim} ${n}/${n_chunks}"
    done
done

echo ""
echo "=== Fold VBench into merged_summary.json ==="
for rel in "${REVIEW_DIRS[@]}"; do
    d="${REPO}/${rel}"
    [ -d "${d}" ] || continue
    python3 scripts/update_merged_with_vbench.py --method-dir "${d}"
done

echo ""
echo "=== Refresh 999v VBench agreement (7-dim ΔVBench incl. LORA_R1) ==="
DATE_TAG="${DATE_TAG}" bash scripts/run_panda_vbench_agreement.sh

echo ""
echo "=== Budget VBench oracle @ 999v (7-dim per-video) ==="
python3 scripts/analyze_adasteer_budget_vbench_oracle.py --bootstrap \
    --series-root "${REPO}/sweep_experiment/results/panda_1000v_adasteer_budget_vbench" \
    --ood-csv "${REPO}/sweep_experiment/reports/per_video_analysis/2026-06-09/diffusion_ood_scores.csv" \
    --output "${BASE}/adasteer_budget_vbench_1000v.md"

echo ""
echo "Done. Key outputs:"
echo "  ${BASE}/vbench_agreement/vbench_agreement_summary.md"
echo "  ${BASE}/vbench_agreement/per_video_vbench_gains.csv"
echo "  ${BASE}/adasteer_budget_vbench_1000v.md"
echo ""
echo "Quick 7-dim population check:"
for rel in "${REVIEW_DIRS[@]}"; do
    f="${REPO}/${rel}/merged_summary.json"
    [ -f "${f}" ] || continue
    python3 -c "
import json, sys
vb = json.load(open('${f}')).get('vbench', {})
dims = [k for k in vb if not k.endswith('_std') and not k.endswith('_per_chunk')]
print('  ${rel}:', ', '.join(f'{d}={vb[d]:.3f}' for d in sorted(dims)))
"
