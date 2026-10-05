#!/usr/bin/env bash
# Cross-metric eval for deploy routers: PSNR/SSIM/LPIPS lookup + optional FVD.
#
# CPU-only (default): sbatch with --cpus-only wrapper
# With FVD: needs GPU partition + saved mp4s on budget pilot
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/${USER}/longcat-video-tta}"
ACCOUNT="${ACCOUNT:-torch_pr_36_mren}"
FEATURE_DATE="${FEATURE_DATE:-2026-07-06}"
DATE_TAG="${DATE_TAG:-2026-07-06}"
RUN_FVD="${RUN_FVD:-0}"

cd "${PROJECT_ROOT}"
mkdir -p sweep_experiment/slurm_log

if [ "$RUN_FVD" = "1" ]; then
  JID=$(sbatch --parsable \
    --account="${ACCOUNT}" \
    --gres=gpu:1 \
    --cpus-per-task=4 \
    --mem=32G \
    --time=02:00:00 \
    --job-name=router_aux_fvd \
    --output="sweep_experiment/slurm_log/deploy_router_aux_fvd_%j.out" \
    --error="sweep_experiment/slurm_log/deploy_router_aux_fvd_%j.err" \
    --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},FEATURE_DATE=${FEATURE_DATE},DATE_TAG=${DATE_TAG},RUN_FVD=1" \
    --wrap="cd ${PROJECT_ROOT} && bash scripts/run_deploy_router_aux_metrics.sh")
else
  JID=$(sbatch --parsable \
    --account="${ACCOUNT}" \
    --cpus-per-task=4 \
    --mem=16G \
    --time=00:45:00 \
    --job-name=router_aux_cpu \
    --output="sweep_experiment/slurm_log/deploy_router_aux_cpu_%j.out" \
    --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},FEATURE_DATE=${FEATURE_DATE},DATE_TAG=${DATE_TAG},RUN_FVD=0" \
    --wrap="cd ${PROJECT_ROOT} && bash scripts/run_deploy_router_aux_metrics.sh")
fi

echo "Deploy router aux metrics: ${JID} (RUN_FVD=${RUN_FVD})"
echo ""
echo "When done:"
echo "  cat sweep_experiment/reports/per_video_analysis/${DATE_TAG}/deploy_router_aux_metrics/summary.md"
