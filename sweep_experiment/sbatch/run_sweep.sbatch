#!/bin/bash
#SBATCH --job-name=tta_sweep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --mem=256G
#SBATCH --gres=gpu:h200:1
#SBATCH --account=torch_pr_36_mren
#SBATCH --output=sweep_experiment/slurm_log/%x_%j.out
#SBATCH --error=sweep_experiment/slurm_log/%x_%j.err
# Allow preemptible H200 GPUs; cancelled jobs will be requeued (per cluster IT guidance)
#SBATCH --comment="preemption=yes;requeue=true"

# ============================================================================
# Unified TTA Sweep Template
#
# This sbatch script handles all TTA methods via the METHOD env var:
#   METHOD = full | lora | temp_lora | delta_a | delta_b | delta_c | norm_tune | film
#
# All hyperparameters are passed via environment variables.
# The sweep runner (run_sweep.py) sets these before submission.
# ============================================================================

set -euo pipefail
export PYTHONNOUSERSITE=1

# ============================================================================
# Path Setup
# ============================================================================
SCRATCH_BASE="/scratch/wc3013"
PROJECT_ROOT="${SCRATCH_BASE}/longcat-video-tta"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${SCRATCH_BASE}/longcat-video-checkpoints}"
DATA_DIR="${DATA_DIR:-${SCRATCH_BASE}/longcat-video-tta/datasets/panda_100_480p}"

# Method and run identification
METHOD="${METHOD:?ERROR: METHOD env var is required (full|lora|delta_a|delta_b|delta_c|norm_tune|film|vae_decoder)}"
RUN_ID="${RUN_ID:?ERROR: RUN_ID env var is required}"
SERIES_NAME="${SERIES_NAME:-sweep}"

# Output directory: sweep_experiment/results/<series_name>/<run_id>
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/sweep_experiment/results/${SERIES_NAME}/${RUN_ID}}"

# ============================================================================
# Common hyperparameters (with sensible defaults)
# ============================================================================
NUM_COND_FRAMES="${NUM_COND_FRAMES:-2}"
NUM_FRAMES="${NUM_FRAMES:-16}"
GEN_START_FRAME="${GEN_START_FRAME:-32}"
TTA_TOTAL_FRAMES="${TTA_TOTAL_FRAMES:-${GEN_START_FRAME}}"
TTA_CONTEXT_FRAMES="${TTA_CONTEXT_FRAMES:-${NUM_COND_FRAMES}}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-4.0}"
RESOLUTION="${RESOLUTION:-480p}"
SEED="${SEED:-42}"
MAX_VIDEOS="${MAX_VIDEOS:-100}"
START_VIDEO_IDX="${START_VIDEO_IDX:-0}"
CHUNK_SIZE="${CHUNK_SIZE:-0}"

# Early stopping (shared by all methods)
ES_DISABLE="${ES_DISABLE:-}"
ES_CHECK_EVERY="${ES_CHECK_EVERY:-5}"
ES_PATIENCE="${ES_PATIENCE:-3}"
ES_ANCHOR_SIGMAS="${ES_ANCHOR_SIGMAS:-0.25,0.5,0.75}"
ES_NOISE_DRAWS="${ES_NOISE_DRAWS:-2}"
ES_STRATEGY="${ES_STRATEGY:-patience}"
ES_HOLDOUT_FRACTION="${ES_HOLDOUT_FRACTION:-0.25}"

# Anchor-loss generation gate (AdaSteer)
ANCHOR_GATE_MODE="${ANCHOR_GATE_MODE:-off}"
ANCHOR_GATE_THRESHOLD="${ANCHOR_GATE_THRESHOLD:-0.0}"
ANCHOR_GATE_SOFT_SCALE="${ANCHOR_GATE_SOFT_SCALE:-0.01}"

# Data augmentation (enabled by default for all TTA methods)
AUG_ENABLED="${AUG_ENABLED:-1}"
AUG_FLIP="${AUG_FLIP:-1}"
AUG_ROTATE_DEG="${AUG_ROTATE_DEG:-10.0}"
AUG_ROTATE_RANDOM_MIN="${AUG_ROTATE_RANDOM_MIN:-5.0}"
AUG_ROTATE_RANDOM_MAX="${AUG_ROTATE_RANDOM_MAX:-15.0}"
AUG_ROTATE_RANDOM_COUNT="${AUG_ROTATE_RANDOM_COUNT:-2}"
AUG_ROTATE_RANDOM_STEP="${AUG_ROTATE_RANDOM_STEP:-1.0}"
AUG_ROTATE_ZOOM="${AUG_ROTATE_ZOOM:-1}"
AUG_SPEED_FACTORS="${AUG_SPEED_FACTORS:-}"

# Disk-saving: skip writing generated .mp4 files (metrics still computed in-memory)
NO_SAVE_VIDEOS="${NO_SAVE_VIDEOS:-1}"

# Selective video retention: path to retain_videos.json (save only listed videos)
SAVE_ONLY_LIST="${SAVE_ONLY_LIST:-}"

# Playback fps for saved MP4s. Default 24 = byte-identical to prior behaviour
# for existing experiments. Currently honoured by the `lora` runner only; set
# to 32 for high-quality long-horizon generation jobs.
FPS="${FPS:-24}"

# Multi-step autoregressive rollout (default 1 = single-step, backward-compatible)
ROLLOUT_STEPS="${ROLLOUT_STEPS:-1}"

# Pre-cached GT features for FVD/FID (path to .npz from precompute_gt_features.py)
GT_FEATURES_CACHE="${GT_FEATURES_CACHE:-}"

# ============================================================================
# Method-specific hyperparameters (only used by their respective methods)
# ============================================================================
# Full / LoRA shared
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
NUM_STEPS="${NUM_STEPS:-20}"
WARMUP_STEPS="${WARMUP_STEPS:-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-10.0}"

# Full-model specific
OPTIMIZER="${OPTIMIZER:-adamw}"

# LoRA specific
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
TARGET_MODULES="${TARGET_MODULES:-qkv,proj}"
LORA_TARGET_BLOCKS="${LORA_TARGET_BLOCKS:-all}"
USE_BUILTIN_LORA="${USE_BUILTIN_LORA:-}"
TARGET_FFN="${TARGET_FFN:-}"
SKIP_GENERATION="${SKIP_GENERATION:-}"

# Temp-LoRA (SlowFast-VGen) specific
TEMP_LORA_STEP_LATENTS="${TEMP_LORA_STEP_LATENTS:-1}"
TEMP_LORA_CTX_LATENTS="${TEMP_LORA_CTX_LATENTS:-0}"
TEMP_LORA_WARM_START="${TEMP_LORA_WARM_START:-1}"
TEMP_LORA_UPDATE_ROLLOUT="${TEMP_LORA_UPDATE_ROLLOUT:-1}"

# Batch-level TTA (retrieval-augmented)
BATCH_VIDEOS="${BATCH_VIDEOS:-1}"
BATCH_METHOD="${BATCH_METHOD:-similarity}"
RETRIEVAL_POOL_DIR="${RETRIEVAL_POOL_DIR:-}"

# Delta A/B/C shared
DELTA_STEPS="${DELTA_STEPS:-20}"
DELTA_LR="${DELTA_LR:-1e-3}"
TTA_GRAD_ACCUM="${TTA_GRAD_ACCUM:-1}"

# Delta-A vector placement (EXP2 ablation; see 2026-08-04 literature memo).
# adaln (default, original global AdaSteer) | residual (mid-late residual band).
DELTA_PLACEMENT="${DELTA_PLACEMENT:-adaln}"
RESIDUAL_BLOCKS="${RESIDUAL_BLOCKS:-}"
INITIAL_DELTA_DIR="${INITIAL_DELTA_DIR:-}"
SAVE_DELTA_DIR="${SAVE_DELTA_DIR:-}"
ANCHOR_REG_WEIGHT="${ANCHOR_REG_WEIGHT:-0.0}"
ANCHOR_REG_SIGMAS="${ANCHOR_REG_SIGMAS:-0.25,0.5,0.75}"
ANCHOR_REG_NOISE_DRAWS="${ANCHOR_REG_NOISE_DRAWS:-1}"

# Delta-B specific
NUM_GROUPS="${NUM_GROUPS:-4}"
DELTA_TARGET="${DELTA_TARGET:-timestep}"
DELTA_DIM="${DELTA_DIM:-}"
DELTA_TARGET_BLOCKS="${DELTA_TARGET_BLOCKS:-all}"

# Delta-C specific
DELTA_MODE="${DELTA_MODE:-per_channel}"

# CLIP gate
CLIP_GATE_ENABLED="${CLIP_GATE_ENABLED:-}"
CLIP_GATE_THRESHOLD="${CLIP_GATE_THRESHOLD:-0.0}"
CLIP_GATE_BACKEND="${CLIP_GATE_BACKEND:-clip}"
CLIP_GATE_MODEL="${CLIP_GATE_MODEL:-openai/clip-vit-large-patch14}"
CLIP_GATE_SAMPLE_FRAMES="${CLIP_GATE_SAMPLE_FRAMES:-4}"
CLIP_GATE_AGGREGATION="${CLIP_GATE_AGGREGATION:-mean}"
CLIP_GATE_SAMPLING_MODE="${CLIP_GATE_SAMPLING_MODE:-full_window}"
CLIP_GATE_LATE_FRACTION="${CLIP_GATE_LATE_FRACTION:-0.4}"
CLIP_GATE_LATE_ONLY="${CLIP_GATE_LATE_ONLY:-}"
CLIP_GATE_FAIL_OPEN="${CLIP_GATE_FAIL_OPEN:-1}"
CLIP_GATE_LOG_ONLY="${CLIP_GATE_LOG_ONLY:-}"

# Caption quality guard
CAPTION_GUARD_MODE="${CAPTION_GUARD_MODE:-fail}"
CAPTION_GUARD_MIN_NONEMPTY_RATIO="${CAPTION_GUARD_MIN_NONEMPTY_RATIO:-0.95}"
CAPTION_GUARD_MIN_UNIQUE_RATIO="${CAPTION_GUARD_MIN_UNIQUE_RATIO:-0.10}"
CAPTION_GUARD_MAX_TOP1_RATIO="${CAPTION_GUARD_MAX_TOP1_RATIO:-0.50}"
CAPTION_GUARD_MAX_GENERIC_TOP1_RATIO="${CAPTION_GUARD_MAX_GENERIC_TOP1_RATIO:-0.20}"
CAPTION_GUARD_TOPK="${CAPTION_GUARD_TOPK:-5}"
FIXED_CAPTION="${FIXED_CAPTION:-}"
FEATURE_FRAME_GUARD_MODE="${FEATURE_FRAME_GUARD_MODE:-fail}"

# TTA caption ablation: when set, drop captions in the TTA training step
# (inference prompt unchanged). Off by default. Used by the
# submit_standard_1000v_noprompt.sh ablation.
TTA_DISABLE_CAPTION="${TTA_DISABLE_CAPTION:-}"

# Anchor-frame x0 consistency auxiliary loss (Modification 1 of
# sweep_experiment/reports/LITERATURE_tta_recipe_modifications_2026-06-12.md).
# Default 0.0 = byte-identical to pre-patch behaviour: the runner sees no
# --anchor-x0-weight flag at all and the loss function short-circuits.
# Honoured by the lora / delta_a runners. Set > 0 to activate the
# pred_x0 = noisy_target - sigma * pred_v MSE term against the clean
# train_latents. Used by submit_smoke_x0_loss.sh.
ANCHOR_X0_WEIGHT="${ANCHOR_X0_WEIGHT:-0.0}"

# Online FVD / FID / VBench++
COMPUTE_FVD="${COMPUTE_FVD:-0}"
COMPUTE_FID="${COMPUTE_FID:-0}"
COMPUTE_VBENCH="${COMPUTE_VBENCH:-0}"
MIN_FVD_VIDEOS="${MIN_FVD_VIDEOS:-256}"

# Norm-tuning specific
NORM_STEPS="${NORM_STEPS:-20}"
NORM_LR="${NORM_LR:-1e-3}"
NORM_TARGET="${NORM_TARGET:-all_norm}"
ALSO_TUNE_DELTA="${ALSO_TUNE_DELTA:-}"

# FiLM adapter specific
FILM_STEPS="${FILM_STEPS:-20}"
FILM_LR="${FILM_LR:-1e-3}"
FILM_MODE="${FILM_MODE:-full}"

# ============================================================================
echo "=============================================================================="
echo "TTA Sweep: ${METHOD} / ${RUN_ID}"
echo "=============================================================================="
echo "Job ID       : ${SLURM_JOB_ID}"
echo "Node         : $(hostname)"
echo "Series       : ${SERIES_NAME}"
echo "Method       : ${METHOD}"
echo "Run ID       : ${RUN_ID}"
echo "Start time   : $(date)"
echo "=============================================================================="
echo "Common config:"
echo "  Cond frames    : ${NUM_COND_FRAMES}"
echo "  Num frames     : ${NUM_FRAMES}"
echo "  Gen start      : ${GEN_START_FRAME}"
echo "  TTA total      : ${TTA_TOTAL_FRAMES}"
echo "  TTA context    : ${TTA_CONTEXT_FRAMES}"
echo "  TTA window     : [$(( GEN_START_FRAME - TTA_TOTAL_FRAMES )), ${GEN_START_FRAME})"
echo "  Infer steps    : ${NUM_INFERENCE_STEPS}"
echo "  Guidance       : ${GUIDANCE_SCALE}"
echo "  Max videos     : ${MAX_VIDEOS}"
echo "  Data dir       : ${DATA_DIR}"
echo "  Checkpoint dir : ${CHECKPOINT_DIR}"
echo "  Output dir     : ${OUTPUT_DIR}"
echo "  Caption guard  : mode=${CAPTION_GUARD_MODE}, min_nonempty=${CAPTION_GUARD_MIN_NONEMPTY_RATIO}, min_unique=${CAPTION_GUARD_MIN_UNIQUE_RATIO}, max_top1=${CAPTION_GUARD_MAX_TOP1_RATIO}"
echo "  Fixed caption  : ${FIXED_CAPTION:-<none>}"
echo "  TTA no-caption : ${TTA_DISABLE_CAPTION:-0}"
echo "  Feature guard  : ${FEATURE_FRAME_GUARD_MODE}"
echo "  Augmentation   : enabled=${AUG_ENABLED:-0}, flip=${AUG_FLIP:-0}, rotate_deg=${AUG_ROTATE_DEG}"
echo "  Online FVD     : ${COMPUTE_FVD}  FID: ${COMPUTE_FID}  VBench: ${COMPUTE_VBENCH}  min_videos: ${MIN_FVD_VIDEOS}"
if [ -d "${PROJECT_ROOT}/.git" ]; then
  echo "  TTA repo commit: $(cd "${PROJECT_ROOT}" && git rev-parse --short HEAD)"
fi
if [ -d "${PROJECT_ROOT}/LongCat-Video/.git" ]; then
  echo "  LongCat commit : $(cd "${PROJECT_ROOT}/LongCat-Video" && git rev-parse --short HEAD)"
else
  echo "  LongCat commit : <not-a-git-repo>"
fi
python3 - <<'PY'
import platform
try:
    import torch
    torch_v = torch.__version__
except Exception:
    torch_v = "unavailable"
try:
    import transformers
    hf_v = transformers.__version__
except Exception:
    hf_v = "unavailable"
print(f"  Python         : {platform.python_version()}")
print(f"  Torch          : {torch_v}")
print(f"  Transformers   : {hf_v}")
PY

# ============================================================================
# Environment Setup
#
# CRITICAL: module load sets PYTHONHOME / PYTHONPATH to the system anaconda,
# which breaks the conda env's sys.path (wrong Python version, missing
# packages). We MUST unset these after conda activate. See:
#   sweep_experiment/docs/SBATCH_CONDA_BEST_PRACTICES.md
# ============================================================================
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

# CRITICAL: unset PYTHONHOME and PYTHONPATH set by "module load anaconda3/..."
# Without this, even explicit $PYTHON uses the system anaconda's sys.path,
# causing "No module named 'torch'" and similar import errors.
unset PYTHONHOME
unset PYTHONPATH

PYTHON="${CONDA_ENV}/bin/python"
if [ ! -x "$PYTHON" ]; then
    echo "ERROR: Python not found at $PYTHON" >&2
    exit 1
fi
echo "Using: $PYTHON ($($PYTHON --version 2>&1))"

# Verify the environment is correct (catches broken conda activation early)
"$PYTHON" -c "import torch; print(f'torch {torch.__version__}, CUDA {torch.cuda.is_available()}')" || {
    echo "ERROR: torch import failed — conda env is broken or PYTHONHOME was not unset" >&2
    exit 1
}

export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="${SCRATCH_BASE}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}"
export TORCH_HOME="${SCRATCH_BASE}/.cache/torch"
export VBENCH_CACHE_DIR="${SCRATCH_BASE}/.cache/vbench"
mkdir -p "$HF_HOME" "$TORCH_HOME" "$VBENCH_CACHE_DIR"

cd "$PROJECT_ROOT"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "sweep_experiment/slurm_log"

# ============================================================================
# GPU Info
# ============================================================================
nvidia-smi --query-gpu=name,memory.free --format=csv
echo ""

# ============================================================================
# Build early stopping flags
# ============================================================================
ES_FLAGS=""
if [ "${ES_DISABLE}" = "1" ]; then
    ES_FLAGS="${ES_FLAGS} --es-disable"
fi
ES_FLAGS="${ES_FLAGS} --es-check-every ${ES_CHECK_EVERY}"
ES_FLAGS="${ES_FLAGS} --es-patience ${ES_PATIENCE}"
ES_FLAGS="${ES_FLAGS} --es-anchor-sigmas ${ES_ANCHOR_SIGMAS}"
ES_FLAGS="${ES_FLAGS} --es-noise-draws ${ES_NOISE_DRAWS}"
ES_FLAGS="${ES_FLAGS} --es-strategy ${ES_STRATEGY}"
ES_FLAGS="${ES_FLAGS} --es-holdout-fraction ${ES_HOLDOUT_FRACTION}"

# Build --no-save-videos flag
SAVE_FLAGS=""
if [ "${NO_SAVE_VIDEOS}" = "1" ]; then
    SAVE_FLAGS="--no-save-videos"
fi
if [ -n "${SAVE_ONLY_LIST}" ]; then
    SAVE_FLAGS="${SAVE_FLAGS} --save-only-list ${SAVE_ONLY_LIST}"
fi
if [ "${ROLLOUT_STEPS}" != "1" ]; then
    SAVE_FLAGS="${SAVE_FLAGS} --rollout-steps ${ROLLOUT_STEPS}"
fi
if [ -n "${GT_FEATURES_CACHE}" ]; then
    SAVE_FLAGS="${SAVE_FLAGS} --gt-features-cache ${GT_FEATURES_CACHE}"
fi

# Build CLIP gate flags (shared across methods)
CLIP_FLAGS=""
if [ "${CLIP_GATE_ENABLED}" = "1" ]; then
    CLIP_FLAGS="${CLIP_FLAGS} --clip-gate-enabled"
fi
CLIP_FLAGS="${CLIP_FLAGS} --clip-gate-threshold ${CLIP_GATE_THRESHOLD}"
CLIP_FLAGS="${CLIP_FLAGS} --clip-gate-backend ${CLIP_GATE_BACKEND}"
CLIP_FLAGS="${CLIP_FLAGS} --clip-gate-model ${CLIP_GATE_MODEL}"
CLIP_FLAGS="${CLIP_FLAGS} --clip-gate-sample-frames ${CLIP_GATE_SAMPLE_FRAMES}"
CLIP_FLAGS="${CLIP_FLAGS} --clip-gate-aggregation ${CLIP_GATE_AGGREGATION}"
CLIP_FLAGS="${CLIP_FLAGS} --clip-gate-sampling-mode ${CLIP_GATE_SAMPLING_MODE}"
CLIP_FLAGS="${CLIP_FLAGS} --clip-gate-late-fraction ${CLIP_GATE_LATE_FRACTION}"
if [ "${CLIP_GATE_LATE_ONLY}" = "1" ]; then
    CLIP_FLAGS="${CLIP_FLAGS} --clip-gate-late-only"
fi
if [ "${CLIP_GATE_LOG_ONLY}" = "1" ]; then
    CLIP_FLAGS="${CLIP_FLAGS} --clip-gate-log-only"
fi
if [ "${CLIP_GATE_FAIL_OPEN}" = "1" ]; then
    CLIP_FLAGS="${CLIP_FLAGS} --clip-gate-fail-open"
else
    CLIP_FLAGS="${CLIP_FLAGS} --clip-gate-fail-closed"
fi

# Build caption guard flags (shared across methods)
CAPTION_FLAGS="--caption-guard-mode ${CAPTION_GUARD_MODE}"
CAPTION_FLAGS="${CAPTION_FLAGS} --caption-guard-min-nonempty-ratio ${CAPTION_GUARD_MIN_NONEMPTY_RATIO}"
CAPTION_FLAGS="${CAPTION_FLAGS} --caption-guard-min-unique-ratio ${CAPTION_GUARD_MIN_UNIQUE_RATIO}"
CAPTION_FLAGS="${CAPTION_FLAGS} --caption-guard-max-top1-ratio ${CAPTION_GUARD_MAX_TOP1_RATIO}"
CAPTION_FLAGS="${CAPTION_FLAGS} --caption-guard-max-generic-top1-ratio ${CAPTION_GUARD_MAX_GENERIC_TOP1_RATIO}"
CAPTION_FLAGS="${CAPTION_FLAGS} --caption-guard-topk ${CAPTION_GUARD_TOPK}"
if [ -n "${FIXED_CAPTION}" ]; then
    FIXED_CAPTION_ESCAPED="$(printf '%q' "${FIXED_CAPTION}")"
    CAPTION_FLAGS="${CAPTION_FLAGS} --fixed-caption ${FIXED_CAPTION_ESCAPED}"
fi
FEATURE_FLAGS="--feature-frame-guard-mode ${FEATURE_FRAME_GUARD_MODE}"

# TTA caption-ablation flag (passed only to runners that own a TTA loop;
# norm_tune/film/delta_b/delta_c don't expose it yet, so we omit it for
# those methods to avoid argparse failures).
NOPROMPT_FLAGS=""
if [ "${TTA_DISABLE_CAPTION}" = "1" ]; then
    NOPROMPT_FLAGS="--tta-disable-caption"
fi

# Anchor-frame x0 loss flag (Modification 1). Only passed when > 0 so the
# pre-patch byte-identical guarantee holds for legacy submit wrappers that
# don't set ANCHOR_X0_WEIGHT. Honoured by the lora / delta_a branches only;
# norm_tune / film / delta_b / delta_c don't expose --anchor-x0-weight yet.
X0_FLAGS=""
case "${ANCHOR_X0_WEIGHT}" in
    0|0.0|0.00|0.000|"")
        : ;;
    *)
        X0_FLAGS="--anchor-x0-weight ${ANCHOR_X0_WEIGHT}"
        ;;
esac

EVAL_FLAGS=""
[ "${COMPUTE_FVD}" = "1" ] && EVAL_FLAGS="${EVAL_FLAGS} --compute-fvd"
[ "${COMPUTE_FID}" = "1" ] && EVAL_FLAGS="${EVAL_FLAGS} --compute-fid"
[ "${COMPUTE_VBENCH}" = "1" ] && EVAL_FLAGS="${EVAL_FLAGS} --compute-vbench"
EVAL_FLAGS="${EVAL_FLAGS} --min-fvd-videos ${MIN_FVD_VIDEOS}"

# Build augmentation flags (shared across all methods)
AUG_FLAGS=""
if [ "${AUG_ENABLED}" = "1" ]; then
    AUG_FLAGS="${AUG_FLAGS} --aug-enabled"
fi
if [ "${AUG_FLIP}" = "1" ]; then
    AUG_FLAGS="${AUG_FLAGS} --aug-flip"
fi
AUG_FLAGS="${AUG_FLAGS} --aug-rotate-deg ${AUG_ROTATE_DEG}"
AUG_FLAGS="${AUG_FLAGS} --aug-rotate-random-min ${AUG_ROTATE_RANDOM_MIN}"
AUG_FLAGS="${AUG_FLAGS} --aug-rotate-random-max ${AUG_ROTATE_RANDOM_MAX}"
AUG_FLAGS="${AUG_FLAGS} --aug-rotate-random-count ${AUG_ROTATE_RANDOM_COUNT}"
AUG_FLAGS="${AUG_FLAGS} --aug-rotate-random-step ${AUG_ROTATE_RANDOM_STEP}"
if [ "${AUG_ROTATE_ZOOM}" != "1" ]; then
    AUG_FLAGS="${AUG_FLAGS} --no-aug-rotate-zoom"
fi
if [ -n "${AUG_SPEED_FACTORS}" ]; then
    AUG_FLAGS="${AUG_FLAGS} --aug-speed-factors ${AUG_SPEED_FACTORS}"
fi

# ============================================================================
# Dispatch to the appropriate method script
# ============================================================================
case "${METHOD}" in

  full)
    echo "Method-specific config:"
    echo "  Learning rate  : ${LEARNING_RATE}"
    echo "  Num steps      : ${NUM_STEPS}"
    echo "  Warmup steps   : ${WARMUP_STEPS}"
    echo "  Batch videos   : ${BATCH_VIDEOS}"
    echo "  Retrieval pool : ${RETRIEVAL_POOL_DIR:-<same as DATA_DIR>}"
    echo "=============================================================================="

    EXTRA_FLAGS=""
    if [ "${SKIP_GENERATION}" = "1" ]; then
        EXTRA_FLAGS="${EXTRA_FLAGS} --skip-generation"
    fi

    POOL_FLAGS=""
    if [ -n "${RETRIEVAL_POOL_DIR}" ]; then
        POOL_FLAGS="--retrieval-pool-dir ${RETRIEVAL_POOL_DIR}"
    fi

    "$PYTHON" lora_experiment/scripts/run_full_tta.py \
        --checkpoint-dir "${CHECKPOINT_DIR}" \
        --data-dir "${DATA_DIR}" \
        --output-dir "${OUTPUT_DIR}" \
        --max-videos "${MAX_VIDEOS}" \
        --start-video-idx "${START_VIDEO_IDX}" \
        --chunk-size "${CHUNK_SIZE}" \
        --learning-rate "${LEARNING_RATE}" \
        --num-steps "${NUM_STEPS}" \
        --warmup-steps "${WARMUP_STEPS}" \
        --weight-decay "${WEIGHT_DECAY}" \
        --max-grad-norm "${MAX_GRAD_NORM}" \
        --optimizer "${OPTIMIZER}" \
        --batch-videos "${BATCH_VIDEOS}" \
        ${POOL_FLAGS} \
        --num-cond-frames "${NUM_COND_FRAMES}" \
        --num-frames "${NUM_FRAMES}" \
        --gen-start-frame "${GEN_START_FRAME}" \
        --tta-total-frames "${TTA_TOTAL_FRAMES}" \
        --tta-context-frames "${TTA_CONTEXT_FRAMES}" \
        --num-inference-steps "${NUM_INFERENCE_STEPS}" \
        --guidance-scale "${GUIDANCE_SCALE}" \
        --resolution "${RESOLUTION}" \
        --seed "${SEED}" \
        ${SAVE_FLAGS} \
        ${EXTRA_FLAGS} \
        ${CAPTION_FLAGS} \
        ${FEATURE_FLAGS} \
        ${CLIP_FLAGS} \
        ${EVAL_FLAGS} \
        ${ES_FLAGS} \
        ${AUG_FLAGS}
    ;;

  lora)
    echo "Method-specific config:"
    echo "  LoRA impl      : ${USE_BUILTIN_LORA:-custom}"
    echo "  LoRA rank      : ${LORA_RANK}"
    echo "  LoRA alpha     : ${LORA_ALPHA}"
    echo "  Target modules : ${TARGET_MODULES}"
    echo "  Target blocks  : ${LORA_TARGET_BLOCKS}"
    echo "  Target FFN     : ${TARGET_FFN:-no}"
    echo "  Learning rate  : ${LEARNING_RATE}"
    echo "  Num steps      : ${NUM_STEPS}"
    echo "  Warmup steps   : ${WARMUP_STEPS}"
    echo "  Batch videos   : ${BATCH_VIDEOS}"
    echo "  Retrieval pool : ${RETRIEVAL_POOL_DIR:-<same as DATA_DIR>}"
    echo "  Anchor-x0 wt   : ${ANCHOR_X0_WEIGHT}"
    echo "  Output fps     : ${FPS}"
    echo "=============================================================================="

    EXTRA_FLAGS=""
    if [ "${TARGET_FFN}" = "1" ]; then
        EXTRA_FLAGS="${EXTRA_FLAGS} --target-ffn"
    fi
    if [ "${SKIP_GENERATION}" = "1" ]; then
        EXTRA_FLAGS="${EXTRA_FLAGS} --skip-generation"
    fi
    if [ "${USE_BUILTIN_LORA}" = "1" ]; then
        EXTRA_FLAGS="${EXTRA_FLAGS} --use-builtin-lora"
    fi

    POOL_FLAGS=""
    if [ -n "${RETRIEVAL_POOL_DIR}" ]; then
        POOL_FLAGS="--retrieval-pool-dir ${RETRIEVAL_POOL_DIR}"
    fi

    "$PYTHON" lora_experiment/scripts/run_lora_tta.py \
        --checkpoint-dir "${CHECKPOINT_DIR}" \
        --data-dir "${DATA_DIR}" \
        --output-dir "${OUTPUT_DIR}" \
        --max-videos "${MAX_VIDEOS}" \
        --start-video-idx "${START_VIDEO_IDX}" \
        --chunk-size "${CHUNK_SIZE}" \
        --lora-rank "${LORA_RANK}" \
        --lora-alpha "${LORA_ALPHA}" \
        --target-modules "${TARGET_MODULES}" \
        --lora-target-blocks "${LORA_TARGET_BLOCKS}" \
        --learning-rate "${LEARNING_RATE}" \
        --num-steps "${NUM_STEPS}" \
        --warmup-steps "${WARMUP_STEPS}" \
        --weight-decay "${WEIGHT_DECAY}" \
        --max-grad-norm "${MAX_GRAD_NORM}" \
        --batch-videos "${BATCH_VIDEOS}" \
        ${POOL_FLAGS} \
        --num-cond-frames "${NUM_COND_FRAMES}" \
        --num-frames "${NUM_FRAMES}" \
        --gen-start-frame "${GEN_START_FRAME}" \
        --tta-total-frames "${TTA_TOTAL_FRAMES}" \
        --tta-context-frames "${TTA_CONTEXT_FRAMES}" \
        --num-inference-steps "${NUM_INFERENCE_STEPS}" \
        --guidance-scale "${GUIDANCE_SCALE}" \
        --resolution "${RESOLUTION}" \
        --seed "${SEED}" \
        --fps "${FPS}" \
        ${SAVE_FLAGS} \
        ${EXTRA_FLAGS} \
        ${CAPTION_FLAGS} \
        ${FEATURE_FLAGS} \
        ${CLIP_FLAGS} \
        ${EVAL_FLAGS} \
        ${ES_FLAGS} \
        ${AUG_FLAGS} \
        ${NOPROMPT_FLAGS} \
        ${X0_FLAGS}
    ;;

  temp_lora)
    # SlowFast-VGen (ICLR 2025) Temp-LoRA fast-learning baseline.
    # Note: this runner does NOT accept ES / augmentation / CLIP-gate flags,
    # so those are intentionally omitted here.
    echo "Method-specific config (Temp-LoRA / SlowFast-VGen):"
    echo "  LoRA impl      : ${USE_BUILTIN_LORA:-custom}"
    echo "  LoRA rank/alpha: ${LORA_RANK}/${LORA_ALPHA}"
    echo "  Target modules : ${TARGET_MODULES}"
    echo "  Target blocks  : ${LORA_TARGET_BLOCKS}"
    echo "  Steps/lr chunk : ${NUM_STEPS} / ${LEARNING_RATE}"
    echo "  Stream step lat: ${TEMP_LORA_STEP_LATENTS}"
    echo "  Stream ctx lat : ${TEMP_LORA_CTX_LATENTS} (0=from tta_context)"
    echo "  Warm-start ctx : ${TEMP_LORA_WARM_START}"
    echo "  Update rollout : ${TEMP_LORA_UPDATE_ROLLOUT}"
    echo "=============================================================================="

    EXTRA_FLAGS=""
    if [ "${TARGET_FFN}" = "1" ]; then
        EXTRA_FLAGS="${EXTRA_FLAGS} --target-ffn"
    fi
    if [ "${USE_BUILTIN_LORA}" = "1" ]; then
        EXTRA_FLAGS="${EXTRA_FLAGS} --use-builtin-lora"
    fi
    if [ "${TEMP_LORA_WARM_START}" = "1" ]; then
        EXTRA_FLAGS="${EXTRA_FLAGS} --warm-start-on-context"
    else
        EXTRA_FLAGS="${EXTRA_FLAGS} --no-warm-start-on-context"
    fi
    if [ "${TEMP_LORA_UPDATE_ROLLOUT}" = "1" ]; then
        EXTRA_FLAGS="${EXTRA_FLAGS} --update-during-rollout"
    else
        EXTRA_FLAGS="${EXTRA_FLAGS} --no-update-during-rollout"
    fi

    "$PYTHON" lora_experiment/scripts/run_temp_lora_tta.py \
        --checkpoint-dir "${CHECKPOINT_DIR}" \
        --data-dir "${DATA_DIR}" \
        --output-dir "${OUTPUT_DIR}" \
        --max-videos "${MAX_VIDEOS}" \
        --start-video-idx "${START_VIDEO_IDX}" \
        --chunk-size "${CHUNK_SIZE}" \
        --lora-rank "${LORA_RANK}" \
        --lora-alpha "${LORA_ALPHA}" \
        --target-modules "${TARGET_MODULES}" \
        --lora-target-blocks "${LORA_TARGET_BLOCKS}" \
        --learning-rate "${LEARNING_RATE}" \
        --num-steps "${NUM_STEPS}" \
        --warmup-steps "${WARMUP_STEPS}" \
        --weight-decay "${WEIGHT_DECAY}" \
        --max-grad-norm "${MAX_GRAD_NORM}" \
        --temp-lora-step-latents "${TEMP_LORA_STEP_LATENTS}" \
        --temp-lora-ctx-latents "${TEMP_LORA_CTX_LATENTS}" \
        --num-cond-frames "${NUM_COND_FRAMES}" \
        --num-frames "${NUM_FRAMES}" \
        --gen-start-frame "${GEN_START_FRAME}" \
        --tta-total-frames "${TTA_TOTAL_FRAMES}" \
        --tta-context-frames "${TTA_CONTEXT_FRAMES}" \
        --num-inference-steps "${NUM_INFERENCE_STEPS}" \
        --guidance-scale "${GUIDANCE_SCALE}" \
        --resolution "${RESOLUTION}" \
        --seed "${SEED}" \
        ${SAVE_FLAGS} \
        ${EXTRA_FLAGS} \
        ${CAPTION_FLAGS} \
        ${FEATURE_FLAGS} \
        ${EVAL_FLAGS} \
        ${NOPROMPT_FLAGS} \
        ${X0_FLAGS}
    ;;

  delta_a)
    echo "Method-specific config:"
    echo "  Delta LR       : ${DELTA_LR}"
    echo "  Delta steps    : ${DELTA_STEPS}"
    echo "  Grad accum     : ${TTA_GRAD_ACCUM}"
    echo "  Batch videos   : ${BATCH_VIDEOS}"
    echo "  Batch method   : ${BATCH_METHOD}"
    echo "  Retrieval pool : ${RETRIEVAL_POOL_DIR:-<same as DATA_DIR>}"
    echo "  Anchor gate    : mode=${ANCHOR_GATE_MODE}, threshold=${ANCHOR_GATE_THRESHOLD}, soft_scale=${ANCHOR_GATE_SOFT_SCALE}"
    echo "  Anchor reg     : weight=${ANCHOR_REG_WEIGHT}, sigmas=${ANCHOR_REG_SIGMAS}, draws=${ANCHOR_REG_NOISE_DRAWS}"
    echo "  Anchor-x0 wt   : ${ANCHOR_X0_WEIGHT}"
    echo "  Placement      : ${DELTA_PLACEMENT}  residual_blocks=${RESIDUAL_BLOCKS:-<auto 55-80%>}"
    echo "=============================================================================="

    POOL_FLAGS=""
    if [ -n "${RETRIEVAL_POOL_DIR}" ]; then
        POOL_FLAGS="--retrieval-pool-dir ${RETRIEVAL_POOL_DIR}"
    fi

    PLACEMENT_FLAGS="--delta-placement ${DELTA_PLACEMENT}"
    if [ -n "${RESIDUAL_BLOCKS}" ]; then
        PLACEMENT_FLAGS="${PLACEMENT_FLAGS} --residual-blocks ${RESIDUAL_BLOCKS}"
    fi

    DELTA_RESUME_FLAGS=""
    if [ -n "${INITIAL_DELTA_DIR}" ]; then
        DELTA_RESUME_FLAGS="${DELTA_RESUME_FLAGS} --initial-delta-dir ${INITIAL_DELTA_DIR}"
    fi
    if [ -n "${SAVE_DELTA_DIR}" ]; then
        DELTA_RESUME_FLAGS="${DELTA_RESUME_FLAGS} --save-delta-dir ${SAVE_DELTA_DIR}"
    fi

    "$PYTHON" delta_experiment/scripts/run_delta_a.py \
        --checkpoint-dir "${CHECKPOINT_DIR}" \
        --data-dir "${DATA_DIR}" \
        --output-dir "${OUTPUT_DIR}" \
        --max-videos "${MAX_VIDEOS}" \
        --start-video-idx "${START_VIDEO_IDX}" \
        --chunk-size "${CHUNK_SIZE}" \
        --delta-steps "${DELTA_STEPS}" \
        --delta-lr "${DELTA_LR}" \
        --tta-grad-accum "${TTA_GRAD_ACCUM}" \
        --batch-videos "${BATCH_VIDEOS}" \
        --batch-method "${BATCH_METHOD}" \
        --anchor-gate-mode "${ANCHOR_GATE_MODE}" \
        --anchor-gate-threshold "${ANCHOR_GATE_THRESHOLD}" \
        --anchor-gate-soft-scale "${ANCHOR_GATE_SOFT_SCALE}" \
        --anchor-reg-weight "${ANCHOR_REG_WEIGHT}" \
        --anchor-reg-sigmas "${ANCHOR_REG_SIGMAS}" \
        --anchor-reg-noise-draws "${ANCHOR_REG_NOISE_DRAWS}" \
        ${PLACEMENT_FLAGS} \
        ${POOL_FLAGS} \
        ${DELTA_RESUME_FLAGS} \
        --num-cond-frames "${NUM_COND_FRAMES}" \
        --num-frames "${NUM_FRAMES}" \
        --gen-start-frame "${GEN_START_FRAME}" \
        --tta-total-frames "${TTA_TOTAL_FRAMES}" \
        --tta-context-frames "${TTA_CONTEXT_FRAMES}" \
        --num-inference-steps "${NUM_INFERENCE_STEPS}" \
        --guidance-scale "${GUIDANCE_SCALE}" \
        --resolution "${RESOLUTION}" \
        --seed "${SEED}" \
        ${SAVE_FLAGS} \
        ${CAPTION_FLAGS} \
        ${FEATURE_FLAGS} \
        ${CLIP_FLAGS} \
        ${EVAL_FLAGS} \
        ${ES_FLAGS} \
        ${AUG_FLAGS} \
        ${NOPROMPT_FLAGS} \
        ${X0_FLAGS}
    ;;

  delta_b)
    echo "Method-specific config:"
    echo "  Delta LR       : ${DELTA_LR}"
    echo "  Delta steps    : ${DELTA_STEPS}"
    echo "  Num groups     : ${NUM_GROUPS}"
    echo "  Delta target   : ${DELTA_TARGET}"
    echo "  Delta dim      : ${DELTA_DIM:-full}"
    echo "  Target blocks  : ${DELTA_TARGET_BLOCKS}"
    echo "=============================================================================="

    DELTA_DIM_FLAG=""
    if [ -n "${DELTA_DIM}" ]; then
        DELTA_DIM_FLAG="--delta-dim ${DELTA_DIM}"
    fi

    "$PYTHON" delta_experiment/scripts/run_delta_b.py \
        --checkpoint-dir "${CHECKPOINT_DIR}" \
        --data-dir "${DATA_DIR}" \
        --output-dir "${OUTPUT_DIR}" \
        --max-videos "${MAX_VIDEOS}" \
        --delta-steps "${DELTA_STEPS}" \
        --delta-lr "${DELTA_LR}" \
        --num-groups "${NUM_GROUPS}" \
        --delta-target "${DELTA_TARGET}" \
        --delta-target-blocks "${DELTA_TARGET_BLOCKS}" \
        --num-cond-frames "${NUM_COND_FRAMES}" \
        --num-frames "${NUM_FRAMES}" \
        --gen-start-frame "${GEN_START_FRAME}" \
        --tta-total-frames "${TTA_TOTAL_FRAMES}" \
        --tta-context-frames "${TTA_CONTEXT_FRAMES}" \
        --num-inference-steps "${NUM_INFERENCE_STEPS}" \
        --guidance-scale "${GUIDANCE_SCALE}" \
        --resolution "${RESOLUTION}" \
        --seed "${SEED}" \
        ${DELTA_DIM_FLAG} \
        ${SAVE_FLAGS} \
        ${CAPTION_FLAGS} \
        ${FEATURE_FLAGS} \
        ${CLIP_FLAGS} \
        ${EVAL_FLAGS} \
        ${ES_FLAGS} \
        ${AUG_FLAGS}
    ;;

  delta_c)
    echo "Method-specific config:"
    echo "  Delta LR       : ${DELTA_LR}"
    echo "  Delta steps    : ${DELTA_STEPS}"
    echo "  Delta mode     : ${DELTA_MODE}"
    echo "=============================================================================="

    "$PYTHON" delta_experiment/scripts/run_delta_c.py \
        --checkpoint-dir "${CHECKPOINT_DIR}" \
        --data-dir "${DATA_DIR}" \
        --output-dir "${OUTPUT_DIR}" \
        --max-videos "${MAX_VIDEOS}" \
        --delta-steps "${DELTA_STEPS}" \
        --delta-lr "${DELTA_LR}" \
        --delta-mode "${DELTA_MODE}" \
        --num-cond-frames "${NUM_COND_FRAMES}" \
        --num-frames "${NUM_FRAMES}" \
        --gen-start-frame "${GEN_START_FRAME}" \
        --tta-total-frames "${TTA_TOTAL_FRAMES}" \
        --tta-context-frames "${TTA_CONTEXT_FRAMES}" \
        --num-inference-steps "${NUM_INFERENCE_STEPS}" \
        --guidance-scale "${GUIDANCE_SCALE}" \
        --resolution "${RESOLUTION}" \
        --seed "${SEED}" \
        ${SAVE_FLAGS} \
        ${CAPTION_FLAGS} \
        ${FEATURE_FLAGS} \
        ${CLIP_FLAGS} \
        ${EVAL_FLAGS} \
        ${ES_FLAGS} \
        ${AUG_FLAGS}
    ;;

  norm_tune)
    echo "Method-specific config:"
    echo "  Norm target    : ${NORM_TARGET}"
    echo "  Norm LR        : ${NORM_LR}"
    echo "  Norm steps     : ${NORM_STEPS}"
    echo "  Also delta     : ${ALSO_TUNE_DELTA:-no}"
    echo "=============================================================================="

    EXTRA_FLAGS=""
    if [ "${ALSO_TUNE_DELTA}" = "1" ]; then
        EXTRA_FLAGS="${EXTRA_FLAGS} --also-tune-delta"
    fi

    "$PYTHON" delta_experiment/scripts/run_norm_tune_tta.py \
        --checkpoint-dir "${CHECKPOINT_DIR}" \
        --data-dir "${DATA_DIR}" \
        --output-dir "${OUTPUT_DIR}" \
        --max-videos "${MAX_VIDEOS}" \
        --norm-steps "${NORM_STEPS}" \
        --norm-lr "${NORM_LR}" \
        --norm-target "${NORM_TARGET}" \
        --num-cond-frames "${NUM_COND_FRAMES}" \
        --num-frames "${NUM_FRAMES}" \
        --gen-start-frame "${GEN_START_FRAME}" \
        --tta-total-frames "${TTA_TOTAL_FRAMES}" \
        --tta-context-frames "${TTA_CONTEXT_FRAMES}" \
        --num-inference-steps "${NUM_INFERENCE_STEPS}" \
        --guidance-scale "${GUIDANCE_SCALE}" \
        --resolution "${RESOLUTION}" \
        --seed "${SEED}" \
        ${SAVE_FLAGS} \
        ${EXTRA_FLAGS} \
        ${CAPTION_FLAGS} \
        ${FEATURE_FLAGS} \
        ${EVAL_FLAGS} \
        ${ES_FLAGS} \
        ${AUG_FLAGS}
    ;;

  film)
    echo "Method-specific config:"
    echo "  FiLM mode      : ${FILM_MODE}"
    echo "  FiLM LR        : ${FILM_LR}"
    echo "  FiLM steps     : ${FILM_STEPS}"
    echo "  Num groups     : ${NUM_GROUPS}"
    echo "=============================================================================="

    "$PYTHON" delta_experiment/scripts/run_film_tta.py \
        --checkpoint-dir "${CHECKPOINT_DIR}" \
        --data-dir "${DATA_DIR}" \
        --output-dir "${OUTPUT_DIR}" \
        --max-videos "${MAX_VIDEOS}" \
        --film-steps "${FILM_STEPS}" \
        --film-lr "${FILM_LR}" \
        --num-groups "${NUM_GROUPS}" \
        --film-mode "${FILM_MODE}" \
        --num-cond-frames "${NUM_COND_FRAMES}" \
        --num-frames "${NUM_FRAMES}" \
        --gen-start-frame "${GEN_START_FRAME}" \
        --tta-total-frames "${TTA_TOTAL_FRAMES}" \
        --tta-context-frames "${TTA_CONTEXT_FRAMES}" \
        --num-inference-steps "${NUM_INFERENCE_STEPS}" \
        --guidance-scale "${GUIDANCE_SCALE}" \
        --resolution "${RESOLUTION}" \
        --seed "${SEED}" \
        ${SAVE_FLAGS} \
        ${CAPTION_FLAGS} \
        ${FEATURE_FLAGS} \
        ${EVAL_FLAGS} \
        ${ES_FLAGS} \
        ${AUG_FLAGS}
    ;;

  vae_decoder)
    # VAE-Decoder-Only TTA (Modification 2 of
    # sweep_experiment/reports/LITERATURE_tta_recipe_modifications_2026-06-12.md).
    # Freeze the DiT; adapt only vae.decoder on round-trip reconstruction.
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
    VAE_TTA_STEPS="${VAE_TTA_STEPS:-10}"
    VAE_TTA_LR="${VAE_TTA_LR:-1e-5}"
    VAE_TTA_LPIPS_WEIGHT="${VAE_TTA_LPIPS_WEIGHT:-0.0}"
    VAE_TTA_GRAD_CLIP="${VAE_TTA_GRAD_CLIP:-1.0}"
    VAE_TTA_WEIGHT_DECAY="${VAE_TTA_WEIGHT_DECAY:-0.0}"

    echo "Method-specific config:"
    echo "  VAE TTA steps  : ${VAE_TTA_STEPS}"
    echo "  VAE TTA LR     : ${VAE_TTA_LR}"
    echo "  VAE LPIPS wt   : ${VAE_TTA_LPIPS_WEIGHT}"
    echo "  Grad clip      : ${VAE_TTA_GRAD_CLIP}"
    echo "  Weight decay   : ${VAE_TTA_WEIGHT_DECAY}"
    echo "=============================================================================="

    "$PYTHON" delta_experiment/scripts/run_vae_decoder_tta.py \
        --checkpoint-dir "${CHECKPOINT_DIR}" \
        --data-dir "${DATA_DIR}" \
        --output-dir "${OUTPUT_DIR}" \
        --max-videos "${MAX_VIDEOS}" \
        --start-video-idx "${START_VIDEO_IDX}" \
        --chunk-size "${CHUNK_SIZE}" \
        --vae-tta-steps "${VAE_TTA_STEPS}" \
        --vae-tta-lr "${VAE_TTA_LR}" \
        --vae-tta-lpips-weight "${VAE_TTA_LPIPS_WEIGHT}" \
        --vae-tta-grad-clip "${VAE_TTA_GRAD_CLIP}" \
        --vae-tta-weight-decay "${VAE_TTA_WEIGHT_DECAY}" \
        --num-cond-frames "${NUM_COND_FRAMES}" \
        --num-frames "${NUM_FRAMES}" \
        --gen-start-frame "${GEN_START_FRAME}" \
        --tta-total-frames "${TTA_TOTAL_FRAMES}" \
        --tta-context-frames "${TTA_CONTEXT_FRAMES}" \
        --num-inference-steps "${NUM_INFERENCE_STEPS}" \
        --guidance-scale "${GUIDANCE_SCALE}" \
        --resolution "${RESOLUTION}" \
        --seed "${SEED}" \
        ${SAVE_FLAGS} \
        ${CAPTION_FLAGS} \
        ${FEATURE_FLAGS} \
        ${CLIP_FLAGS} \
        ${EVAL_FLAGS} \
        ${NOPROMPT_FLAGS}
    ;;

  *)
    echo "ERROR: Unknown METHOD '${METHOD}'. Use: full|lora|temp_lora|delta_a|delta_b|delta_c|norm_tune|film|vae_decoder" >&2
    exit 1
    ;;
esac

echo ""
echo "=============================================================================="
echo "TTA Sweep Complete: ${METHOD} / ${RUN_ID}"
echo "=============================================================================="
echo "End time: $(date)"
echo "Results: ${OUTPUT_DIR}"
echo "=============================================================================="
