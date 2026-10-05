#!/bin/bash
# =============================================================================
# One-time setup for the VBench backfill pipeline.
#
# Creates a separate ``vbench-backfill`` conda env pinned to versions that
# work with VBench 0.1.5 (NumPy 1.x, PyTorch 2.5.x). The longcat env stays
# untouched so TTA continues to work with NumPy 2.x / PyTorch 2.6.
#
# Pre-downloads VBench's pretrained checkpoints to a known cache dir so that
# GPU compute jobs don't need outbound network at SLURM run time.
#
# Run on the LOGIN NODE (not a compute node) — needs network access.
#
#   bash scripts/setup_vbench_backfill_env.sh
#
# Idempotent: safe to re-run.
#
# All large I/O (pip wheel extraction, HF/Torch caches, conda env, build dirs)
# is forced onto /scratch so we never blow out /home or /tmp tmpfs quotas.
# =============================================================================

set -euo pipefail

ENV_NAME="${VBENCH_ENV_NAME:-vbench-backfill}"
SCRATCH_BASE="${SCRATCH_BASE:-/scratch/$USER}"
CACHE_DIR="${VBENCH_CACHE_DIR:-${SCRATCH_BASE}/vbench-cache}"

# ---- Force ALL caches/tmp onto /scratch -------------------------------------
# pip needs space for wheel download AND extraction. cudnn alone is ~665 MB.
# /tmp on login nodes is often a small tmpfs that fills up fast.
SCRATCH_TMP="${SCRATCH_BASE}/tmp/vbench-setup"
SCRATCH_PIP_CACHE="${SCRATCH_BASE}/.cache/pip"
SCRATCH_BUILD="${SCRATCH_BASE}/.cache/pip-build"
mkdir -p "${SCRATCH_TMP}" "${SCRATCH_PIP_CACHE}" "${SCRATCH_BUILD}" "${CACHE_DIR}"

export TMPDIR="${SCRATCH_TMP}"
export TEMP="${SCRATCH_TMP}"
export TMP="${SCRATCH_TMP}"
export PIP_CACHE_DIR="${SCRATCH_PIP_CACHE}"
export PIP_BUILD_DIR="${SCRATCH_BUILD}"
export PIP_NO_CACHE_DIR=0
export XDG_CACHE_HOME="${SCRATCH_BASE}/.cache"

# Verify scratch is actually writable + has space
if ! touch "${SCRATCH_TMP}/.write_test" 2>/dev/null; then
    echo "ERROR: cannot write to ${SCRATCH_TMP}" >&2
    exit 1
fi
rm -f "${SCRATCH_TMP}/.write_test"

echo "=============================================================================="
echo "VBench backfill env setup"
echo "=============================================================================="
echo "  ENV_NAME       : ${ENV_NAME}"
echo "  CACHE_DIR      : ${CACHE_DIR}"
echo "  TMPDIR         : ${TMPDIR}"
echo "  PIP_CACHE_DIR  : ${PIP_CACHE_DIR}"
echo "  PIP_BUILD_DIR  : ${PIP_BUILD_DIR}"
echo "  XDG_CACHE_HOME : ${XDG_CACHE_HOME}"
echo "=============================================================================="

# ---- Pre-flight: confirm /scratch has enough space (need ~6-8 GB peak) -----
# torch + cudnn wheels can spike to ~3 GB during extraction; checkpoints later
# add another ~3-4 GB. Be loud if we don't have enough.
SCRATCH_AVAIL_KB=$(df -P "${SCRATCH_BASE}" | awk 'NR==2 {print $4}')
SCRATCH_AVAIL_GB=$((SCRATCH_AVAIL_KB / 1024 / 1024))
echo "[pre-flight] /scratch free space: ${SCRATCH_AVAIL_GB} GB"
if [ "${SCRATCH_AVAIL_GB}" -lt 10 ]; then
    echo "WARN: less than 10 GB free on /scratch. Pip extraction may fail."
fi

# ---- Clean up any partial env from a previous failed run --------------------
# Detect partial env: env exists but has no 'numpy' (we know we install numpy
# very early, so a partial install almost certainly hasn't reached vbench).
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    if [ -d "/scratch/$USER/conda-envs/${ENV_NAME}" ] && \
       ! conda run -n "${ENV_NAME}" python -c "import numpy" >/dev/null 2>&1; then
        echo "[cleanup] partial env detected (no numpy); removing and recreating."
        conda env remove -n "${ENV_NAME}" -y || true
    fi
fi

# ---- Create or reuse env ----------------------------------------------------
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "[1/3] env '${ENV_NAME}' already exists; skipping creation."
else
    echo "[1/3] Creating conda env '${ENV_NAME}' (python=3.10) ..."
    conda create -n "${ENV_NAME}" python=3.10 -y
fi

# Activate env (works under bash with conda init)
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

# Sanity-print: confirm Python is on /scratch, not /home
PY_PATH="$(which python)"
PIP_CMD=(python -m pip)
echo ""
echo "  Active env path : ${CONDA_PREFIX}"
echo "  Python binary   : ${PY_PATH}"
if [[ "${PY_PATH}" != "${CONDA_PREFIX}/bin/python" ]]; then
    echo "ERROR: 'python' is not the conda env interpreter." >&2
    echo "       expected: ${CONDA_PREFIX}/bin/python" >&2
    echo "       got:      ${PY_PATH}" >&2
    echo "       Re-run: module load anaconda3/2025.06 && source .../conda.sh && conda activate ${ENV_NAME}" >&2
    exit 1
fi
case "${CONDA_PREFIX}" in
    /scratch/*)  echo "  (env is on /scratch — good)";;
    *)           echo "  WARN: env is NOT on /scratch — pip wheels may fill /home";;
esac
echo ""

# ---- Install pinned deps ----------------------------------------------------
echo "[2/3] Installing pinned dependencies for VBench 0.1.5 ..."
echo "       (wheels download to ${PIP_CACHE_DIR}, extract in ${TMPDIR})"

"${PIP_CMD[@]}" install \
    --cache-dir "${PIP_CACHE_DIR}" \
    'setuptools>=70,<80' \
    'numpy==1.26.4' \
    'torch==2.5.1' \
    'torchvision==0.20.1' \
    'vbench==0.1.5' \
    'decord' \
    'opencv-python-headless==4.11.0.86' \
    'einops' \
    'timm' \
    'pyiqa' \
    'open-clip-torch' \
    'ftfy' \
    'regex' \
    'tqdm'

# Belt-and-suspenders: if a transitive dep dragged opencv-python (with libGL
# requirement) onto the env, replace it with the headless variant which
# provides the same cv2 API but no system-libGL dependency.
if "${PIP_CMD[@]}" show opencv-python >/dev/null 2>&1; then
    echo "  [fix] removing opencv-python (libGL-dependent) ..."
    "${PIP_CMD[@]}" uninstall -y opencv-python
    "${PIP_CMD[@]}" install --cache-dir "${PIP_CACHE_DIR}" 'opencv-python-headless==4.11.0.86'
fi

# Belt-and-suspenders #2: opencv-python-headless 4.13+ requires numpy>=2,
# which conflicts with vbench 0.1.5's numpy<2 pin. Re-pin numpy if anything
# upgraded it during the install.
"${PIP_CMD[@]}" install --cache-dir "${PIP_CACHE_DIR}" 'numpy==1.26.4'

# Always ensure headless opencv is present. Use --no-deps so pip cannot
# upgrade numpy to 2.x when reinstalling opencv (breaks pyiqa/imgaug/vbench).
"${PIP_CMD[@]}" install --cache-dir "${PIP_CACHE_DIR}" --no-deps \
    'opencv-python-headless==4.11.0.86'
if ! python -c "import cv2; print('  cv2 =', cv2.__version__)"; then
    echo "ERROR: opencv-python-headless install failed — cv2 still missing." >&2
    exit 1
fi

# ---- Verify all 7 dimensions import cleanly --------------------------------
echo "[3/3] Verifying dimension imports + pre-downloading checkpoints ..."

export HF_HOME="${CACHE_DIR}/hf"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TORCH_HOME="${CACHE_DIR}/torch"
mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" "${TORCH_HOME}"

python3 - <<'PY'
import os, sys, importlib, traceback

print("\n===== Importable dimension modules =====")
DIMS = ["subject_consistency","background_consistency","motion_smoothness",
        "dynamic_degree","aesthetic_quality","imaging_quality","temporal_flickering"]
ok = []
for dim in DIMS:
    try:
        importlib.import_module(f"vbench.{dim}")
        print(f"  vbench.{dim} -> OK")
        ok.append(dim)
    except Exception as e:
        print(f"  vbench.{dim} -> FAIL: {type(e).__name__}: {e}")

if len(ok) < 7:
    print(f"\n[error] {len(ok)}/7 dimensions importable — fix deps before backfill.")
    sys.exit(1)

print("\n===== Instantiating VBench (triggers checkpoint downloads) =====")
import torch, vbench as _v
pkg_dir = os.path.dirname(_v.__file__)
full_info = os.path.join(pkg_dir, "VBench_full_info.json")
if not os.path.exists(full_info):
    full_info = os.path.join(os.path.dirname(pkg_dir), "vbench", "VBench_full_info.json")

cache = os.environ.get("VBENCH_CACHE_DIR", os.path.expanduser("~/vbench-cache"))
output_path = os.path.join(cache, "_pretrain_check")
os.makedirs(output_path, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  device         : {device}")
print(f"  full_info_json : {full_info}")
print(f"  output_path    : {output_path}")

from vbench import VBench
try:
    vb = VBench(device, full_info, output_path)
    print("  VBench instance creation: OK")
except Exception as e:
    print(f"  VBench instance creation: FAIL: {type(e).__name__}: {e}")
    traceback.print_exc()
    sys.exit(1)
PY

echo ""
echo "=============================================================================="
echo "Setup complete."
echo ""
echo "Cache dir : ${CACHE_DIR}"
echo "Env name  : ${ENV_NAME}  (NOT longcat — keep them separate!)"
echo ""
echo "Next: run discovery to see how many method dirs need backfill:"
echo "  python3 scripts/discover_vbench_backfill_targets.py"
echo ""
echo "Then submit the sweep:"
echo "  bash scripts/submit_vbench_backfill_all.sh"
echo "=============================================================================="
