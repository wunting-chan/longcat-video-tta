#!/bin/bash
#SBATCH --job-name=mm_setup
#SBATCH --account=torch_pr_36_mren
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/wc3013/video-attractors-runs/multimodel/slurm/%x_%j.out
#SBATCH --error=/scratch/wc3013/video-attractors-runs/multimodel/slurm/%x_%j.err
# Multi-model Phase A (approved 2026-10-10): environments + weights for the probe models.
# CPU only. Creates new envs under /scratch/wc3013/conda-envs (never touches longcat) and
# downloads weights into /scratch/wc3013/.cache/huggingface. Third-party repos stay read-only.
set -uo pipefail
module purge; module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
unset PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1 HF_HOME=/scratch/wc3013/.cache/huggingface
export PIP_CACHE_DIR=/scratch/wc3013/pip-cache TMPDIR=/scratch/wc3013/tmp/mm_setup CONDA_PKGS_DIRS=/scratch/wc3013/conda-pkgs
mkdir -p "$TMPDIR"
E=/scratch/wc3013/conda-envs; TP=/scratch/wc3013/third_party
FA=https://github.com/Dao-AILab/flash-attention/releases/download
STATUS=/scratch/wc3013/video-attractors-runs/multimodel/setup_status.txt; : > $STATUS
step() { echo "== $(date +%T) $*"; }
ok() { echo "$1 OK" | tee -a $STATUS; }
bad() { echo "$1 FAILED" | tee -a $STATUS; }

# 1. va_wan: CausVid, Deep Forcing, Infinity-RoPE, SkyReels-V2 (all Wan2.1 code), NOVA
if [ "${SKIP_WAN:-0}" != 1 ]; then
step va_wan
[ -x $E/va_wan/bin/python ] || conda create -y -p $E/va_wan python=3.10 >/dev/null
P=$E/va_wan/bin/pip
$P install -q torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124 &&
$P install -q "$FA/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl" &&
$P install -q "diffusers==0.31.0" "transformers==4.49.0" "tokenizers==0.21.1" "accelerate==1.6.0" "numpy==1.26.4" \
   opencv-python-headless==4.10.0.84 imageio imageio-ffmpeg easydict ftfy omegaconf einops "av==13.1.0" sentencepiece \
   "pydantic==2.10.6" scipy scikit-image matplotlib lmdb wandb dashscope decord peft "huggingface_hub[cli]" safetensors protobuf codewithgpu &&
$P install -q --no-deps -e $TP/NOVA &&
$E/va_wan/bin/python -c "import torch, flash_attn, diffusers, transformers, diffnext; print('va_wan', torch.__version__, flash_attn.__version__)" \
  && ok va_wan || bad va_wan
fi

# 2. va_magi: MAGI-1 (README: torch 2.4.0, cu124; MagiAttention optional off-Hopper, not installed)
if [ "${SKIP_MAGI:-0}" != 1 ]; then
step va_magi
[ -x $E/va_magi/bin/python ] || conda create -y -p $E/va_magi python=3.10.12 >/dev/null
P=$E/va_magi/bin/pip
$P install -q torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu124 &&
$P install -q "$FA/v2.6.3/flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl" &&
grep -v -E "^(flash-attn|flashinfer)" $TP/MAGI-1/requirements.txt > $TMPDIR/magi_req.txt &&
$P install -q -r $TMPDIR/magi_req.txt &&
{ $P install -q flashinfer-python==0.2.0.post2 --extra-index-url https://flashinfer.ai/whl/cu124/torch2.4/ || echo "flashinfer optional install failed"; } &&
conda install -y -p $E/va_magi -c conda-forge ffmpeg=4.4 >/dev/null &&
$E/va_magi/bin/python -c "import torch, flash_attn, transformers; print('va_magi', torch.__version__, flash_attn.__version__)" \
  && ok va_magi || bad va_magi
fi

# 3. weights (Self Forcing ckpt already at /scratch/wc3013/wan-checkpoints/self_forcing_dmd.pt)
step weights
HF=$E/va_wan/bin/huggingface-cli
$HF download tianweiy/CausVid --include "autoregressive_checkpoint/model.pt" >/dev/null && ok w_causvid || bad w_causvid
$HF download Skywork/SkyReels-V2-DF-1.3B-540P >/dev/null && ok w_skyreels || bad w_skyreels
$HF download sand-ai/MAGI-1 --include "ckpt/magi/4.5B_distill/*" "ckpt/vae/*" "ckpt/t5/*" >/dev/null && ok w_magi || bad w_magi
$HF download BAAI/nova-d48w1024-osp480 >/dev/null && ok w_nova || bad w_nova
cat $STATUS
echo SETUP_DONE
