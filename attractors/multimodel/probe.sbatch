#!/bin/bash
#SBATCH --job-name=mm_probe
#SBATCH --account=torch_pr_36_mren
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/wc3013/video-attractors-runs/multimodel/slurm/%x_%j.out
#SBATCH --error=/scratch/wc3013/video-attractors-runs/multimodel/slurm/%x_%j.err
# Multi-model Phase A probe (approved 2026-10-10): one held-out prompt (MovieGenBench line 9 =
# prompt index 8), seed 0, each repo's own inference script, unmodified. Records wall time,
# peak GPU memory and the output length, so Phase B can be costed.
#   MODEL  = causvid | deepforcing | infrope | skyreels | magi | nova
#   SECS   = target video length in seconds (default 30)
set -uo pipefail
module purge; module load anaconda3/2025.06
unset PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1 HF_HOME=/scratch/wc3013/.cache/huggingface TORCH_HOME=/scratch/wc3013/.cache/torch
export TMPDIR=/scratch/wc3013/tmp/mm_probe_${SLURM_JOB_ID}; mkdir -p $TMPDIR
TP=/scratch/wc3013/third_party; E=/scratch/wc3013/conda-envs
M=/scratch/wc3013/video-attractors-runs/multimodel
SECS=${SECS:-30}; OUT=$M/probe/${MODEL}_${SECS}s; RD=$M/rundir/$MODEL
mkdir -p $OUT $RD
sed -n 9p $TP/Self-Forcing/prompts/MovieGenVideoBench_extended.txt > $OUT/prompt.txt
ln -sfn /scratch/wc3013/video-attractors-runs/rundir/wan_models $RD/wan_models
snap() { ls -d $HF_HOME/hub/models--$1/snapshots/*/ | head -1; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee $OUT/gpu.txt
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> $OUT/mem.log; sleep 5; done ) &
MEMPID=$!
LAT=$(( SECS * 4 ))                          # Wan-1.3B students: 4 latents per second at 16 fps
cd $RD
T0=$(date +%s)
case $MODEL in
  causvid)   # 21-latent rollouts, 3-latent overlap -> 72 new frames (4.5 s) per rollout
    PY=$E/va_wan/bin/python
    PYTHONPATH=$TP/CausVid $PY -u $M/seeded_run.py 0 $TP/CausVid/minimal_inference/longvideo_autoregressive_inference.py \
      --config_path $TP/CausVid/configs/wan_causal_dmd.yaml \
      --checkpoint_folder $(snap tianweiy--CausVid)autoregressive_checkpoint \
      --prompt_file_path $OUT/prompt.txt --output_folder $OUT \
      --num_rollout $(( (SECS * 16 + 71) / 72 )) --num_overlap_frames 3 ;;
  deepforcing)   # README "Deep Sink & Participative Compression" settings (DS_PC_inference.sh)
    ln -sfn $TP/DeepForcing/configs $RD/configs
    PYTHONPATH=$TP/DeepForcing $E/va_wan/bin/python -u $TP/DeepForcing/inference.py \
      --config_path configs/self_forcing_dmd/self_forcing_dmd_sink10.yaml \
      --checkpoint_path /scratch/wc3013/wan-checkpoints/self_forcing_dmd.pt \
      --data_path $OUT/prompt.txt --output_folder $OUT --use_ema --seed 0 \
      --num_output_frames $LAT --Budget 16 --Recent 4 ;;
  infrope)   # inference.sh settings; length via --num_output_frames (no [Ns] tag in the prompt)
    ln -sfn $TP/infinity-rope/configs $RD/configs
    PYTHONPATH=$TP/infinity-rope $E/va_wan/bin/python -u $TP/infinity-rope/inference.py \
      --config_path configs/self_forcing_dmd.yaml \
      --checkpoint_path /scratch/wc3013/wan-checkpoints/self_forcing_dmd.pt \
      --data_path $OUT/prompt.txt --output_folder $OUT --use_ema --seed 0 \
      --num_output_frames $LAT ;;
  skyreels)   # README long-video DF settings; 24 fps, 97-frame windows, 17-frame history -> +80 per window
    NF=$(( 97 + 80 * ( (SECS * 24 - 97 + 79) / 80 ) ))
    cd $TP/SkyReels-V2 && $E/va_wan/bin/python -u generate_video_df.py \
      --model_id $(snap Skywork--SkyReels-V2-DF-1.3B-540P) --resolution 540P --ar_step 0 \
      --base_num_frames 97 --num_frames $NF --overlap_history 17 --addnoise_condition 20 \
      --seed 0 --outdir $OUT --prompt "$(cat $OUT/prompt.txt)" ;;
  magi)   # 4.5B distill config with local weight paths, seed 0, 480x832 to match the Wan models
    S=$(snap sand-ai--MAGI-1)
    $E/va_magi/bin/python - <<PY
import json
c = json.load(open("$TP/MAGI-1/example/4.5B/4.5B_distill_config.json"))
r = c["runtime_config"]
r.update(load="${S}ckpt/magi/4.5B_distill", t5_pretrained="${S}ckpt/t5", vae_pretrained="${S}ckpt/vae",
         seed=0, num_frames=$SECS * 24, video_size_h=480, video_size_w=832)
json.dump(c, open("$OUT/magi_config.json", "w"), indent=2)
PY
    export MASTER_ADDR=localhost MASTER_PORT=$(( 20000 + SLURM_JOB_ID % 10000 )) GPUS_PER_NODE=1 NNODES=1 WORLD_SIZE=1
    export PAD_HQ=1 PAD_DURATION=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OFFLOAD_T5_CACHE=true OFFLOAD_VAE_CACHE=true
    cd $TP/MAGI-1 && PYTHONPATH=$TP/MAGI-1 $E/va_magi/bin/python -u inference/pipeline/entry.py \
      --config_file $OUT/magi_config.json --mode t2v --prompt "$(cat $OUT/prompt.txt)" --output_path $OUT/magi.mp4 ;;
  nova)   # NOVA latents: README 9 latents = 33 frames at 12 fps (4 frames per latent after the first)
    $E/va_wan/bin/python -u $M/nova_run.py --prompt-file $OUT/prompt.txt --latents $(( (SECS * 12 - 1) / 4 + 1 )) \
      --seed 0 --out $OUT/nova.mp4 ;;
  *) echo "unknown MODEL $MODEL"; exit 2 ;;
esac
RC=$?
T1=$(date +%s); kill $MEMPID 2>/dev/null
VID=$(ls -t $OUT/*.mp4 $OUT/*/*.mp4 2>/dev/null | head -1)
$E/va_wan/bin/python - <<PY | tee $OUT/probe.json
import json, imageio.v3 as iio
vid = "$VID"; d = {"model": "$MODEL", "target_s": $SECS, "rc": $RC, "wall_s": $T1 - $T0, "video": vid}
mem = [int(x) for x in open("$OUT/mem.log").read().split() if x.strip().isdigit()]
d["peak_mem_mib"] = max(mem) if mem else None
if vid:
    meta = iio.immeta(vid, plugin="pyav"); n = sum(1 for _ in iio.imiter(vid, plugin="pyav"))
    fps = meta.get("fps"); d.update(frames=n, fps=fps, video_s=n / fps if fps else None)
    if d.get("video_s"): d["wall_per_video_s"] = d["wall_s"] / d["video_s"]
print(json.dumps(d))
PY
echo PROBE_DONE rc=$RC
