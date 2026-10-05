# Cluster & sbatch Onboarding Guide (LongCat-Video-TTA)

**Audience:** an AI agent (or human) that is brand new to this project and needs
to submit fine-tuning / inference jobs on our SLURM cluster. Read this file
end-to-end before writing any sbatch script. If you follow it, you can fine-tune
LongCat-Video on a handful of user videos and run long-horizon continuation
without tripping over any of the cluster's quirks.

This guide is deliberately self-contained. It duplicates a few facts from the
top-level `AGENTS.md` on purpose so you don't have to cross-reference while
writing jobs.

**Interaction model (important):** you (the agent) work **locally only** — you
do NOT have cluster access and never run `sbatch`/`squeue` yourself. Your job is
to **author** the sbatch scripts and the exact submit/monitor commands; the
human operator is the one who runs them on the cluster and pastes back output.
So: produce complete, copy-paste-ready commands and files, state clearly which
directory to run them from, and don't assume you can observe job state directly
— ask the operator to paste `squeue`/log output when you need it. The monitoring
commands in §5 are things you write *for the operator to run*, not for you.

---

**Laptop SSH/SCP (LOCKED):** from the user's Mac the host is always
`wc3013@torch`. Never invent `torch-login-*.hpc.nyu.edu` as the scp/ssh
target. The prompt `[wc3013@torch-login-a-1]` is the node after login.

## 0. TL;DR — the six non-negotiable cluster quirks

If you remember nothing else, remember these. Every one of them has bitten us
before.

1. **`--account=torch_pr_36_mren` is REQUIRED on every job.** Without the
   correct account flag, `sbatch` either rejects the job or it never leaves the
   queue. This is our SLURM allocation / billing account.
2. **Everything lives under `/scratch`, never `/home`.** `/home` is small and
   not meant for data, checkpoints, caches, or results. Our scratch root is
   `/scratch/wc3013`. Point all inputs, outputs, and caches there.
3. **The conda env is invoked through a specific incantation, and you MUST
   `unset PYTHONHOME` / `unset PYTHONPATH` after `module load`.** If you skip the
   unset, `import torch` fails with "No module named 'torch'" because the system
   Anaconda's `sys.path` shadows our env. See §3.
4. **Request the GPU the job can fill.** Wan V2V generate stays
   `--gres=gpu:h200:1` with `VIDEO_WORKERS=2` (137-frame KV ≈ 39 GB;
   candidate k-batch does not fit). VBench is `--gres=gpu:1
   --constraint=l40s` — do not put it on the 2-way H200 cap. See
   `paper_tables/2026-08-23_wan_gpu_batch_policy.md`. Jobs are
   preemptible; max wall is 48 h. Keep generate resumable. See §4.
5. **Redirect all model/framework caches (HuggingFace, Torch, VBench) into
   `/scratch`.** Default cache locations land in `/home` and fill the quota.
6. **Do NOT run `git` from the local iCloud workspace.** The local Mac repo
   dehydrates and times out. All git ops happen on the cluster (`git pull`) or
   via the `/tmp` clone subagent pattern documented in `AGENTS.md` §5. On the
   cluster itself, `git pull --ff-only origin main` inside the project root is
   fine.

---

## 1. Cluster topology & canonical paths

| Thing | Path | Notes |
|---|---|---|
| Scratch root | `/scratch/wc3013` | All persistent data + outputs. Referred to as `SCRATCH_BASE` in scripts. |
| Project root | `/scratch/wc3013/longcat-video-tta` | The git checkout of this repo on the cluster. `PROJECT_ROOT`. |
| Model checkpoints | `/scratch/wc3013/longcat-video-checkpoints` | LongCat-Video weights. `CHECKPOINT_DIR`. |
| Conda env | `/scratch/wc3013/conda-envs/longcat` | The one and only env for this project. `CONDA_ENV`. |
| Datasets | `/scratch/wc3013/longcat-video-tta/datasets/<name>` | Each dataset is a directory (see §6). |
| Results | `/scratch/wc3013/longcat-video-tta/<experiment>/results/<series>/<run>` | Per-run output dirs. |
| SLURM logs | `<experiment>/slurm_log/%x_%j.out` (and `.err`) | `%x`=job-name, `%j`=job-id. Create the dir in-script. |
| HF cache | `/scratch/wc3013/.cache/huggingface` | Set `HF_HOME` + `TRANSFORMERS_CACHE`. |
| Torch cache | `/scratch/wc3013/.cache/torch` | Set `TORCH_HOME`. |
| VBench cache | `/scratch/wc3013/.cache/vbench` | Set `VBENCH_CACHE_DIR` (only if computing VBench). |
| Temp / pip | `/scratch/wc3013/tmp`, `/scratch/wc3013/pip-cache` | Set `TMPDIR` / `PIP_CACHE_DIR` for data-prep jobs so `/tmp` and `/home` don't fill. |

The Python entry points you'll call live under the project root:

- `lora_experiment/scripts/run_lora_tta.py` — LoRA fine-tuning + continuation.
- `lora_experiment/scripts/run_full_tta.py` — full-model fine-tuning + continuation.
- `delta_experiment/scripts/run_delta_a.py` — AdaSteer (our lightweight method) + a no-TTA baseline.
- `sweep_experiment/sbatch/run_sweep.sbatch` — the **unified runner** that wraps all of the above and is driven entirely by environment variables (see §7). Prefer this over writing a runner from scratch.

---

## 2. The SLURM header we always use

Copy this header verbatim and only change `--job-name`, `--time`, and the log
paths. Every value here has a reason.

```bash
#!/bin/bash
#SBATCH --job-name=my_job              # short, greppable
#SBATCH --account=torch_pr_36_mren     # REQUIRED — our allocation account
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G                     # 192G–256G is our norm for a single H200 job
#SBATCH --gres=gpu:h200:1              # one H200; we do not multi-GPU these jobs
#SBATCH --time=48:00:00                # 48h is this cluster's MAX limit. See §4.
#SBATCH --output=my_experiment/slurm_log/%x_%j.out
#SBATCH --error=my_experiment/slurm_log/%x_%j.err
# Allow preemptible H200s; cancelled jobs get requeued (per cluster IT guidance)
#SBATCH --comment="preemption=yes;requeue=true"
```

Notes:
- `--account=torch_pr_36_mren` can be set in the header **or** passed on the
  command line (`sbatch --account=torch_pr_36_mren ...`). Our submit wrappers
  pass it on the CLI so it can be overridden by an `ACCOUNT` env var; standalone
  sbatch files hard-code it in the header. Either is fine — just never omit it.
- `--mem`: use `256G` for generation/eval-heavy jobs (FVD/VBench load extra
  models), `192G` for lean fine-tune-only jobs.
- The `--comment` preemption line is what cluster IT asked us to set so requeue
  behaves. Keep it on long jobs.

---

## 3. Environment invocation (THE quirk that breaks everything)

`module load` mutates `PYTHONHOME` / `PYTHONPATH` to point at the *system*
Anaconda. If you then `conda activate` our env, Python still resolves imports
against the system Anaconda's `sys.path` — wrong Python version, no `torch`.
The fix is to `unset` both variables **after** activating. This block goes into
every job, right after the SLURM header:

```bash
set -euo pipefail
export PYTHONNOUSERSITE=1        # ignore ~/.local site-packages

SCRATCH_BASE="/scratch/wc3013"
PROJECT_ROOT="${SCRATCH_BASE}/longcat-video-tta"
CONDA_ENV="${SCRATCH_BASE}/conda-envs/longcat"

module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

# CRITICAL: undo the PYTHONHOME/PYTHONPATH that `module load` set, or torch
# import fails even though the env is correct.
unset PYTHONHOME
unset PYTHONPATH

# Use the env's interpreter explicitly (don't trust a bare `python`).
PYTHON="${CONDA_ENV}/bin/python"

# Prepend the env's libs so CUDA/other .so files resolve from our env.
export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"

# Redirect ALL caches into /scratch (never /home).
export HF_HOME="${SCRATCH_BASE}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}"
export TORCH_HOME="${SCRATCH_BASE}/.cache/torch"
export VBENCH_CACHE_DIR="${SCRATCH_BASE}/.cache/vbench"
mkdir -p "$HF_HOME" "$TORCH_HOME" "$VBENCH_CACHE_DIR"

# Sanity check — fail fast if the env is broken.
"$PYTHON" -c "import torch; print(f'torch {torch.__version__}, CUDA {torch.cuda.is_available()}')" || {
    echo "ERROR: torch import failed — conda env broken or PYTHONHOME not unset" >&2
    exit 1
}

cd "$PROJECT_ROOT"
mkdir -p my_experiment/slurm_log
```

If you ever see `ModuleNotFoundError: No module named 'torch'` from a job whose
env is clearly installed, the `unset PYTHONHOME/PYTHONPATH` lines are missing or
out of order. That is the #1 cause.

---

## 4. Preemption, wall-time, and making jobs survivable

- H200 GPUs are **preemptible**: SLURM can kill your job and requeue it. Keep
  `--comment="preemption=yes;requeue=true"`.
- **The maximum wall-time on this cluster is 48 h** (`--time=48:00:00`). You can
  request anything up to that; requesting the max is fine. For long single jobs
  that need to run continuously (e.g. this multi-minute generation task), set
  `--time=48:00:00` — we expect to need the full limit.
- Because 48 h is a hard ceiling and the GPUs are preemptible, still make long
  work **resumable / chunked** (see §8) so a preemption or a job that would
  exceed 48 h only loses partial progress. We have repeatedly lost single-job
  long-horizon runs to timeouts; chunking is the insurance.
- Make long work resumable: our runners checkpoint per-video progress and skip
  completed work on requeue. Prefer them over ad-hoc scripts.
- Typical wall-times we use:
  - short-horizon fine-tune + gen, ~100 videos/chunk: `12:00:00`
  - long-horizon (many rollout steps): `20:00:00`–`48:00:00`
  - **this 5-video, ~5-minute generation job: `48:00:00` (the max)**
  - data prep (CPU only): `04:00:00`

---

## 5. Monitoring & basic SLURM commands

```bash
# Submit (from PROJECT_ROOT so relative log paths resolve):
cd /scratch/wc3013/longcat-video-tta
sbatch my_experiment/sbatch/my_job.sbatch

# Watch your queue:
squeue -u $USER
squeue -u $USER | grep my_job          # filter by job-name prefix

# Tail a running job's log:
tail -f my_experiment/slurm_log/my_job_<jobid>.out

# Cancel:
scancel <jobid>
scancel -u $USER --name=my_job         # cancel all with a given name

# Why is my job pending / how did it exit:
squeue -j <jobid> --start
sacct -j <jobid> --format=JobID,JobName,State,Elapsed,MaxRSS,ExitCode
```

---

## 6. Dataset layout — how to place the user's videos

A "dataset directory" that the runners accept looks like this:

```
/scratch/wc3013/longcat-video-tta/datasets/<name>/
├── videos/
│   ├── clip_000.mp4
│   ├── clip_001.mp4
│   └── ...
└── metadata.csv        # one row per video
```

`metadata.csv` columns we rely on (schema shared with `ucf101_pool_max` and the
Panda pools):

| Column | Meaning |
|---|---|
| `videoID` | unique id; must match the `videos/<videoID>.mp4` filename stem |
| `caption` | text prompt describing the clip (used as the generation prompt) |
| `source_video_id` | (optional) original source id, for same-source filtering |
| `chunk_index`, `chunk_start_sec` | (optional) provenance for segment pools |

For the 5-video use case, the minimum you need is `videos/*.mp4` plus a
`metadata.csv` with a `videoID` and a good `caption` per clip. The caption
matters: it's the text prompt the model continues under, so write a concrete
description of the scene/subject (there is a caption-quality guard that will
warn/fail on empty or duplicate captions — see `CAPTION_GUARD_MODE` in §7).

**480p is the standard resolution** (`RESOLUTION=480p`). If the user's clips
aren't already 832×480, transcode them (ffmpeg is available in the env, or
install via `conda install -c conda-forge ffmpeg`). Keep clips comfortably
longer than `NUM_FRAMES` so there are enough conditioning + (for eval) ground
truth frames.

---

## 7. The unified runner (`run_sweep.sbatch`) and how env vars drive it

Rather than writing a bespoke sbatch per experiment, we drive the single
`sweep_experiment/sbatch/run_sweep.sbatch` template with environment variables
passed through `sbatch --export`. It contains the §2 header and §3 environment
block already, then dispatches on `METHOD`.

Required env vars:
- `METHOD` — `lora` | `full` | `delta_a` | `temp_lora` | `norm_tune` | `film` | `vae_decoder`
- `RUN_ID` — a name for this run (becomes a results subdir)

Commonly-set env vars (defaults in parentheses):

| Var | Meaning |
|---|---|
| `DATA_DIR` | dataset dir (§6) |
| `OUTPUT_DIR` | where results/videos/metrics go |
| `SERIES_NAME` | groups runs under `results/<series>/` |
| `MAX_VIDEOS` (100) | cap on videos processed |
| `START_VIDEO_IDX` (0), `CHUNK_SIZE` (0) | for chunked submission (§8) |
| `NUM_FRAMES` (16) | total frames per generation window (cond + gen) |
| `NUM_COND_FRAMES` (2) | conditioning frames re-fed each rollout step |
| `GEN_START_FRAME` (32) | index where generation starts within the source clip |
| `TTA_TOTAL_FRAMES`, `TTA_CONTEXT_FRAMES` | which frames the fine-tune loop trains on |
| `ROLLOUT_STEPS` (1) | **# of autoregressive continuation steps** — the long-horizon knob (§9) |
| `NUM_INFERENCE_STEPS` (50), `GUIDANCE_SCALE` (4.0) | sampler settings |
| `RESOLUTION` (480p), `SEED` (42) | |
| `LORA_RANK` (8), `LORA_ALPHA` (16), `LORA_TARGET_BLOCKS` (all), `TARGET_MODULES` (qkv,proj) | LoRA config |
| `LEARNING_RATE` (2e-4), `NUM_STEPS` (20), `WARMUP_STEPS` (3), `WEIGHT_DECAY` (0.01), `MAX_GRAD_NORM` (10.0) | optimizer |
| `BATCH_VIDEOS` (1) | if >1, fine-tune ONE adapter jointly over that many videos (see §9b) |
| `NO_SAVE_VIDEOS` (1) | **set to `0` to actually write the generated .mp4s** |
| `FPS` (24) | playback fps for saved `.mp4`s (`lora` runner only). **Set `FPS=32` for this high-quality job** (§9b). Default 24 keeps other experiments unchanged. |
| `COMPUTE_FVD` / `COMPUTE_FID` / `COMPUTE_VBENCH` (0) | online metrics (need GT frames) |
| `CAPTION_GUARD_MODE` (fail) | `fail` \| `warn` \| `off` — set `warn` for tiny custom datasets |

The `--export` string must start with `ALL,` so the job inherits your shell env
(needed for the module/conda plumbing) and then appends your overrides:

```bash
sbatch --account=torch_pr_36_mren \
    --job-name=demo \
    --time=12:00:00 \
    --export="ALL,METHOD=lora,RUN_ID=demo,DATA_DIR=/scratch/.../datasets/my5,OUTPUT_DIR=/scratch/.../results/my5/demo,MAX_VIDEOS=5,NO_SAVE_VIDEOS=0" \
    sweep_experiment/sbatch/run_sweep.sbatch
```

> Gotcha: `--export` is a single comma-separated string with **no spaces**.
> Values with spaces (e.g. a fixed caption) need special handling; prefer
> putting captions in `metadata.csv` instead.

---

## 8. Chunking pattern (for large N — you can skip this for 5 videos)

For hundreds/thousands of videos we split into chunks so each job stays under
the 24 h wall and preemption only loses one chunk. A submit wrapper (`.sh`)
loops over chunks and calls `sbatch` once per chunk, setting
`START_VIDEO_IDX=chunk*CHUNK_SIZE` and `CHUNK_SIZE`. Results land in
`.../<run>/chunk_<k>/` and are stitched afterward with
`sweep_experiment/scripts/merge_chunks.py --results-dir <run_dir> --recursive`.

**For the 5-video use case this is unnecessary** — a single job with
`MAX_VIDEOS=5` is correct. The chunking machinery is documented here only so you
recognize it in existing wrappers like
`sweep_experiment/sbatch/submit_longhorizon_1000v_chunked.sh`.

---

## 9. Frame geometry & long-horizon (5-minute) continuation

### 9a. What the frame knobs mean

The model works on a sliding window of `NUM_FRAMES` frames. Of a source clip:
- frames before `GEN_START_FRAME` are available as context / conditioning,
- the fine-tune ("TTA") loop trains on the window
  `[GEN_START_FRAME - TTA_TOTAL_FRAMES, GEN_START_FRAME)` using
  `TTA_CONTEXT_FRAMES` conditioning frames,
- generation then produces the frames of the window, re-conditioning on the last
  `NUM_COND_FRAMES` each step.

Our validated **long-horizon** geometry (from
`submit_longhorizon_1000v_chunked.sh`) is:

```
NUM_FRAMES=76         # 14 conditioning + 62 generated per window
NUM_COND_FRAMES=14
GEN_START_FRAME=48
TTA_TOTAL_FRAMES=48
TTA_CONTEXT_FRAMES=14
NUM_INFERENCE_STEPS=50
GUIDANCE_SCALE=4.0
```

### 9b. Reaching ~5 minutes: `ROLLOUT_STEPS`

A single window is short. To continue for minutes you chain windows
autoregressively via `ROLLOUT_STEPS`. Each step re-conditions on the tail of the
previous output and emits roughly `(NUM_FRAMES - NUM_COND_FRAMES)` new frames.

**Playback fps for this job: set `FPS=32` (the default is 24).** We want
high-resolution, high-quality output, so the generated `.mp4`s should be written
at **32 fps**. The runner exposes this as an `FPS` env var (wired to the
`--fps` flag on `lora_experiment/scripts/run_lora_tta.py`). Its **default is 24
on purpose** — that keeps every existing experiment/agent on this project
byte-identical — so *you* must explicitly add `FPS=32` to the `--export` string
for this generation job. Do NOT leave it at 24 for this task.

> Scope note: as of this writing the `--fps`/`FPS` knob is honoured by the
> `lora` runner only (the one used in the recipe below). The other methods
> (`full`, `delta_a`, etc.) still save at 24 fps; if you switch methods and need
> 32 fps, wire `--fps` into that runner the same way first.

Rough budget (use **32 fps** for this job):

```
new_frames_per_step ≈ NUM_FRAMES - NUM_COND_FRAMES        # e.g. 76 - 14 = 62
total_seconds       ≈ ROLLOUT_STEPS * new_frames_per_step / 32
ROLLOUT_STEPS       ≈ target_seconds * 32 / new_frames_per_step
```

For **5 minutes (300 s)** at 32 fps with the long-horizon geometry above:

```
ROLLOUT_STEPS ≈ 300 * 32 / 62 ≈ 155 steps
```

That is a *lot* of autoregressive steps. Two things to flag to the user:

1. **Drift/quality decay is expected** over that many steps — this is exactly
   the long-horizon regime our TTA work targets. Fine-tuning on their clips
   (LoRA/AdaSteer) is what keeps the background/subject on-model deeper into the
   rollout.
2. **Cost & wall-time:** ~155 sequential generation passes per video is a lot.
   Request the **max `--time=48:00:00`** for this job, and even then prefer
   (a) running one video per job, and/or (b) starting with a smaller
   `ROLLOUT_STEPS` (e.g. 8–16, ~15–30 s at 32 fps) to validate quality/settings
   before scaling to the full ~155. If a single video can't finish 155 steps
   within 48 h, split the rollout across resumable/chunked jobs (§8).

To fine-tune **one adapter across all 5 clips** (recommended when they share a
background/subject) set `BATCH_VIDEOS=5` and `MAX_VIDEOS=5`; the runner pools the
clips into a single adaptation instead of adapting per-video.

---

## 10. Ready-to-use recipes for the 5-video task

### Recipe A — Prepare the dataset (one-time, do this first)

Put the 5 clips on the cluster and build the dataset dir:

```bash
cd /scratch/wc3013/longcat-video-tta
mkdir -p datasets/user5/videos
# copy the 5 mp4s into datasets/user5/videos/ as clip_000.mp4 ... clip_004.mp4
# then create datasets/user5/metadata.csv with columns: videoID,caption
#   clip_000,"a person cooking at a wooden kitchen counter, static camera"
#   ...
```

(Transcode to 832×480 if needed with ffmpeg. Write concrete captions — the
caption guard will complain about empty/duplicate ones; for a 5-row file set
`CAPTION_GUARD_MODE=warn`.)

### Recipe B — LoRA fine-tune + long-horizon continuation (single job)

Start with a **short validation rollout** to confirm everything works, then
scale `ROLLOUT_STEPS` up toward 5 minutes.

```bash
cd /scratch/wc3013/longcat-video-tta

sbatch --account=torch_pr_36_mren \
    --job-name=user5_lora_lh \
    --time=48:00:00 \
    --export="ALL,\
METHOD=lora,\
RUN_ID=user5_lora,\
SERIES_NAME=user5_longhorizon,\
DATA_DIR=/scratch/wc3013/longcat-video-tta/datasets/user5,\
OUTPUT_DIR=/scratch/wc3013/longcat-video-tta/lora_experiment/results/user5_longhorizon/user5_lora,\
MAX_VIDEOS=5,\
BATCH_VIDEOS=5,\
NUM_FRAMES=76,NUM_COND_FRAMES=14,GEN_START_FRAME=48,\
TTA_TOTAL_FRAMES=48,TTA_CONTEXT_FRAMES=14,\
NUM_INFERENCE_STEPS=50,GUIDANCE_SCALE=4.0,RESOLUTION=480p,SEED=42,\
LORA_RANK=8,LORA_ALPHA=16,LORA_TARGET_BLOCKS=all,NUM_STEPS=20,LEARNING_RATE=2e-4,\
WARMUP_STEPS=3,WEIGHT_DECAY=0.01,MAX_GRAD_NORM=10.0,\
ROLLOUT_STEPS=8,\
FPS=32,\
NO_SAVE_VIDEOS=0,\
CAPTION_GUARD_MODE=warn,FEATURE_FRAME_GUARD_MODE=warn,\
COMPUTE_FVD=0,COMPUTE_FID=0,COMPUTE_VBENCH=0" \
    sweep_experiment/sbatch/run_sweep.sbatch
```

- `--time=48:00:00` is this cluster's max; request it for the real run (§4).
- `ROLLOUT_STEPS=8` ≈ ~15 s of video at 32 fps — a fast sanity check. Once the
  output looks right, resubmit with a larger value (e.g. `ROLLOUT_STEPS=155` for
  ~5 min at 32 fps), and consider one job per video to fit the wall-time.
- **`FPS=32`** is set for this high-quality job (§9b). The runner default is 24
  (kept for other experiments), so this override must stay in the `--export`.
- `NO_SAVE_VIDEOS=0` is essential — otherwise metrics are computed but no `.mp4`
  is written and you have nothing to show the user.
- Metrics are off (`COMPUTE_*=0`) because for a "continue my video" demo there
  is no held-out ground truth to score against; turn `COMPUTE_VBENCH=1` on if
  you want no-reference quality numbers.

### Recipe C — AdaSteer (our lightweight method) as an alternative to LoRA

AdaSteer adapts a tiny set of parameters and is our house method; it tends to
preserve subject identity over long rollouts. Same command as Recipe B but:

```
METHOD=delta_a
# replace the LoRA_* / NUM_STEPS / LEARNING_RATE block with:
DELTA_STEPS=10,DELTA_LR=5.0e-3,DELTA_PLACEMENT=adaln
```

To get a **no-fine-tuning baseline** for comparison, run `METHOD=delta_a` with
`DELTA_STEPS=0` (this is our canonical "NOTTA" arm).

---

## 11. Where results land & what to hand back

After a job finishes, results are under `OUTPUT_DIR`:
- generated `.mp4` continuations (when `NO_SAVE_VIDEOS=0`),
- a per-run `summary.json` / results JSON with timings and any computed metrics.

If you produced multiple chunks, merge with
`sweep_experiment/scripts/merge_chunks.py`. For a single 5-video job there is
nothing to merge.

Record-keeping expectations (from `AGENTS.md`): if this run produces numbers the
user might cite, log the raw output under
`sweep_experiment/reports/experiment_outputs/YYYY-MM-DD.md` and push it. For a
one-off user demo this is optional, but keep the habit.

---

## 12. Common failure modes & fixes

| Symptom | Cause | Fix |
|---|---|---|
| Job never starts / rejected | Missing/wrong `--account` | Add `--account=torch_pr_36_mren`. |
| `No module named 'torch'` | `PYTHONHOME`/`PYTHONPATH` not unset after `module load` | Add the `unset` lines (§3), after `conda activate`. |
| Quota/`Disk full` errors | Writing to `/home` or default caches | Point `DATA_DIR`/`OUTPUT_DIR`/`HF_HOME`/`TORCH_HOME`/`TMPDIR` into `/scratch`. |
| Job killed ~mid-run, then restarts | Preemption | Expected on H200; keep `--comment` requeue line and use resumable runners / chunking. |
| Timeout on long rollout | `--time` too low or too many videos/steps in one job | Request the max `--time=48:00:00`; lower `ROLLOUT_STEPS`, run one video per job, or chunk the rollout (§8). |
| Caption guard aborts the run | Empty/duplicate captions in a tiny dataset | Write real captions; set `CAPTION_GUARD_MODE=warn`. |
| No `.mp4` in output | `NO_SAVE_VIDEOS=1` (default) | Set `NO_SAVE_VIDEOS=0`. |
| `git` hangs / `ETIMEDOUT` locally | Local repo is on iCloud (dehydrated) | Do git on the cluster (`git pull`) or via the `/tmp` clone subagent (`AGENTS.md` §5). |

---

## 13. Pre-flight checklist before you hit submit

- [ ] `--account=torch_pr_36_mren` present.
- [ ] All paths under `/scratch/wc3013` (data, output, caches). Nothing in `/home`.
- [ ] Env block includes `module purge/load`, `conda activate`, and the
      `unset PYTHONHOME`/`unset PYTHONPATH` lines.
- [ ] `--gres=gpu:h200:1`, `--time` set (max is `48:00:00`; use it for this generation job), preemption `--comment` set.
- [ ] `slurm_log/` dir is created in-script (`mkdir -p`).
- [ ] `METHOD` and `RUN_ID` set; `--export` starts with `ALL,` and has no spaces.
- [ ] `NO_SAVE_VIDEOS=0` if you need the videos.
- [ ] `ROLLOUT_STEPS` sized for the target duration at **32 fps** (start small, then scale).
- [ ] `FPS=32` in the `--export` (runner defaults to 24) for this high-quality job.
- [ ] `CAPTION_GUARD_MODE=warn` for tiny custom datasets.
- [ ] Submitting from `PROJECT_ROOT` so relative paths resolve.
```
