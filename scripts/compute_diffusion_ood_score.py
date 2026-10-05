#!/usr/bin/env python3
"""Compute per-video diffusion-OOD scores using the LongCat-Video base model.

# ============================================================================
# MODEL / LOSS API AUDIT  --  mirrors the TTA runners verbatim
# ============================================================================
# Pipeline class : LongCatVideoPipeline (built from tokenizer + UMT5
#                  text_encoder + AutoencoderKLWan VAE +
#                  FlowMatchEulerDiscreteScheduler +
#                  LongCatVideoTransformer3DModel) — sourced via
#                  delta_experiment/scripts/common.py::load_longcat_components,
#                  line ~70. Same call signature the TTA runners use.
# Dtype          : torch.bfloat16 throughout (same as run_tinylora.py / run_lora_tta.py
#                  / run_delta_a.py — they all pass dtype=torch.bfloat16 into
#                  load_longcat_components and load_video_frames / encode_prompt).
# Scheduler      : FlowMatchEulerDiscreteScheduler. num_train_timesteps = 1000
#                  (the common.py default; the runners do not override this when
#                  calling compute_flow_matching_loss_conditioned). Valid integer
#                  timesteps span [0, 1000); sigma = timestep / 1000.
# Loss formula   : FLOW-MATCHING (velocity prediction), NOT epsilon prediction:
#                      x_t       = (1 - sigma) * x_0 + sigma * noise
#                      target_v  = noise - x_0
#                      pred_v    = dit(hidden_states=x_t, timestep=sigma*1000,
#                                      encoder_hidden_states=prompt_embeds,
#                                      encoder_attention_mask=prompt_mask,
#                                      num_cond_latents=N_cond)
#                      loss      = F.mse_loss(pred_v[target], target_v)
#                  See compute_flow_matching_loss_conditioned in
#                  delta_experiment/scripts/common.py (~line 439). The TTA
#                  runners optimise this exact loss; we forward-only-evaluate it
#                  here as the OOD score (no backprop, no optimiser step, no
#                  LoRA / Delta-A / TinyLoRA adapters).
# Visible-frame  : CONCAT, not mask. The visible-frame window's latents are
#  conditioning   split into (cond_latents, target_latents) along the temporal
#                  axis at `num_ctx_lat = 1 + (tta_context_frames - 1) // 4`
#                  (VAE temporal scale = 4, same constant baked into all three
#                  TTA runners; see delta_experiment/scripts/run_delta_a.py
#                  line ~864). At loss time the cond portion stays clean and is
#                  prepended to the noised target portion along dim=2:
#                      hidden_states = cat([cond_latents, noisy_target], dim=2)
#                  Per-token timesteps are 0 for the cond tokens and sigma*1000
#                  for the target tokens, and the DiT is told `num_cond_latents
#                  = N_cond` so the attention module treats them as clean
#                  conditioning rather than denoising targets.
# ============================================================================

For each video in --videos-dir, load the visible-frame portion (matching the
TTA runners' --tta-visible-frames convention), encode to latents via the VAE,
split into clean cond / target latents the same way the TTA loop does, add
noise to the target portion at three representative timesteps (low / mid /
high), run the transformer forward in conditioned mode, and report the
per-video flow-matching MSE on the target portion only. Repeat with the
caption empty to get an unconditional score that factors out caption quality.

Outputs a CSV that joins by ``video_id`` with the per-video gains CSV produced
by ``scripts/analyze_per_video_tta_gain.py``. The hypothesis is that ΔPSNR
(TTA gain) correlates positively with diffusion loss: videos the base model
finds unfamiliar (high loss) benefit more from TTA.

CSV schema (one row per ``video_id``, plus the per-timestep loss columns
emitted at the actual --timesteps values):

    video_id,
    diffusion_loss_caption_t{T} for each T in --timesteps,
    diffusion_loss_uncond_t{T}  for each T in --timesteps,
    mean_diffusion_loss_caption,
    mean_diffusion_loss_uncond,
    delta_caption_minus_uncond,
    latent_norm_mean,
    latent_norm_std,
    latent_kurtosis,
    score_norm_caption_t{T} for each T in --timesteps,
    score_norm_uncond_t{T}  for each T in --timesteps,
    mean_score_norm_caption,
    mean_score_norm_uncond,
    n_visible_frames,
    n_gen_target_frames,
    seed

The "caption" and "uncond" columns let downstream analysis factor caption-
conditioning quality out of raw OOD-ness; ``delta_caption_minus_uncond``
flags videos where the caption strongly disagrees with the visuals.

Usage (sbatch wrapper at scripts/sbatch/run_compute_diffusion_ood.sbatch):
    python3 scripts/compute_diffusion_ood_score.py \\
        --checkpoint-dir /scratch/$USER/longcat-video-checkpoints \\
        --videos-dir datasets/panda_1000_480p \\
        --captions-csv datasets/panda_1000_480p/metadata.csv \\
        --output sweep_experiment/reports/per_video_analysis/2026-06-09/diffusion_ood_scores.csv \\
        --tta-visible-frames auto \\
        --timesteps 100,500,900 \\
        --seed 0
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import math
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Path setup — mirror the TTA runners so the LongCat-Video / common imports
# resolve to the SAME modules the TTA loop loads. We add both the repo root
# (for common.py imports inside delta_experiment/scripts) and the
# delta_experiment/scripts dir (so `from common import ...` works).
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "delta_experiment" / "scripts"))
sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Constants — sourced verbatim from
# `sweep_experiment/sbatch/submit_standard_1000v_chunked.sh` env vars and
# matched by all three TTA runners (delta_a / lora_tta / tinylora). Keeping
# them in this script means a `--tta-visible-frames auto` invocation
# reproduces the same TTA-visible window the runners actually used at run
# time, which is what makes the OOD score commensurate with the TTA gains.
# ---------------------------------------------------------------------------
TTA_TOTAL_FRAMES: int = 48       # pre-anchor pixel frames the TTA loop loads
TTA_CONTEXT_FRAMES: int = 14     # leading clean-context portion of that window
GEN_START_FRAME: int = 48        # first frame the diffusion sampler emits
NUM_FRAMES: int = 28             # diffusion window length
NUM_COND_FRAMES: int = 14        # conditioning prefix; (NUM_FRAMES - NUM_COND_FRAMES) are new

AUTO_TTA_VISIBLE_RANGE: Tuple[int, int] = (
    max(0, GEN_START_FRAME - TTA_TOTAL_FRAMES),
    GEN_START_FRAME,
)
# Derived: the diffusion sampler's actual generation region (post-anchor
# frames). Reported in the CSV as `n_gen_target_frames` for documentation;
# it does NOT participate in the OOD loss (which is computed entirely on
# the in-visible-window target portion, per the conditioned flow-matching
# loss formula above).
AUTO_GEN_TARGET_FRAMES: int = NUM_FRAMES - NUM_COND_FRAMES

VAE_TEMPORAL_SCALE: int = 4      # AutoencoderKLWan temporal downsample factor
NUM_TRAIN_TIMESTEPS: int = 1000  # FlowMatchEulerDiscreteScheduler default

DEFAULT_TIMESTEPS: str = "100,500,900"

PROGRESS_EVERY: int = 25


# ---------------------------------------------------------------------------
# Canonical video-id extraction (mirrors analyze_per_video_tta_gain.py /
# extract_video_features_for_tta.py so the OOD CSV is join-key compatible).
# ---------------------------------------------------------------------------
from scripts.caption_utils import (
    canonical_video_id as _canonical_video_id,
    load_resolved_captions_csv,
    resolve_caption_for_clip,
)


def _load_captions_csv(path: Path) -> Dict[str, str]:
    """Return {canonical_video_id -> resolved caption string}."""
    return load_resolved_captions_csv(path, canonical_id=_canonical_video_id)


def _list_video_paths(videos_dir: Path) -> List[Path]:
    """Mirrors extract_video_features_for_tta.list_video_paths."""
    candidates: List[Path] = []
    subdir = videos_dir / "videos"
    if subdir.is_dir():
        for ext in ("*.mp4", "*.avi"):
            candidates.extend(subdir.glob(ext))
    if not candidates:
        for ext in ("*.mp4", "*.avi"):
            candidates.extend(videos_dir.rglob(ext))
    return sorted(candidates, key=lambda p: _canonical_video_id(p.name))


# ---------------------------------------------------------------------------
# Frame-range CLI parsing (matches extract_video_features_for_tta convention)
# ---------------------------------------------------------------------------
def _parse_frame_range_arg(arg: str, default: Tuple[int, int]) -> Tuple[int, int]:
    if not arg or arg.lower() == "auto":
        return default
    if ":" in arg:
        a, b = arg.split(":", 1)
        return int(a), int(b)
    raise argparse.ArgumentTypeError(
        f"--tta-visible-frames must be 'auto' or 'A:B', got {arg!r}"
    )


def _parse_timesteps_arg(arg: str) -> List[int]:
    if not arg:
        raise argparse.ArgumentTypeError("--timesteps cannot be empty")
    parts = [p.strip() for p in arg.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("--timesteps cannot be empty")
    out: List[int] = []
    for p in parts:
        try:
            t = int(p)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"--timesteps entries must be ints, got {p!r}"
            ) from exc
        if t < 0 or t >= NUM_TRAIN_TIMESTEPS:
            raise argparse.ArgumentTypeError(
                f"--timesteps entry {t} out of valid range "
                f"[0, {NUM_TRAIN_TIMESTEPS})"
            )
        out.append(t)
    return out


# ---------------------------------------------------------------------------
# Per-video noise seed — deterministic, decorrelated across videos so the
# CSV is reproducible row-by-row regardless of processing order.
# ---------------------------------------------------------------------------
def _per_video_seed(base_seed: int, video_id: str) -> int:
    h = hashlib.blake2b(video_id.encode("utf-8"), digest_size=8).digest()
    return (base_seed + int.from_bytes(h, "big")) & 0x7FFF_FFFF


# ---------------------------------------------------------------------------
# Output CSV schema helpers
# ---------------------------------------------------------------------------
def _fieldnames(timesteps: List[int]) -> List[str]:
    cols = ["video_id"]
    for t in timesteps:
        cols.append(f"diffusion_loss_caption_t{t}")
    for t in timesteps:
        cols.append(f"diffusion_loss_uncond_t{t}")
    for t in timesteps:
        cols.append(f"score_norm_caption_t{t}")
    for t in timesteps:
        cols.append(f"score_norm_uncond_t{t}")
    cols.extend([
        "mean_diffusion_loss_caption",
        "mean_diffusion_loss_uncond",
        "delta_caption_minus_uncond",
        "latent_norm_mean",
        "latent_norm_std",
        "latent_kurtosis",
        "mean_score_norm_caption",
        "mean_score_norm_uncond",
        "n_visible_frames",
        "n_gen_target_frames",
        "seed",
    ])
    return cols


def _format_value(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return ""
        return f"{v:.6f}"
    return str(v)


def _load_existing_ids(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    if not path.exists():
        return [], []
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames) if reader.fieldnames else []
        rows = [dict(r) for r in reader]
    return rows, fields


# ---------------------------------------------------------------------------
# Aux stats (Tier-1 OOD proxies; no transformer forward involved)
# ---------------------------------------------------------------------------
def _latent_norm_stats(latents) -> Tuple[float, float, float]:
    """Return (norm_mean, norm_std, excess_kurtosis).

    ``latents`` is the full visible-window latent tensor with shape
    [B, C, T, H, W]. The norm we report is the per-spatiotemporal-token L2
    norm along the CHANNEL dimension (dim=1) — i.e. for each (b, t, h, w)
    position we collapse the C-dim into a single scalar ||v||_2, then take
    mean / std across positions. This matches the "per-token L2 norm
    averaged" convention used as a Tier-1 OOD proxy in diffusion OOD
    literature (LongCat's latent codebook has 16 channels at the AutoencoderKLWan
    output, so the per-token norm is informative without further reduction).

    Kurtosis is computed in float32 on the flattened latent tensor using
    scipy.stats.kurtosis(fisher=True) (excess kurtosis: 0 == Gaussian).
    """
    import torch
    from scipy.stats import kurtosis as _kurtosis

    lat = latents.detach().to(torch.float32).cpu()
    # Channel-norm per spatiotemporal position -> [B, T, H, W]
    norm_per_token = lat.norm(dim=1)
    norm_mean = float(norm_per_token.mean().item())
    norm_std = float(norm_per_token.std().item())
    flat = lat.flatten().numpy()
    kurt = float(_kurtosis(flat, fisher=True, bias=False))
    return norm_mean, norm_std, kurt


# ---------------------------------------------------------------------------
# Core per-video OOD computation
# ---------------------------------------------------------------------------
def _compute_one_video(
    *,
    video_path: Path,
    video_id: str,
    caption_raw: str,
    visible_range: Tuple[int, int],
    tta_context_frames: int,
    timesteps: List[int],
    base_seed: int,
    device: str,
    vae,
    dit,
    tokenizer,
    text_encoder,
    dit_patch_t: int,
) -> Dict[str, float]:
    """Forward-only OOD score for a single video. Mirrors
    compute_flow_matching_loss_conditioned() from common.py with sigma /
    noise pinned by the per-video seed for reproducibility.
    """
    import torch
    import torch.nn.functional as F

    from common import (  # noqa: F401  (imported here so import errors surface per-video)
        load_video_frames, encode_video, encode_prompt,
    )

    vs, ve = visible_range
    n_visible = ve - vs
    if n_visible <= 0:
        raise ValueError(f"empty visible range {visible_range}")

    # ---- Decode + VAE encode the visible window -----------------------------
    pixel_frames = load_video_frames(
        str(video_path), n_visible, height=480, width=832,
        start_frame=vs,
    ).to(device, torch.bfloat16)

    with torch.inference_mode():
        all_latents = encode_video(vae, pixel_frames, normalize=True)
    # all_latents shape: [1, C=16, T_lat, H_lat, W_lat]
    T_lat = all_latents.shape[2]
    num_ctx_lat = 1 + (max(1, tta_context_frames) - 1) // VAE_TEMPORAL_SCALE
    num_ctx_lat = max(1, min(num_ctx_lat, T_lat - 1))
    cond_latents = all_latents[:, :, :num_ctx_lat].contiguous()
    target_latents = all_latents[:, :, num_ctx_lat:].contiguous()
    if target_latents.shape[2] <= 0:
        raise ValueError(
            f"target latent split is empty (T_lat={T_lat}, "
            f"num_ctx_lat={num_ctx_lat}); raise --tta-visible-frames or "
            f"lower --tta-context-frames"
        )

    # ---- Latent stats (cheap; computed on the full visible-window latents) -
    norm_mean, norm_std, kurt = _latent_norm_stats(all_latents)

    # Free the pixel tensor before any DiT forward passes
    del pixel_frames

    # ---- Caption + uncond prompt embeddings --------------------------------
    caption_text = resolve_caption_for_clip(caption_raw)
    with torch.inference_mode():
        cap_embeds, cap_mask = encode_prompt(
            tokenizer, text_encoder, caption_text,
            device=device, dtype=torch.bfloat16,
        )
        uncond_embeds, uncond_mask = encode_prompt(
            tokenizer, text_encoder, "",
            device=device, dtype=torch.bfloat16,
        )

    # ---- Per-(timestep, mode) loss -----------------------------------------
    seed = _per_video_seed(base_seed, video_id)
    gen = torch.Generator(device=device)

    B, C, T_cond, H_lat, W_lat = cond_latents.shape
    T_target = target_latents.shape[2]
    T_total = T_cond + T_target
    N_cond = T_cond // dit_patch_t
    N_target = T_target // dit_patch_t
    N_total = T_total // dit_patch_t

    losses: Dict[str, float] = {}

    def _forward_one(timestep_int: int, embeds, mask, noise) -> Tuple[float, float]:
        # Match compute_flow_matching_loss_conditioned() in
        # delta_experiment/scripts/common.py exactly: noise in bfloat16
        # (.randn_like(target_latents) in the reference), arithmetic in
        # bfloat16, only the final MSE is cast to float32 to suppress
        # bf16-MSE underflow on very low loss values.
        sigma = float(timestep_int) / float(NUM_TRAIN_TIMESTEPS)
        sigma_expanded = torch.tensor(
            sigma, device=device, dtype=torch.float32,
        ).view(1, 1, 1, 1, 1)
        noise_bf16 = noise.to(target_latents.dtype)
        noisy_target = (1.0 - sigma_expanded) * target_latents + sigma_expanded * noise_bf16
        hidden_states = torch.cat([cond_latents, noisy_target], dim=2).to(torch.bfloat16)
        timestep = torch.zeros(B, N_total, device=device, dtype=torch.bfloat16)
        timestep[:, N_cond:] = float(timestep_int)
        with torch.inference_mode():
            pred = dit(
                hidden_states=hidden_states,
                timestep=timestep,
                encoder_hidden_states=embeds,
                encoder_attention_mask=mask,
                num_cond_latents=N_cond,
            )
        pred_target = pred[:, :, T_cond:].to(torch.float32)
        velocity_target = (noise_bf16 - target_latents).to(torch.float32)
        mse = float(F.mse_loss(pred_target, velocity_target).item())
        score_norm = float((pred_target ** 2).mean().item())
        return mse, score_norm

    for t in timesteps:
        # Re-seed per (video, timestep) so each timestep gets an independent,
        # reproducible noise sample; reusing the same noise across caption /
        # uncond at a given timestep makes the delta column an apples-to-apples
        # contrast of conditioning effect at that noise level.
        gen.manual_seed(seed + int(t))
        noise = torch.randn(
            target_latents.shape, generator=gen,
            device=device, dtype=torch.float32,
        )
        cap_mse, cap_sn = _forward_one(t, cap_embeds, cap_mask, noise)
        unc_mse, unc_sn = _forward_one(t, uncond_embeds, uncond_mask, noise)
        losses[f"diffusion_loss_caption_t{t}"] = cap_mse
        losses[f"diffusion_loss_uncond_t{t}"] = unc_mse
        losses[f"score_norm_caption_t{t}"] = cap_sn
        losses[f"score_norm_uncond_t{t}"] = unc_sn

    cap_vals = [losses[f"diffusion_loss_caption_t{t}"] for t in timesteps]
    unc_vals = [losses[f"diffusion_loss_uncond_t{t}"] for t in timesteps]
    sn_cap_vals = [losses[f"score_norm_caption_t{t}"] for t in timesteps]
    sn_unc_vals = [losses[f"score_norm_uncond_t{t}"] for t in timesteps]
    mean_cap = float(np.mean(cap_vals)) if cap_vals else float("nan")
    mean_unc = float(np.mean(unc_vals)) if unc_vals else float("nan")
    mean_sn_cap = float(np.mean(sn_cap_vals)) if sn_cap_vals else float("nan")
    mean_sn_unc = float(np.mean(sn_unc_vals)) if sn_unc_vals else float("nan")

    out: Dict[str, float] = {
        "video_id": video_id,
        **losses,
        "mean_diffusion_loss_caption": mean_cap,
        "mean_diffusion_loss_uncond": mean_unc,
        "delta_caption_minus_uncond": mean_cap - mean_unc,
        "latent_norm_mean": norm_mean,
        "latent_norm_std": norm_std,
        "latent_kurtosis": kurt,
        "mean_score_norm_caption": mean_sn_cap,
        "mean_score_norm_uncond": mean_sn_unc,
        "n_visible_frames": int(n_visible),
        "n_gen_target_frames": int(AUTO_GEN_TARGET_FRAMES),
        "seed": int(seed),
    }
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--checkpoint-dir", type=str, required=True,
                    help="LongCat-Video checkpoint directory (subdirs: "
                         "tokenizer, text_encoder, vae, scheduler, dit).")
    ap.add_argument("--videos-dir", type=Path, required=True,
                    help="Dataset root containing videos/ subdir or *.mp4 directly.")
    ap.add_argument("--captions-csv", type=Path, required=True,
                    help="Panda metadata.csv (or UCF-style: filename,text).")
    ap.add_argument("--output", type=Path, required=True,
                    help="Per-video OOD-score CSV. Stream-written after each "
                         "video so partial runs survive job crashes / "
                         "pre-emption. With --resume, existing rows are kept "
                         "and only missing video_ids are computed.")
    ap.add_argument(
        "--tta-visible-frames", type=str, default="auto",
        help="'auto' (resolves to "
             f"{AUTO_TTA_VISIBLE_RANGE[0]}:{AUTO_TTA_VISIBLE_RANGE[1]}, "
             "matching the panda_1000v_standard TTA-runner window "
             "[max(0, GEN_START_FRAME - TTA_TOTAL_FRAMES) : GEN_START_FRAME]) "
             "or an explicit 'A:B' python-slice range. Determines which pixel "
             "frames are decoded + VAE-encoded for the OOD-score loss.",
    )
    ap.add_argument(
        "--tta-context-frames", type=int, default=TTA_CONTEXT_FRAMES,
        help=f"Number of leading pixel frames in the visible window treated "
             f"as CLEAN conditioning context (timestep=0; default {TTA_CONTEXT_FRAMES} "
             "matches NUM_COND_FRAMES, i.e. the TTA runners' "
             "tta_context_frames default). The remaining visible-window "
             "frames are the noised target portion the OOD loss is computed on.",
    )
    ap.add_argument(
        "--timesteps", type=str, default=DEFAULT_TIMESTEPS,
        help=f"Comma-separated integer timesteps in [0, {NUM_TRAIN_TIMESTEPS}). "
             "Default '100,500,900' spans low / mid / high noise regimes for "
             f"the FlowMatchEulerDiscreteScheduler (sigma = t / {NUM_TRAIN_TIMESTEPS}).",
    )
    ap.add_argument("--seed", type=int, default=0,
                    help="Base seed for the per-video noise sampler. The actual "
                         "noise seed per video is base + blake2b(video_id), so "
                         "videos are reproducible independently of each other "
                         "and of processing order.")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--max-videos", type=int, default=0,
                    help="0 = all. Otherwise process the first N (by canonical id) "
                         "for smoke-testing.")
    ap.add_argument("--resume", action="store_true",
                    help="If the output CSV already exists, skip videos already "
                         "in it. Same idempotent pattern as build_panda_segment_pool.py.")
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    visible_range = _parse_frame_range_arg(
        args.tta_visible_frames, default=AUTO_TTA_VISIBLE_RANGE,
    )
    timesteps = _parse_timesteps_arg(args.timesteps)
    n_visible = visible_range[1] - visible_range[0]

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Resolve cond / target latent split sizes for the banner so the user can
    # eyeball the split before the model loads.
    num_ctx_lat_preview = 1 + (max(1, args.tta_context_frames) - 1) // VAE_TEMPORAL_SCALE
    n_visible_lat_preview = 1 + (max(1, n_visible) - 1) // VAE_TEMPORAL_SCALE

    # Load existing rows for --resume
    existing_rows, existing_fields = _load_existing_ids(args.output)
    fieldnames = _fieldnames(timesteps)
    if existing_rows and args.resume:
        # Refuse to resume across an incompatible --timesteps change, since
        # that would silently produce a CSV with two disjoint loss-column
        # families (existing rows have only the old t* columns, new rows have
        # only the new t* columns). Force the user to either re-run from
        # scratch or pass the same --timesteps the existing CSV was built with.
        existing_ts_cols = sorted(
            c for c in existing_fields
            if c.startswith("diffusion_loss_caption_t")
        )
        new_ts_cols = sorted(
            f"diffusion_loss_caption_t{t}" for t in timesteps
        )
        if existing_ts_cols and existing_ts_cols != new_ts_cols:
            print(
                f"[error] --resume schema mismatch: existing CSV has "
                f"timestep columns {existing_ts_cols} but --timesteps={timesteps} "
                f"would produce {new_ts_cols}. Re-run without --resume to "
                f"overwrite, or pass --timesteps matching the existing CSV.",
                file=sys.stderr,
            )
            return 2
        # Preserve any extra historical columns to avoid silently dropping
        # data when the schema evolves in a compatible direction.
        for extra in existing_fields:
            if extra and extra not in fieldnames:
                fieldnames.append(extra)

    existing_ids = {
        (r.get("video_id") or "").strip()
        for r in existing_rows if (r.get("video_id") or "").strip()
    } if args.resume else set()

    print("=" * 78)
    print("LongCat-Video diffusion-OOD score (per-video, base model, no adapters)")
    print("=" * 78)
    print(f"Checkpoint dir       : {args.checkpoint_dir}")
    print(f"Videos dir           : {args.videos_dir}")
    print(f"Captions CSV         : {args.captions_csv}")
    print(f"Output               : {args.output}")
    print(f"TTA-visible frames   : {visible_range[0]}:{visible_range[1]}  "
          f"(n={n_visible} pixel frames -> ~{n_visible_lat_preview} latent frames)")
    print(f"TTA-context frames   : {args.tta_context_frames}  "
          f"(-> ~{num_ctx_lat_preview} clean cond latents)")
    print(f"Gen-target frames    : {AUTO_GEN_TARGET_FRAMES}  "
          "(diffusion sampler's post-anchor output; documented in CSV, "
          "not part of OOD loss)")
    print(f"Timesteps            : {timesteps}  "
          f"(sigma = t / {NUM_TRAIN_TIMESTEPS}: "
          f"{[round(t / NUM_TRAIN_TIMESTEPS, 3) for t in timesteps]})")
    print(f"Loss formula         : flow-matching v-prediction MSE on target portion only "
          f"(num_cond_latents = N_cond)")
    print(f"Model dtype          : torch.bfloat16")
    print(f"Device               : {args.device}")
    print(f"Base seed            : {args.seed}")
    print(f"Resume               : {args.resume}  "
          f"({len(existing_ids)} existing rows in --output)")
    print("=" * 78)

    # ---- Enumerate videos + captions --------------------------------------
    captions_by_id = _load_captions_csv(args.captions_csv)
    video_paths = _list_video_paths(args.videos_dir)
    if args.max_videos and args.max_videos > 0:
        video_paths = video_paths[: args.max_videos]
    if not video_paths:
        print(f"[error] no video files found under {args.videos_dir}",
              file=sys.stderr)
        return 2

    # Filter against existing IDs when resuming
    todo: List[Path] = []
    for vp in video_paths:
        vid = _canonical_video_id(vp.name)
        if vid in existing_ids:
            continue
        todo.append(vp)
    n_skip = len(video_paths) - len(todo)

    print(f"Videos discovered    : {len(video_paths)}")
    print(f"Captions loaded      : {len(captions_by_id)}")
    print(f"Already in --output  : {n_skip}  (skipped via --resume)")
    print(f"Videos to process    : {len(todo)}")
    print("=" * 78)

    if not todo:
        print("Nothing to do.")
        return 0

    # ---- Load LongCat-Video base components (no adapters) ----------------
    print("\nLoading LongCat-Video base model components...")
    try:
        from common import load_longcat_components
    except ImportError as exc:
        print(f"[error] failed to import common.load_longcat_components: {exc}",
              file=sys.stderr)
        traceback.print_exc()
        return 2
    import torch
    components = load_longcat_components(
        args.checkpoint_dir, device=args.device, dtype=torch.bfloat16,
    )
    dit = components["dit"]
    vae = components["vae"]
    tokenizer = components["tokenizer"]
    text_encoder = components["text_encoder"]
    dit.eval()
    vae.eval()
    text_encoder.eval()
    for p in dit.parameters():
        p.requires_grad_(False)
    for p in vae.parameters():
        p.requires_grad_(False)
    for p in text_encoder.parameters():
        p.requires_grad_(False)

    try:
        dit_patch_t = int(dit.config.patch_size[0])
    except (AttributeError, IndexError, TypeError) as exc:
        print(f"[error] could not resolve dit.config.patch_size[0]: {exc}",
              file=sys.stderr)
        return 2
    print(f"  DiT patch_t        : {dit_patch_t}")
    print(f"  DiT blocks         : {len(dit.blocks)}")
    print()

    # ---- Open CSV in stream mode (append when resuming, else truncate) ---
    # --resume + existing rows -> append (no header rewrite, preserve history).
    # Anything else -> truncate + write header. This makes a re-run without
    # --resume an explicit "start over" gesture so accidental schema drift
    # (e.g. changing --timesteps) does not silently produce a mixed CSV.
    appending = args.resume and bool(existing_rows) and args.output.exists()
    out_handle = args.output.open(
        "a" if appending else "w", newline="", encoding="utf-8",
    )
    writer = csv.DictWriter(out_handle, fieldnames=fieldnames)
    if not appending:
        writer.writeheader()
        out_handle.flush()

    # ---- Per-video loop ---------------------------------------------------
    n_done = 0
    n_errored = 0
    t0 = time.time()
    last_print_t = t0
    first_row: Optional[Dict[str, float]] = None

    for v_idx, vp in enumerate(todo):
        vid = _canonical_video_id(vp.name)
        cap = captions_by_id.get(vid, "")
        try:
            row = _compute_one_video(
                video_path=vp, video_id=vid, caption_raw=cap,
                visible_range=visible_range,
                tta_context_frames=args.tta_context_frames,
                timesteps=timesteps, base_seed=args.seed,
                device=args.device,
                vae=vae, dit=dit, tokenizer=tokenizer,
                text_encoder=text_encoder, dit_patch_t=dit_patch_t,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[error] {vp.name}: {exc}", file=sys.stderr)
            traceback.print_exc()
            n_errored += 1
            # Free GPU memory after an error
            try:
                import torch as _torch
                _torch.cuda.empty_cache()
            except Exception:
                pass
            continue
        writer.writerow({k: _format_value(row.get(k)) for k in fieldnames})
        out_handle.flush()
        n_done += 1
        if first_row is None:
            first_row = row
        try:
            import torch as _torch
            _torch.cuda.empty_cache()
        except Exception:
            pass

        if (v_idx + 1) % PROGRESS_EVERY == 0 or (v_idx + 1) == len(todo):
            dt = time.time() - last_print_t
            last_print_t = time.time()
            elapsed = time.time() - t0
            rate = (v_idx + 1) / max(elapsed, 1e-6)
            eta = max(0.0, (len(todo) - (v_idx + 1)) / max(rate, 1e-9))
            print(
                f"  [{v_idx + 1}/{len(todo)}] done={n_done} err={n_errored} "
                f"(+{dt:.1f}s; {rate:.2f} vid/s; ETA {eta / 60:.1f}m) "
                f"vid={vid}",
                flush=True,
            )

    out_handle.close()

    dt_total = time.time() - t0
    print()
    print("=" * 78)
    print(f"Wrote {args.output}")
    print(f"  new rows this run     : {n_done}")
    print(f"  errored               : {n_errored}")
    print(f"  skipped (already-done): {n_skip}")
    print(f"  wall time             : {dt_total:.1f}s  "
          f"({dt_total / max(n_done, 1):.1f}s / video)")
    if first_row is not None:
        print()
        print("Example row (first successfully-processed video):")
        for k in fieldnames:
            v = first_row.get(k)
            v_s = _format_value(v) if not isinstance(v, str) else v
            print(f"  {k:34s} = {v_s}")
    print("=" * 78)
    return 0 if n_errored == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
