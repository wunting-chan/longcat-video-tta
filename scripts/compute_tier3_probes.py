#!/usr/bin/env python3
"""Per-video Tier-3 mini-TTA probes (H-T3-1 grad_norm_θ0 + H-T3-2
single_step_loss_drop) against a fresh LoRA r=8 adapter on LongCat-Video.

# ============================================================================
# WHAT THIS SCRIPT MEASURES  (see PLAN_gating_experiment_2026-06-11.md §2.4 +
# HYPOTHESES_per_video_tta_suitability_2026-06-09.md §H-T3-1 / §H-T3-2)
# ============================================================================
# For each video v in --videos-dir, and for each fixed timestep t in
# --timesteps (default 100,500,900 — same schedule as
# scripts/compute_diffusion_ood_score.py):
#
#   1. Reset a freshly-injected LoRA r=8 adapter to its zero-output init
#      (lora_up = 0, lora_down ~ Kaiming) and (re)create a fresh AdamW
#      over its parameters.  This is the per-video, per-timestep "fresh
#      probe" guarantee: NO carry-over from one video to the next, and NO
#      carry-over from one timestep to the next within a single video, so
#      each measurement is a pure probe of the local TTA loss-surface
#      geometry at the unadapted weights.
#
#   2. With the deterministic noise drawn from blake2b(video_id) + t (so
#      this row is reproducible regardless of processing order, mirroring
#      compute_diffusion_ood_score.py's _per_video_seed convention),
#      compute the visible-window conditioning-aware flow-matching loss
#      L_t0 = MSE(pred_v[target], noise - target_latents) WITH GRAD on
#      (inference_mode disabled so the backward works).
#
#   3. Backward, then record the L2 norm of the gradient over only the
#      LoRA-tunable parameters (the {lora_down, lora_up} weights of every
#      injected adapter):
#          grad_norm_lora_t{t} = || cat([p.grad.flatten() for p in lora_params]) ||_2
#      This is the H-T3-1 grad_norm_θ0 feature for this video at this t.
#
#   4. Take ONE AdamW step at the headline LORA_R8_TTA learning rate (5e-5
#      from sweep_experiment/sbatch/submit_standard_1000v_chunked.sh line
#      ~198: LORA_RANK=8, LORA_ALPHA=16, NUM_STEPS=10, LEARNING_RATE=5e-5;
#      overridable via --lora-lr / the LORA_LR env var in the sbatch
#      wrapper).  No warmup — this is a *single-step* probe, not a TTA
#      run, so the headline 3-step warmup schedule does not apply; using
#      the full LR matches the "what does ONE Adam step do to this video"
#      question of H-T3-2.
#
#   5. Re-evaluate the loss at the same (t, noise) under the now-adapted
#      LoRA weights to get L_t1 (no_grad).  Record:
#          loss_drop_pct_t{t} = (L_t0 − L_t1) / max(L_t0, 1e-6)
#
# The probe is "method-relevant" because the LoRA injection + optimiser
# setup (rank=8, alpha=16, target qkv+proj, target all DiT blocks,
# AdamW(weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8)) mirror
# lora_experiment/scripts/run_lora_tta.py::inject_lora_into_dit /
# finetune_lora_on_conditioning verbatim.  See AUDIT BLOCK below.
#
# # ===========================================================================
# # AUDIT — LoRA setup mirrors lora_experiment/scripts/run_lora_tta.py
# # ===========================================================================
# # Rank / alpha / targets : matches headline LORA_R8_TTA recipe (see
# #     sweep_experiment/sbatch/submit_standard_1000v_chunked.sh §2 ADA
# #     block, around line 196:
# #       LORA_RANK=8, LORA_ALPHA=16, LORA_TARGET_BLOCKS=all,
# #       NUM_STEPS=10, LEARNING_RATE=5.0e-5, WARMUP_STEPS=3,
# #       WEIGHT_DECAY=0.01, MAX_GRAD_NORM=10.0, TARGET_FFN=0
# #     ).  Defaults below intentionally match this row so the probe
# #     measures the *exact* LoRA-r8 family the paper deploys.
# # Loss formula           : flow-matching v-prediction at fixed (sigma, noise)
# #     using the same per-token timestep + num_cond_latents convention as
# #     compute_flow_matching_loss_conditioned (delta_experiment/scripts/
# #     common.py ~line 439) and compute_diffusion_ood_score.py.  We DO NOT
# #     re-use compute_flow_matching_loss_conditioned itself because its
# #     internal sigma sample is random per call; we need a deterministic
# #     (sigma, noise) per (video, t) so loss_t0 and loss_t1 are an
# #     apples-to-apples drop and not a sigma-resample artefact.
# # Inference mode         : MUST be off during loss_t0 / backward (we need
# #     grads); MAY be on during loss_t1 (re-eval after the Adam step).  The
# #     base DiT / VAE / text_encoder parameters are frozen via
# #     requires_grad_(False) — only the LoRA params have requires_grad=True.
# # Eval / train flips     : base DiT in eval() throughout (no dropout / BN
# #     drift); each video's probe loop sets dit.train() before the loss_t0
# #     forward (matches run_lora_tta.py's dit.train() at the top of
# #     finetune_lora_on_conditioning) and dit.eval() before loss_t1 (no-op
# #     for our DiT but kept for parity).
# # Reset                  : reset_lora_weights() is called at the START of
# #     every (video, timestep) probe, AND once more at the end of each
# #     video as a defensive measure before the next video's first
# #     iteration.  Optimiser is fully re-created per (video, timestep) too
# #     so Adam's running moments do not bleed across probes.
# # ===========================================================================
#
# CSV schema (one row per video_id; per-timestep columns are emitted for
# each value in --timesteps so the schema is reproducible from the CLI
# args alone, mirroring compute_diffusion_ood_score.py):
#
#     video_id,
#     grad_norm_lora_t{T},          # H-T3-1, per timestep
#     mean_grad_norm_lora,          # mean across --timesteps
#     loss_drop_pct_t{T},           # H-T3-2, per timestep
#     mean_loss_drop_pct,           # mean across --timesteps
#     loss_t0_t{T}, loss_t1_t{T},   # raw losses for audit
#     n_visible_frames, n_gen_target_frames,
#     lora_rank, lora_alpha, lora_lr, lora_targets, seed
#
# Output filename default `tier3_probe_features.csv` matches the gating
# plan §3.1 expected output filename so downstream join keys line up.

Usage (see scripts/sbatch/run_compute_tier3_probes.sbatch for the
canonical sbatch invocation):

    python3 scripts/compute_tier3_probes.py \\
        --checkpoint-dir /scratch/$USER/longcat-video-checkpoints \\
        --videos-dir datasets/panda_1000_480p \\
        --captions-csv datasets/panda_1000_480p/metadata.csv \\
        --output sweep_experiment/reports/per_video_analysis/2026-06-09/tier3_probe_features.csv \\
        --tta-visible-frames auto \\
        --timesteps 100,500,900 \\
        --lora-rank 8 --lora-alpha 16 --lora-lr 5.0e-5 \\
        --seed 0
"""
from __future__ import annotations

import argparse
import contextlib
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
# Path setup — mirror compute_diffusion_ood_score.py so the LongCat-Video
# imports + delta_experiment.common imports resolve to the SAME modules
# the TTA runners use. Also expose lora_experiment/scripts so the LoRA
# injection helpers from run_lora_tta import cleanly.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "delta_experiment" / "scripts"))
sys.path.insert(0, str(_REPO_ROOT / "lora_experiment" / "scripts"))
sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Constants — mirror compute_diffusion_ood_score.py so the probe uses the
# same visible-window split and noise schedule as the OOD scorer (so the
# joined per_video_gains.csv + diffusion_ood_scores.csv + Tier-3 probe CSV
# rows reference the same (video_id, slice) tuple).
# ---------------------------------------------------------------------------
TTA_TOTAL_FRAMES: int = 48
TTA_CONTEXT_FRAMES: int = 14
GEN_START_FRAME: int = 48
NUM_FRAMES: int = 28
NUM_COND_FRAMES: int = 14

AUTO_TTA_VISIBLE_RANGE: Tuple[int, int] = (
    max(0, GEN_START_FRAME - TTA_TOTAL_FRAMES),
    GEN_START_FRAME,
)
AUTO_GEN_TARGET_FRAMES: int = NUM_FRAMES - NUM_COND_FRAMES

VAE_TEMPORAL_SCALE: int = 4
NUM_TRAIN_TIMESTEPS: int = 1000

DEFAULT_TIMESTEPS: str = "100,500,900"

# Headline LORA_R8_TTA recipe — sourced verbatim from
# sweep_experiment/sbatch/submit_standard_1000v_chunked.sh §2 around line
# 196.  These defaults make `--lora-rank 8 --lora-alpha 16 --lora-lr 5e-5
# --lora-targets qkv,proj` the canonical probe configuration.
DEFAULT_LORA_RANK: int = 8
DEFAULT_LORA_ALPHA: float = 16.0
DEFAULT_LORA_LR: float = 5.0e-5
DEFAULT_LORA_WEIGHT_DECAY: float = 0.01
DEFAULT_LORA_TARGETS: str = "qkv,proj"
DEFAULT_LORA_TARGET_BLOCKS: str = "all"
DEFAULT_LORA_TARGET_FFN: bool = False

PROGRESS_EVERY: int = 25


# ---------------------------------------------------------------------------
# Canonical video-id extraction (mirrors analyze_per_video_tta_gain.py /
# extract_video_features_for_tta.py / compute_diffusion_ood_score.py).
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
    candidates: List[Path] = []
    subdir = videos_dir / "videos"
    if subdir.is_dir():
        for ext in ("*.mp4", "*.avi"):
            candidates.extend(subdir.glob(ext))
    if not candidates:
        for ext in ("*.mp4", "*.avi"):
            candidates.extend(videos_dir.rglob(ext))
    return sorted(candidates, key=lambda p: _canonical_video_id(p.name))


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


def _parse_targets_arg(arg: str) -> List[str]:
    """Mirror run_lora_tta.py's --target-modules CLI parsing."""
    parts = [p.strip().lower() for p in (arg or "").split(",") if p.strip()]
    valid = {"qkv", "proj"}
    bad = [p for p in parts if p not in valid]
    if bad:
        raise argparse.ArgumentTypeError(
            f"--lora-targets entries must be in {sorted(valid)}; got {bad}"
        )
    if not parts:
        raise argparse.ArgumentTypeError("--lora-targets cannot be empty")
    return parts


def _per_video_seed(base_seed: int, video_id: str) -> int:
    h = hashlib.blake2b(video_id.encode("utf-8"), digest_size=8).digest()
    return (base_seed + int.from_bytes(h, "big")) & 0x7FFF_FFFF


# ---------------------------------------------------------------------------
# Output CSV schema helpers
# ---------------------------------------------------------------------------
def _fieldnames(timesteps: List[int]) -> List[str]:
    cols = ["video_id"]
    for t in timesteps:
        cols.append(f"grad_norm_lora_t{t}")
    cols.append("mean_grad_norm_lora")
    for t in timesteps:
        cols.append(f"loss_drop_pct_t{t}")
    cols.append("mean_loss_drop_pct")
    for t in timesteps:
        cols.append(f"loss_t0_t{t}")
        cols.append(f"loss_t1_t{t}")
    cols.extend([
        "n_visible_frames",
        "n_gen_target_frames",
        "lora_rank",
        "lora_alpha",
        "lora_lr",
        "lora_targets",
        "seed",
    ])
    return cols


def _format_value(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return ""
        return f"{v:.6g}"
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
# Per-block gradient checkpointing for the LongCat-Video DiT.
#
# The probe needs gradients (H-T3-1 grad_norm_lora) so all 48 LongCat-Video
# transformer blocks' activations are retained across the forward pass.  At
# the panda_1000v_standard sequence length × bf16 × 48 blocks, that exceeds
# the H200's 140 GB budget and the overnight Phase-0 run OOM'd on every
# video at FFN.w2 (the last per-block tensor allocated in the forward path):
#
#     File "LongCat-Video/longcat_video/modules/longcat_video_dit.py", l. 343
#         block_outputs = block(...)
#     File "LongCat-Video/longcat_video/modules/blocks.py", line 39
#         return self.w2(F.silu(self.w1(x)) * self.w3(x))
#     torch.OutOfMemoryError: CUDA out of memory.
#
# Per-block torch.utils.checkpoint trades one extra forward pass for ~Nx
# less peak activation memory; estimated ~13 GB instead of ~140 GB on the
# H200 panda_1000v_standard probe (48 blocks × ~3 GB activation + one
# extra ~10 GB forward at the end of the backward pass).
#
# Implementation: monkey-patch each ``dit.blocks[i].forward`` for the
# duration of a context manager.  Confined to this probe script -- the
# production DiT module ``longcat_video_dit.py`` is NOT modified, so all
# inference / training code paths are unaffected.
#
# Semantic invariance: ``torch.utils.checkpoint(use_reentrant=False)``
# produces bit-identical loss tensors on the forward pass and gradients
# that match the non-checkpointed path up to the usual recompute
# numerical roundoff (well below the precision at which we report the
# H-T3-1 grad-norm scalar).
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def _dit_blocks_gradient_checkpoint(dit):
    """Context-manage per-block torch.utils.checkpoint over ``dit.blocks``.

    Each ``dit.blocks[i].forward`` is wrapped in
    ``torch.utils.checkpoint.checkpoint(..., use_reentrant=False, ...)``
    on entry and restored to the original bound method on exit (including
    on exception).  ``use_reentrant=False`` is required because the block
    inputs (``hidden_states``, etc.) are detached upstream and therefore
    do NOT have ``requires_grad=True`` -- only the LoRA params inside the
    block do.  The non-reentrant path also handles **kwargs cleanly,
    which the LongCat block forwards rely on.
    """
    import torch.utils.checkpoint as _ckpt

    blocks = list(dit.blocks)
    originals = [b.forward for b in blocks]

    def _make_wrapped(orig_forward):
        def _wrapped(*args, **kwargs):
            return _ckpt.checkpoint(
                orig_forward, *args, use_reentrant=False, **kwargs,
            )
        return _wrapped

    for b, orig in zip(blocks, originals):
        b.forward = _make_wrapped(orig)
    try:
        yield
    finally:
        for b, orig in zip(blocks, originals):
            b.forward = orig


# ---------------------------------------------------------------------------
# Fixed-(sigma, noise) forward path. We re-implement
# compute_flow_matching_loss_conditioned with grad enabled and DETERMINISTIC
# (sigma, noise) so loss_t0 / loss_t1 at a given (video, t) measure the
# adapter's effect cleanly rather than a sigma-resample artefact. The
# tensor shapes / concat order / per-token timestep convention match
# delta_experiment/scripts/common.py::compute_flow_matching_loss_conditioned
# verbatim.
# ---------------------------------------------------------------------------
def _fixed_conditioned_loss(
    *,
    dit,
    cond_latents,
    target_latents,
    prompt_embeds,
    prompt_mask,
    timestep_int: int,
    noise,
    dit_patch_t: int,
    device: str,
    use_no_grad: bool,
):
    """Conditioning-aware flow-matching loss at a fixed (timestep, noise).

    Returns the SCALAR loss tensor.  Caller decides whether to backward()
    or only read .item() (controlled via use_no_grad).
    """
    import torch
    import torch.nn.functional as F

    B, C, T_cond, H_lat, W_lat = cond_latents.shape
    T_target = target_latents.shape[2]
    T_total = T_cond + T_target
    N_cond = T_cond // dit_patch_t
    N_target = T_target // dit_patch_t
    N_total = T_total // dit_patch_t

    sigma = float(timestep_int) / float(NUM_TRAIN_TIMESTEPS)
    sigma_expanded = torch.tensor(
        sigma, device=device, dtype=torch.float32,
    ).view(1, 1, 1, 1, 1)
    noise_bf16 = noise.to(target_latents.dtype)
    noisy_target = (1.0 - sigma_expanded) * target_latents + sigma_expanded * noise_bf16
    hidden_states = torch.cat([cond_latents, noisy_target], dim=2).to(torch.bfloat16)
    timestep = torch.zeros(B, N_total, device=device, dtype=torch.bfloat16)
    timestep[:, N_cond:] = float(timestep_int)

    def _forward():
        pred = dit(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            encoder_attention_mask=prompt_mask,
            num_cond_latents=N_cond,
        )
        pred_target = pred[:, :, T_cond:].to(torch.float32)
        velocity_target = (noise_bf16 - target_latents).to(torch.float32)
        return F.mse_loss(pred_target, velocity_target)

    if use_no_grad:
        with torch.no_grad():
            return _forward()
    # Per-block gradient checkpointing keeps peak activation memory to
    # ~13 GB instead of the ~140 GB the non-checkpointed forward needs
    # on the H200 (the forward retains all 48 LongCat-Video blocks'
    # activations until the backward).  See
    # ``_dit_blocks_gradient_checkpoint`` docstring for the full
    # rationale + semantic-invariance argument.
    with _dit_blocks_gradient_checkpoint(dit):
        return _forward()


# ---------------------------------------------------------------------------
# Per-video Tier-3 probe
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
    lora_modules,
    lora_params,
    reset_lora_fn,
    lora_rank: int,
    lora_alpha: float,
    lora_lr: float,
    lora_weight_decay: float,
    lora_targets: str,
) -> Dict[str, float]:
    """One forward + one backward + one Adam step + one re-eval per
    (video, timestep), with the LoRA adapter reset at the start of every
    iteration so each measurement is a fresh probe.
    """
    import torch
    from torch.optim import AdamW

    from common import (  # noqa: F401  (per-video import so import errors surface here)
        load_video_frames, encode_video, encode_prompt,
    )

    vs, ve = visible_range
    n_visible = ve - vs
    if n_visible <= 0:
        raise ValueError(f"empty visible range {visible_range}")

    pixel_frames = load_video_frames(
        str(video_path), n_visible, height=480, width=832,
        start_frame=vs,
    ).to(device, torch.bfloat16)

    with torch.inference_mode():
        all_latents = encode_video(vae, pixel_frames, normalize=True)
    T_lat = all_latents.shape[2]
    num_ctx_lat = 1 + (max(1, tta_context_frames) - 1) // VAE_TEMPORAL_SCALE
    num_ctx_lat = max(1, min(num_ctx_lat, T_lat - 1))
    # detach + clone so the autograd graph the latents carry from the VAE
    # cache does NOT chain back into the dit-grad we are about to compute;
    # only the LoRA params should have requires_grad=True from this point.
    cond_latents = all_latents[:, :, :num_ctx_lat].contiguous().detach()
    target_latents = all_latents[:, :, num_ctx_lat:].contiguous().detach()
    if target_latents.shape[2] <= 0:
        raise ValueError(
            f"target latent split is empty (T_lat={T_lat}, "
            f"num_ctx_lat={num_ctx_lat}); raise --tta-visible-frames or "
            f"lower --tta-context-frames"
        )

    del pixel_frames

    caption_text = resolve_caption_for_clip(caption_raw)
    with torch.inference_mode():
        cap_embeds, cap_mask = encode_prompt(
            tokenizer, text_encoder, caption_text,
            device=device, dtype=torch.bfloat16,
        )

    seed = _per_video_seed(base_seed, video_id)
    gen = torch.Generator(device=device)

    grad_norms: Dict[int, float] = {}
    losses_t0: Dict[int, float] = {}
    losses_t1: Dict[int, float] = {}
    loss_drops: Dict[int, float] = {}

    for t in timesteps:
        # ---- Fresh probe per (video, t): reset adapter + optimiser ----
        reset_lora_fn(lora_modules)
        optimizer = AdamW(
            lora_params,
            lr=lora_lr,
            betas=(0.9, 0.999),
            weight_decay=lora_weight_decay,
            eps=1e-8,
        )
        optimizer.zero_grad(set_to_none=True)

        # Deterministic noise per (video, t).
        gen.manual_seed(seed + int(t))
        noise = torch.randn(
            target_latents.shape, generator=gen,
            device=device, dtype=torch.float32,
        )

        # ---- loss_t0 + backward + grad-norm (H-T3-1) ----
        dit.train()
        loss_t0_tensor = _fixed_conditioned_loss(
            dit=dit,
            cond_latents=cond_latents,
            target_latents=target_latents,
            prompt_embeds=cap_embeds,
            prompt_mask=cap_mask,
            timestep_int=t,
            noise=noise,
            dit_patch_t=dit_patch_t,
            device=device,
            use_no_grad=False,
        )
        loss_t0 = float(loss_t0_tensor.item())
        loss_t0_tensor.backward()

        # ||grad||_2 over only the LoRA params (no clip; clip_grad_norm_
        # with max_norm=inf returns the unclipped total norm — matches
        # run_lora_tta.py::finetune_lora_on_conditioning's grad-norm
        # logging convention).
        grad_total_norm = torch.nn.utils.clip_grad_norm_(
            lora_params, float("inf"),
        )
        grad_norm_value = float(grad_total_norm.item())

        # ---- ONE Adam step (H-T3-2) ----
        optimizer.step()

        # ---- loss_t1 (no_grad, same (t, noise)) ----
        dit.eval()
        loss_t1_tensor = _fixed_conditioned_loss(
            dit=dit,
            cond_latents=cond_latents,
            target_latents=target_latents,
            prompt_embeds=cap_embeds,
            prompt_mask=cap_mask,
            timestep_int=t,
            noise=noise,
            dit_patch_t=dit_patch_t,
            device=device,
            use_no_grad=True,
        )
        loss_t1 = float(loss_t1_tensor.item())

        grad_norms[t] = grad_norm_value
        losses_t0[t] = loss_t0
        losses_t1[t] = loss_t1
        loss_drops[t] = (loss_t0 - loss_t1) / max(loss_t0, 1e-6)

        # Free the per-step state before the next timestep iteration
        del optimizer, loss_t0_tensor, loss_t1_tensor, noise
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    # Defensive: leave the adapter in the zero-output state before the
    # next video's first timestep iteration (which will also reset, but
    # this protects against partial-failure leakage).
    reset_lora_fn(lora_modules)

    out: Dict[str, float] = {
        "video_id": video_id,
    }
    for t in timesteps:
        out[f"grad_norm_lora_t{t}"] = grad_norms[t]
    out["mean_grad_norm_lora"] = (
        float(np.mean(list(grad_norms.values()))) if grad_norms else float("nan")
    )
    for t in timesteps:
        out[f"loss_drop_pct_t{t}"] = loss_drops[t]
    out["mean_loss_drop_pct"] = (
        float(np.mean(list(loss_drops.values()))) if loss_drops else float("nan")
    )
    for t in timesteps:
        out[f"loss_t0_t{t}"] = losses_t0[t]
        out[f"loss_t1_t{t}"] = losses_t1[t]
    out["n_visible_frames"] = int(n_visible)
    out["n_gen_target_frames"] = int(AUTO_GEN_TARGET_FRAMES)
    out["lora_rank"] = int(lora_rank)
    out["lora_alpha"] = float(lora_alpha)
    out["lora_lr"] = float(lora_lr)
    out["lora_targets"] = lora_targets
    out["seed"] = int(seed)
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
                    help="LongCat-Video checkpoint directory.")
    ap.add_argument("--videos-dir", type=Path, required=True,
                    help="Dataset root containing videos/ subdir or *.mp4 directly.")
    ap.add_argument("--captions-csv", type=Path, required=True,
                    help="Panda metadata.csv (filename,caption) or "
                         "UCF-style (filename,text).")
    ap.add_argument("--output", type=Path, required=True,
                    help="Per-video Tier-3 probe CSV. Stream-written after "
                         "each video so partial runs survive job crashes / "
                         "pre-emption; with --resume, existing rows are "
                         "kept and only missing video_ids are computed.")
    ap.add_argument(
        "--tta-visible-frames", type=str, default="auto",
        help="'auto' resolves to "
             f"{AUTO_TTA_VISIBLE_RANGE[0]}:{AUTO_TTA_VISIBLE_RANGE[1]} "
             "(matches the panda_1000v_standard TTA-runner visible window).",
    )
    ap.add_argument(
        "--tta-context-frames", type=int, default=TTA_CONTEXT_FRAMES,
        help=f"Default {TTA_CONTEXT_FRAMES} matches NUM_COND_FRAMES.",
    )
    ap.add_argument(
        "--timesteps", type=str, default=DEFAULT_TIMESTEPS,
        help=f"Comma-separated integer timesteps in [0, {NUM_TRAIN_TIMESTEPS}). "
             "Default '100,500,900' = the same low/mid/high schedule used by "
             "compute_diffusion_ood_score.py.",
    )
    ap.add_argument("--lora-rank", type=int, default=DEFAULT_LORA_RANK,
                    help="LoRA rank (default 8 = headline LORA_R8_TTA).")
    ap.add_argument("--lora-alpha", type=float, default=DEFAULT_LORA_ALPHA,
                    help="LoRA alpha (default 16 = headline LORA_R8_TTA).")
    ap.add_argument("--lora-lr", type=float, default=DEFAULT_LORA_LR,
                    help=f"AdamW learning rate for the single Adam step "
                         f"(default {DEFAULT_LORA_LR} = headline LORA_R8_TTA "
                         f"LEARNING_RATE from "
                         f"submit_standard_1000v_chunked.sh).")
    ap.add_argument("--lora-weight-decay", type=float,
                    default=DEFAULT_LORA_WEIGHT_DECAY,
                    help=f"AdamW weight decay (default "
                         f"{DEFAULT_LORA_WEIGHT_DECAY} = headline recipe).")
    ap.add_argument("--lora-targets", type=str, default=DEFAULT_LORA_TARGETS,
                    help="Comma-separated subset of {qkv,proj} (default "
                         f"'{DEFAULT_LORA_TARGETS}' = headline recipe).")
    ap.add_argument("--lora-target-blocks", type=str,
                    default=DEFAULT_LORA_TARGET_BLOCKS,
                    help="'all' (default), 'last_N', or '0,5,10' explicit "
                         "block indices (mirrors run_lora_tta.py).")
    ap.add_argument("--lora-target-ffn", action="store_true",
                    help="Also LoRA-adapt the FFN w1/w2/w3 (off by default; "
                         "matches headline TARGET_FFN=0).")
    ap.add_argument("--seed", type=int, default=0,
                    help="Base seed for the per-video noise sampler; the "
                         "actual noise seed per (video, t) is "
                         "base + blake2b(video_id) + t, so videos are "
                         "reproducible independently of each other.")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--max-videos", type=int, default=0,
                    help="0 = all. Otherwise process the first N (by "
                         "canonical id) for smoke-testing.")
    ap.add_argument("--resume", action="store_true",
                    help="Skip videos already present in --output.")
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    visible_range = _parse_frame_range_arg(
        args.tta_visible_frames, default=AUTO_TTA_VISIBLE_RANGE,
    )
    timesteps = _parse_timesteps_arg(args.timesteps)
    lora_targets_list = _parse_targets_arg(args.lora_targets)
    n_visible = visible_range[1] - visible_range[0]

    args.output.parent.mkdir(parents=True, exist_ok=True)

    num_ctx_lat_preview = 1 + (max(1, args.tta_context_frames) - 1) // VAE_TEMPORAL_SCALE
    n_visible_lat_preview = 1 + (max(1, n_visible) - 1) // VAE_TEMPORAL_SCALE

    existing_rows, existing_fields = _load_existing_ids(args.output)
    fieldnames = _fieldnames(timesteps)
    if existing_rows and args.resume:
        existing_ts_cols = sorted(
            c for c in existing_fields
            if c.startswith("grad_norm_lora_t")
        )
        new_ts_cols = sorted(
            f"grad_norm_lora_t{t}" for t in timesteps
        )
        if existing_ts_cols and existing_ts_cols != new_ts_cols:
            print(
                f"[error] --resume schema mismatch: existing CSV has "
                f"{existing_ts_cols} but --timesteps={timesteps} would "
                f"produce {new_ts_cols}. Re-run without --resume to "
                f"overwrite, or pass --timesteps matching the existing CSV.",
                file=sys.stderr,
            )
            return 2
        for extra in existing_fields:
            if extra and extra not in fieldnames:
                fieldnames.append(extra)

    existing_ids = {
        (r.get("video_id") or "").strip()
        for r in existing_rows if (r.get("video_id") or "").strip()
    } if args.resume else set()

    print("=" * 78)
    print("LongCat-Video Tier-3 probes (H-T3-1 grad_norm + H-T3-2 single-step loss drop)")
    print("=" * 78)
    print(f"Checkpoint dir       : {args.checkpoint_dir}")
    print(f"Videos dir           : {args.videos_dir}")
    print(f"Captions CSV         : {args.captions_csv}")
    print(f"Output               : {args.output}")
    print(f"TTA-visible frames   : {visible_range[0]}:{visible_range[1]}  "
          f"(n={n_visible} pixel frames -> ~{n_visible_lat_preview} latent frames)")
    print(f"TTA-context frames   : {args.tta_context_frames}  "
          f"(-> ~{num_ctx_lat_preview} clean cond latents)")
    print(f"Gen-target frames    : {AUTO_GEN_TARGET_FRAMES}  (documented in CSV)")
    print(f"Timesteps            : {timesteps}  "
          f"(sigma = t / {NUM_TRAIN_TIMESTEPS}: "
          f"{[round(t / NUM_TRAIN_TIMESTEPS, 3) for t in timesteps]})")
    print(f"LoRA recipe          : rank={args.lora_rank}, alpha={args.lora_alpha}, "
          f"targets={lora_targets_list}, target_blocks={args.lora_target_blocks}, "
          f"target_ffn={args.lora_target_ffn}")
    print(f"Optimiser            : AdamW(lr={args.lora_lr}, "
          f"weight_decay={args.lora_weight_decay}, "
          f"betas=(0.9, 0.999), eps=1e-8) — ONE step per (video, t)")
    print(f"Model dtype          : torch.bfloat16")
    print(f"Device               : {args.device}")
    print(f"Base seed            : {args.seed}")
    print(f"Resume               : {args.resume}  "
          f"({len(existing_ids)} existing rows in --output)")
    print("=" * 78)

    captions_by_id = _load_captions_csv(args.captions_csv)
    video_paths = _list_video_paths(args.videos_dir)
    if args.max_videos and args.max_videos > 0:
        video_paths = video_paths[: args.max_videos]
    if not video_paths:
        print(f"[error] no video files found under {args.videos_dir}",
              file=sys.stderr)
        return 2

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
    vae.eval()
    text_encoder.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    for p in text_encoder.parameters():
        p.requires_grad_(False)
    # Base DiT params: frozen.  LoRA params (added below) are the ONLY
    # tensors with requires_grad=True from this point.
    for p in dit.parameters():
        p.requires_grad_(False)

    try:
        dit_patch_t = int(dit.config.patch_size[0])
    except (AttributeError, IndexError, TypeError) as exc:
        print(f"[error] could not resolve dit.config.patch_size[0]: {exc}",
              file=sys.stderr)
        return 2
    print(f"  DiT patch_t        : {dit_patch_t}")
    print(f"  DiT blocks         : {len(dit.blocks)}")

    # ---- Inject LoRA --------------------------------------------------------
    print("\nInjecting LoRA adapters (rank=%d, alpha=%.1f, targets=%s, "
          "blocks=%s, ffn=%s)..."
          % (args.lora_rank, args.lora_alpha, lora_targets_list,
             args.lora_target_blocks, args.lora_target_ffn))
    try:
        from run_lora_tta import (
            inject_lora_into_dit, get_lora_parameters,
            count_lora_parameters, reset_lora_weights,
        )
    except ImportError as exc:
        print(f"[error] failed to import run_lora_tta helpers: {exc}",
              file=sys.stderr)
        traceback.print_exc()
        return 2

    lora_modules = inject_lora_into_dit(
        dit,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=0.0,
        target_modules=lora_targets_list,
        target_ffn=args.lora_target_ffn,
        target_blocks=args.lora_target_blocks,
    )
    lora_params = get_lora_parameters(lora_modules)
    counts = count_lora_parameters(lora_modules)
    print(f"  LoRA modules       : {len(lora_modules)}")
    print(f"  LoRA params        : total={counts['total_lora']:,}, "
          f"trainable={counts['trainable']:,}")
    if not lora_params:
        print("[error] LoRA injection produced no trainable params",
              file=sys.stderr)
        return 2

    appending = args.resume and bool(existing_rows) and args.output.exists()
    out_handle = args.output.open(
        "a" if appending else "w", newline="", encoding="utf-8",
    )
    writer = csv.DictWriter(out_handle, fieldnames=fieldnames)
    if not appending:
        writer.writeheader()
        out_handle.flush()

    n_done = 0
    n_errored = 0
    t0 = time.time()
    last_print_t = t0
    first_row: Optional[Dict[str, float]] = None
    lora_targets_str = ",".join(lora_targets_list)

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
                lora_modules=lora_modules,
                lora_params=lora_params,
                reset_lora_fn=reset_lora_weights,
                lora_rank=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_lr=args.lora_lr,
                lora_weight_decay=args.lora_weight_decay,
                lora_targets=lora_targets_str,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[error] {vp.name}: {exc}", file=sys.stderr)
            traceback.print_exc()
            n_errored += 1
            # Make sure a partial-failure does NOT leave the adapter in
            # a non-reset state for the next video.
            try:
                reset_lora_weights(lora_modules)
            except Exception:
                pass
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
