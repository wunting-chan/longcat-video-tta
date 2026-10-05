#!/usr/bin/env python3
"""
LoRA Test-Time Adaptation (TTA) for LongCat-Video.

Fine-tunes lightweight LoRA adapters on conditioning frames for each video,
then generates continuations using LongCat-Video's video continuation pipeline.

Key features:
- Injects LoRA adapters into the DiT's Linear layers (qkv, proj, ffn)
- Fine-tunes ONLY on conditioning frames (no ground truth in training)
- Uses LongCat-Video's native video continuation pipeline for generation
- Resets LoRA weights between videos
- Checkpoints progress for resumability
- Optional early stopping via anchor loss on held-out frames

Usage:
    python run_lora_tta.py \\
        --checkpoint-dir /path/to/longcat-video-checkpoints \\
        --data-dir /path/to/dataset \\
        --output-dir results/lora_tta_r8_lr2e-4 \\
        --lora-rank 8 --learning-rate 2e-4 --num-steps 20
"""

import argparse
import copy
import gc
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

# Ensure common.py and early_stopping.py are importable from delta_experiment
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[1]
_DELTA_SCRIPTS = _REPO_ROOT / "delta_experiment" / "scripts"
sys.path.insert(0, str(_DELTA_SCRIPTS))
sys.path.insert(0, str(_REPO_ROOT))

from common import (
    load_longcat_components,
    load_video_frames,
    encode_video,
    encode_prompt,
    compute_flow_matching_loss,
    compute_flow_matching_loss_conditioned,
    generate_video_continuation,
    save_results,
    load_checkpoint,
    save_checkpoint,
    torch_gc,
    load_ucf101_video_list,
    decode_latents,
    compute_psnr,
    compute_ssim_batch,
    compute_lpips_batch,
    build_augmented_latent_variants,
    add_augmentation_args,
    add_tta_frame_args,
    add_caption_guard_args,
    add_caption_override_args,
    add_tta_disable_caption_args,
    tta_caption_for,
    add_feature_frame_guard_args,
    add_clip_gate_args,
    parse_speed_factors,
    split_tta_latents,
    evaluate_generation_metrics,
    build_retrieval_pool,
    retrieve_neighbors,
    evaluate_clip_gate,
    summarize_clip_gate_stats,
    validate_caption_quality,
    apply_fixed_caption,
    validate_tta_feature_budget,
    add_online_eval_args,
    OnlineFrechetAccumulator,
    finalize_online_eval,
    aggregate_quality_metrics,
)
from early_stopping import (
    AnchoredEarlyStopper,
    add_early_stopping_args,
    build_early_stopper_from_args,
)


# ============================================================================
# GPU memory tracking
# ============================================================================

def gpu_mem_stats(device="cuda"):
    """Return a dict of current GPU memory usage in GiB."""
    if not torch.cuda.is_available():
        return {"allocated_gib": 0, "reserved_gib": 0, "peak_gib": 0, "total_gib": 0, "free_gib": 0}
    alloc = torch.cuda.memory_allocated(device) / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
    peak = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    total = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
    return {
        "allocated_gib": round(alloc, 2),
        "reserved_gib": round(reserved, 2),
        "peak_gib": round(peak, 2),
        "total_gib": round(total, 2),
        "free_gib": round(total - alloc, 2),
    }


def log_gpu_mem(label, device="cuda"):
    stats = gpu_mem_stats(device)
    print(f"  [GPU] {label}: "
          f"alloc={stats['allocated_gib']:.2f} GiB, "
          f"peak={stats['peak_gib']:.2f} GiB, "
          f"free={stats['free_gib']:.2f} GiB / "
          f"{stats['total_gib']:.1f} GiB total")
    return stats


# ============================================================================
# Built-in LoRA support (uses LongCat-Video's native LoRAModule)
# ============================================================================

_LONGCAT_ROOT = _REPO_ROOT / "LongCat-Video"
sys.path.insert(0, str(_LONGCAT_ROOT))
from longcat_video.modules.lora_utils import LoRAModule


def inject_builtin_lora_into_dit(
    dit: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    target_modules: List[str] = ("qkv", "proj"),
    target_ffn: bool = False,
    target_blocks: str = "all",
) -> List[LoRAModule]:
    """Inject LoRA via LongCat's native LoRAModule + forward-hook mechanism.

    Instead of replacing nn.Linear modules (like our custom LoRALinear),
    this creates standalone LoRAModule instances and patches the original
    module's forward to add the LoRA contribution -- exactly matching the
    official LongCat-Video LoRA workflow.
    """
    device = next(dit.parameters()).device
    dtype = next(dit.parameters()).dtype

    block_indices = _parse_target_blocks(target_blocks, len(dit.blocks))
    label = (f"{sorted(block_indices)} ({len(block_indices)}/{len(dit.blocks)})"
             if block_indices is not None else f"all ({len(dit.blocks)})")
    print(f"  [builtin] LoRA target blocks: {label}")

    lora_modules: List[LoRAModule] = []

    def _maybe_add(module: nn.Module, name: str, n_sep: int = 1):
        if not isinstance(module, nn.Linear):
            return
        lora = LoRAModule(name, module, multiplier=1.0,
                          lora_dim=rank, alpha=alpha, n_seperate=n_sep)
        lora = lora.to(device=device, dtype=dtype)
        lora_modules.append(lora)

        if not hasattr(module, "org_forward"):
            module.org_forward = module.forward
        hooked_fwd = _build_hooked_forward(module, lora)
        module.forward = hooked_fwd

    for block_idx, block in enumerate(dit.blocks):
        if block_indices is not None and block_idx not in block_indices:
            continue

        if hasattr(block, "attn"):
            attn = block.attn
            if "qkv" in target_modules and hasattr(attn, "qkv"):
                _maybe_add(attn.qkv, f"blocks.{block_idx}.attn.qkv", n_sep=3)
            if "proj" in target_modules and hasattr(attn, "proj"):
                _maybe_add(attn.proj, f"blocks.{block_idx}.attn.proj")

        if hasattr(block, "cross_attn"):
            xattn = block.cross_attn
            if "qkv" in target_modules:
                if hasattr(xattn, "q_linear"):
                    _maybe_add(xattn.q_linear, f"blocks.{block_idx}.cross_attn.q_linear")
                if hasattr(xattn, "kv_linear"):
                    _maybe_add(xattn.kv_linear, f"blocks.{block_idx}.cross_attn.kv_linear", n_sep=2)
            if "proj" in target_modules and hasattr(xattn, "proj"):
                _maybe_add(xattn.proj, f"blocks.{block_idx}.cross_attn.proj")

        if target_ffn and hasattr(block, "ffn"):
            ffn = block.ffn
            for layer_name in ("w1", "w2", "w3"):
                if hasattr(ffn, layer_name):
                    _maybe_add(getattr(ffn, layer_name),
                               f"blocks.{block_idx}.ffn.{layer_name}")

    return lora_modules


def _build_hooked_forward(module: nn.Module, lora: LoRAModule):
    """Build a patched forward that adds the LoRA contribution."""
    def hooked_forward(x, *args, **kwargs):
        org_output = module.org_forward(x, *args, **kwargs)
        if lora.use_lora:
            lx = lora.lora_down(x.to(lora.lora_down.weight.dtype))
            lx = lora.lora_up(lx)
            org_output = org_output + lx.to(org_output.dtype) * lora.multiplier * lora.alpha_scale
        return org_output
    return hooked_forward


def get_builtin_lora_parameters(lora_modules: List[LoRAModule]) -> List[nn.Parameter]:
    """Collect trainable parameters from builtin LoRA modules."""
    params = []
    for lora in lora_modules:
        params.extend(lora.parameters())
    return [p for p in params if not isinstance(p, torch.Tensor) or p.requires_grad]


def count_builtin_lora_parameters(lora_modules: List[LoRAModule]) -> Dict[str, int]:
    """Count trainable parameters in builtin LoRA modules."""
    trainable = sum(p.numel() for lora in lora_modules
                    for p in lora.parameters() if p.requires_grad)
    total = sum(p.numel() for lora in lora_modules for p in lora.parameters())
    return {"total_lora": total, "trainable": trainable}


def reset_builtin_lora_weights(lora_modules: List[LoRAModule]):
    """Re-initialize builtin LoRA weights to zero-output state."""
    for lora in lora_modules:
        nn.init.kaiming_uniform_(lora.lora_down.weight, a=math.sqrt(5))
        if hasattr(lora.lora_up, "blocks"):
            for blk in lora.lora_up.blocks:
                nn.init.zeros_(blk.weight)
        else:
            nn.init.zeros_(lora.lora_up.weight)


def unhook_builtin_lora(dit: nn.Module):
    """Restore all forward methods patched by inject_builtin_lora_into_dit."""
    for _, module in dit.named_modules():
        if hasattr(module, "org_forward"):
            module.forward = module.org_forward
            delattr(module, "org_forward")


# ============================================================================
# Custom LoRA implementation (our original approach)
# ============================================================================

class LoRALinear(nn.Module):
    """Low-rank adapter that wraps an existing nn.Linear.

    The original weight stays frozen; only lora_down and lora_up are trained.
    Output = original_forward(x) + (x @ lora_down^T @ lora_up^T) * (alpha / rank)
    """

    def __init__(
        self,
        original: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.original = original
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = original.in_features
        out_features = original.out_features

        self.lora_down = nn.Linear(in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, out_features, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Kaiming init for down, zero init for up (standard LoRA init)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x, *args, **kwargs):
        # Original computation (frozen)
        orig_out = self.original(x, *args, **kwargs)
        # LoRA delta
        lora_out = self.lora_up(self.lora_down(self.dropout(x.to(self.lora_down.weight.dtype))))
        return orig_out + lora_out * self.scaling


def _parse_target_blocks(target_blocks: str, num_blocks: int) -> Optional[set]:
    """Parse --lora-target-blocks into a set of block indices.

    Accepts:
      "all"    -> None (every block)
      "last_N" -> last N block indices
      "0,5,10" -> explicit comma-separated indices
    """
    target_blocks = target_blocks.strip().lower()
    if target_blocks == "all":
        return None
    if target_blocks.startswith("last_"):
        n = int(target_blocks.split("_", 1)[1])
        if n <= 0 or n > num_blocks:
            raise ValueError(f"last_{n} invalid for {num_blocks} blocks")
        return set(range(num_blocks - n, num_blocks))
    indices = set(int(x.strip()) for x in target_blocks.split(","))
    for idx in indices:
        if idx < 0 or idx >= num_blocks:
            raise ValueError(f"Block index {idx} out of range [0, {num_blocks})")
    return indices


def inject_lora_into_dit(
    dit: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    target_modules: List[str] = ("qkv", "proj"),
    target_ffn: bool = False,
    target_blocks: str = "all",
) -> List[LoRALinear]:
    """Inject LoRA adapters into the DiT's transformer blocks.

    Parameters
    ----------
    dit : LongCatVideoTransformer3DModel
    rank : LoRA rank
    alpha : LoRA alpha (scaling = alpha / rank)
    dropout : Dropout rate for LoRA
    target_modules : Which attention modules to target. Options: "qkv", "proj"
    target_ffn : If True, also target FFN (w1, w2, w3) layers
    target_blocks : Which blocks to inject LoRA into.
        "all" = every block, "last_N" = last N blocks, or comma-separated indices.

    Returns
    -------
    List of all LoRALinear modules created.
    """
    lora_modules = []
    device = next(dit.parameters()).device
    dtype = next(dit.parameters()).dtype

    block_indices = _parse_target_blocks(target_blocks, len(dit.blocks))
    if block_indices is not None:
        print(f"  LoRA target blocks: {sorted(block_indices)} "
              f"({len(block_indices)}/{len(dit.blocks)})")
    else:
        print(f"  LoRA target blocks: all ({len(dit.blocks)})")

    for block_idx, block in enumerate(dit.blocks):
        if block_indices is not None and block_idx not in block_indices:
            continue
        # Self-attention layers
        if hasattr(block, "attn"):
            attn = block.attn
            if "qkv" in target_modules and hasattr(attn, "qkv"):
                orig = attn.qkv
                lora = LoRALinear(orig, rank=rank, alpha=alpha, dropout=dropout)
                lora = lora.to(device=device, dtype=dtype)
                attn.qkv = lora
                lora_modules.append(lora)

            if "proj" in target_modules and hasattr(attn, "proj"):
                orig = attn.proj
                if isinstance(orig, nn.Linear):
                    lora = LoRALinear(orig, rank=rank, alpha=alpha, dropout=dropout)
                    lora = lora.to(device=device, dtype=dtype)
                    attn.proj = lora
                    lora_modules.append(lora)

        # Cross-attention layers
        if hasattr(block, "cross_attn"):
            xattn = block.cross_attn
            if "qkv" in target_modules:
                if hasattr(xattn, "q_linear") and isinstance(xattn.q_linear, nn.Linear):
                    orig = xattn.q_linear
                    lora = LoRALinear(orig, rank=rank, alpha=alpha, dropout=dropout)
                    lora = lora.to(device=device, dtype=dtype)
                    xattn.q_linear = lora
                    lora_modules.append(lora)

                if hasattr(xattn, "kv_linear") and isinstance(xattn.kv_linear, nn.Linear):
                    orig = xattn.kv_linear
                    lora = LoRALinear(orig, rank=rank, alpha=alpha, dropout=dropout)
                    lora = lora.to(device=device, dtype=dtype)
                    xattn.kv_linear = lora
                    lora_modules.append(lora)

            if "proj" in target_modules and hasattr(xattn, "proj"):
                orig = xattn.proj
                if isinstance(orig, nn.Linear):
                    lora = LoRALinear(orig, rank=rank, alpha=alpha, dropout=dropout)
                    lora = lora.to(device=device, dtype=dtype)
                    xattn.proj = lora
                    lora_modules.append(lora)

        # FFN layers
        if target_ffn and hasattr(block, "ffn"):
            ffn = block.ffn
            for layer_name in ("w1", "w2", "w3"):
                if hasattr(ffn, layer_name):
                    orig = getattr(ffn, layer_name)
                    if isinstance(orig, nn.Linear):
                        lora = LoRALinear(orig, rank=rank, alpha=alpha, dropout=dropout)
                        lora = lora.to(device=device, dtype=dtype)
                        setattr(ffn, layer_name, lora)
                        lora_modules.append(lora)

    return lora_modules


def get_lora_parameters(lora_modules: List[LoRALinear]) -> List[nn.Parameter]:
    """Collect all trainable LoRA parameters."""
    params = []
    for lora in lora_modules:
        params.extend(lora.lora_down.parameters())
        params.extend(lora.lora_up.parameters())
    return params


def count_lora_parameters(lora_modules: List[LoRALinear]) -> Dict[str, int]:
    """Count total / trainable LoRA parameters."""
    total = sum(p.numel() for lora in lora_modules for p in lora.parameters())
    trainable = sum(
        p.numel()
        for lora in lora_modules
        for p in [*lora.lora_down.parameters(), *lora.lora_up.parameters()]
    )
    return {"total_lora": total, "trainable": trainable}


def reset_lora_weights(lora_modules: List[LoRALinear]):
    """Re-initialize all LoRA weights to their initial state (zero output)."""
    for lora in lora_modules:
        nn.init.kaiming_uniform_(lora.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(lora.lora_up.weight)


def save_lora_weights(lora_modules: List[LoRALinear], path: str):
    """Save LoRA weights to a file."""
    state = {}
    for i, lora in enumerate(lora_modules):
        state[f"lora_{i}.down"] = lora.lora_down.weight.detach().cpu()
        state[f"lora_{i}.up"] = lora.lora_up.weight.detach().cpu()
    torch.save(state, path)


# ============================================================================
# LoRA TTA training loop
# ============================================================================

def finetune_lora_on_conditioning(
    dit: nn.Module,
    lora_modules,
    cond_latents: torch.Tensor,
    train_latents: torch.Tensor,
    prompt_embeds: torch.Tensor,
    prompt_mask: torch.Tensor,
    num_steps: int = 20,
    lr: float = 2e-4,
    warmup_steps: int = 3,
    weight_decay: float = 0.01,
    max_grad_norm: float = 10.0,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    early_stopper: Optional[AnchoredEarlyStopper] = None,
    lora_param_fn=None,
    train_latents_variants: Optional[List[Dict]] = None,
    anchor_x0_weight: float = 0.0,
) -> Dict:
    """Fine-tune LoRA adapters using conditioning-aware loss.

    Parameters
    ----------
    cond_latents  : clean context latents [B, C, T_cond, H, W]
    train_latents : target latents to noise and compute loss on [B, C, T_train, H, W]
    train_latents_variants : optional augmented variants of train_latents

    Returns
    -------
    dict with keys: losses, train_time, early_stopping_info, grad_norms, lora_delta_l2
    """
    if lora_param_fn is not None:
        lora_params = lora_param_fn()
    else:
        lora_params = get_lora_parameters(lora_modules)
    if not lora_params:
        raise ValueError("No LoRA parameters found.")

    optimizer = AdamW(
        lora_params,
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=weight_decay,
        eps=1e-8,
    )

    if train_latents_variants is None:
        train_latents_variants = [{"latents": train_latents, "name": "orig"}]

    def _save_fn():
        return [p.data.clone() for p in lora_params]

    def _restore_from_snapshot(snapshot):
        # Backward/interop safety: older callers may provide a state_dict-like
        # snapshot while current flow stores an ordered list of tensors.
        if isinstance(snapshot, dict):
            _restore_lora_from_state(dit, snapshot)
            return
        for p, saved in zip(lora_params, snapshot):
            p.data.copy_(saved)

    dit.train()
    losses = []
    grad_norms = []
    train_start = time.time()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    es_check_time = 0.0
    for step in range(num_steps):
        optimizer.zero_grad(set_to_none=True)

        # LR warmup
        if step < warmup_steps and warmup_steps > 0:
            warmup_lr = lr * (step + 1) / warmup_steps
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr

        vi = torch.randint(0, len(train_latents_variants), (1,)).item()
        step_train = train_latents_variants[vi]["latents"]

        loss = compute_flow_matching_loss_conditioned(
            dit=dit,
            cond_latents=cond_latents,
            target_latents=step_train,
            prompt_embeds=prompt_embeds,
            prompt_mask=prompt_mask,
            device=device,
            dtype=dtype,
            anchor_x0_weight=anchor_x0_weight,
        )

        loss.backward()

        raw_grad_norm = torch.nn.utils.clip_grad_norm_(lora_params, float("inf"))
        grad_norms.append(raw_grad_norm.item())
        if raw_grad_norm > max_grad_norm:
            scale = max_grad_norm / (raw_grad_norm + 1e-6)
            for p in lora_params:
                if p.grad is not None:
                    p.grad.mul_(scale)

        optimizer.step()

        losses.append(loss.item())
        del loss

        if step % 10 == 0:
            torch.cuda.empty_cache()

        if early_stopper is not None:
            es_t0 = time.time()
            should_stop, es_info = early_stopper.step(
                step + 1, save_fn=_save_fn,
            )
            es_check_time += time.time() - es_t0
            if should_stop:
                print(f"  Early stopping at step {step + 1}: {es_info}")
                break

    train_time = time.time() - train_start
    dit.eval()

    es_state = None
    if early_stopper is not None:
        early_stopper.restore(restore_fn=_restore_from_snapshot)
        es_state = early_stopper.state

    lora_delta_l2 = 0.0
    with torch.no_grad():
        for p in lora_params:
            lora_delta_l2 += p.data.float().pow(2).sum().item()
    lora_delta_l2 = lora_delta_l2 ** 0.5

    train_mem = gpu_mem_stats(device) if torch.cuda.is_available() else {}

    if grad_norms:
        print(f"  Grad norm stats: min={min(grad_norms):.4f}, "
              f"max={max(grad_norms):.4f}, "
              f"mean={sum(grad_norms)/len(grad_norms):.4f}")
    print(f"  LoRA param L2 norm = {lora_delta_l2:.6f}")
    if train_mem:
        print(f"  Train peak GPU: {train_mem.get('peak_gib', 0):.2f} GiB "
              f"/ {train_mem.get('total_gib', 0):.1f} GiB")

    torch.cuda.empty_cache()

    return {
        "losses": losses,
        "train_time": train_time,
        "es_check_time": es_check_time,
        "early_stopping_info": es_state,
        "grad_norms": grad_norms,
        "lora_delta_l2": lora_delta_l2,
        "gpu_mem_train": train_mem,
    }


def _restore_lora_from_state(model: nn.Module, state_dict: dict):
    """Restore LoRA parameters from a snapshot state dict."""
    current = model.state_dict()
    for k, v in state_dict.items():
        if k in current:
            current[k].copy_(v)


def finetune_lora_batch(
    dit: nn.Module,
    lora_modules,
    batch_data: List[Dict],
    num_steps: int = 20,
    lr: float = 2e-4,
    warmup_steps: int = 3,
    weight_decay: float = 0.01,
    max_grad_norm: float = 1.0,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    lora_param_fn=None,
    anchor_x0_weight: float = 0.0,
) -> Dict:
    """Fine-tune shared LoRA adapters across multiple videos (round-robin).

    Used for retrieval-augmented batch-level TTA: the LoRA adapters are
    trained on the eval video + K-1 retrieved neighbours, cycling through
    them at each step.
    """
    if lora_param_fn is not None:
        lora_params = lora_param_fn()
    else:
        lora_params = get_lora_parameters(lora_modules)

    optimizer = AdamW(
        lora_params, lr=lr, betas=(0.9, 0.999),
        weight_decay=weight_decay, eps=1e-8,
    )

    dit.train()
    losses = []
    n_vids = len(batch_data)
    train_start = time.time()

    for step in range(num_steps):
        optimizer.zero_grad(set_to_none=True)

        if step < warmup_steps and warmup_steps > 0:
            warmup_lr = lr * (step + 1) / warmup_steps
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr

        vi = step % n_vids
        bd = batch_data[vi]

        cond_lat = bd["cond_latents"].to(device)
        train_lat = bd["train_latents"].to(device)
        pe = bd["prompt_embeds"].to(device)
        pm = bd["prompt_mask"].to(device) if bd["prompt_mask"] is not None else None

        loss = compute_flow_matching_loss_conditioned(
            dit=dit,
            cond_latents=cond_lat,
            target_latents=train_lat,
            prompt_embeds=pe,
            prompt_mask=pm,
            device=device,
            dtype=dtype,
            anchor_x0_weight=anchor_x0_weight,
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(lora_params, max_grad_norm)
        optimizer.step()

        losses.append(loss.item())
        del cond_lat, train_lat, pe, pm, loss

    train_time = time.time() - train_start
    dit.eval()
    torch.cuda.empty_cache()

    return {
        "losses": losses,
        "train_time": train_time,
        "es_check_time": 0.0,
        "early_stopping_info": None,
    }


# ============================================================================
# Evaluation helpers
# ============================================================================

def save_video_from_numpy(frames: np.ndarray, output_path: str, fps: int = 24):
    """Save video from numpy array [N, H, W, 3] in [0, 1]."""
    import imageio

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    frames_u8 = (np.clip(frames, 0, 1) * 255).astype(np.uint8)
    imageio.mimwrite(output_path, frames_u8, fps=fps, codec="libx264", quality=9)


# ============================================================================
# Main experiment
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="LoRA TTA for LongCat-Video")

    # Data / output
    parser.add_argument("--checkpoint-dir", type=str, required=True,
                        help="Path to LongCat-Video checkpoint directory")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Path to video dataset directory")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for results")
    parser.add_argument("--max-videos", type=int, default=100)
    parser.add_argument("--start-video-idx", type=int, default=0,
                        help="Start processing from this index in the video list (for chunked runs)")
    parser.add_argument("--chunk-size", type=int, default=0,
                        help="Number of videos to process from start-video-idx (0 = all remaining)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--restart", action="store_true",
                        help="Restart from beginning (ignore checkpoint)")

    # LoRA arguments
    parser.add_argument("--lora-rank", type=int, default=8,
                        help="LoRA rank (default: 8)")
    parser.add_argument("--lora-alpha", type=float, default=16.0,
                        help="LoRA alpha for scaling (default: 16)")
    parser.add_argument("--lora-dropout", type=float, default=0.0,
                        help="LoRA dropout rate")
    parser.add_argument("--target-ffn", action="store_true",
                        help="Also apply LoRA to FFN (w1, w2, w3) layers")
    parser.add_argument("--target-modules", type=str, default="qkv,proj",
                        help="Comma-separated attention modules to target (qkv, proj)")
    parser.add_argument("--lora-target-blocks", type=str, default="all",
                        help="Which DiT blocks to inject LoRA into. "
                             "'all' = every block, 'last_N' = last N blocks "
                             "(e.g. 'last_4', 'last_8'), or comma-separated "
                             "block indices (e.g. '44,45,46,47')")
    parser.add_argument("--use-builtin-lora", action="store_true",
                        help="Use LongCat-Video's native LoRAModule instead of "
                             "our custom LoRALinear wrapper. Enables n_separate "
                             "for fused QKV (3-way) and KV (2-way) projections.")
    parser.add_argument("--save-lora-weights", action="store_true",
                        help="Save per-video LoRA weights")

    # Training arguments
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)

    # Video continuation arguments
    parser.add_argument("--num-cond-frames", type=int, default=2,
                        help="Number of conditioning frames from the input video")
    parser.add_argument("--num-frames", type=int, default=16,
                        help="Total number of frames to generate (including cond)")
    parser.add_argument("--gen-start-frame", type=int, default=32,
                        help="Fixed anchor frame where generation starts. "
                             "Cond = video[anchor-cond : anchor]. "
                             "Ensures fair comparison across configs.")
    parser.add_argument("--num-inference-steps", type=int, default=50,
                        help="Number of diffusion denoising steps for generation")
    parser.add_argument("--guidance-scale", type=float, default=4.0,
                        help="Guidance scale for generation")
    parser.add_argument("--resolution", type=str, default="480p",
                        help="Target resolution for generation")
    parser.add_argument("--skip-generation", action="store_true",
                        help="Skip video generation (only train)")
    parser.add_argument("--no-save-videos", action="store_true",
                        help="Skip saving generated videos to disk (metrics still computed in-memory)")
    parser.add_argument("--save-only-list", type=str, default=None,
                        help="Path to retain_videos.json; save MP4s only for listed videos")
    parser.add_argument("--fps", type=int, default=24,
                        help="Playback frame rate for saved MP4s (default: 24). "
                             "Only affects file encoding, not generation; safe to "
                             "leave at 24 for existing experiments.")
    parser.add_argument("--rollout-steps", type=int, default=1,
                        help="Number of autoregressive generation steps (default 1 = single-step)")

    # Retrieval-augmented batch TTA
    parser.add_argument("--batch-videos", type=int, default=1,
                        help="Number of videos per TTA batch. 1=instance-level (default), "
                             "K>1=retrieval-augmented batch-level.")
    parser.add_argument("--batch-method", type=str, default="similarity",
                        choices=["sequential", "similarity"],
                        help="(Legacy) Retrieval is always by text-prompt similarity.")
    parser.add_argument("--retrieval-pool-dir", type=str, default=None,
                        help="Directory containing the larger retrieval pool dataset. "
                             "Required when --batch-videos > 1.")

    parser.add_argument("--anchor-x0-weight", type=float, default=0.0,
                        help="Weight for the anchor-frame x0 consistency auxiliary loss "
                             "(rectified-flow recovery; Modification 1 of "
                             "sweep_experiment/reports/LITERATURE_tta_recipe_modifications_2026-06-12.md). "
                             "Default 0.0 = byte-identical to pre-patch behaviour.")

    # Early stopping
    add_early_stopping_args(parser)
    add_augmentation_args(parser)
    add_tta_frame_args(parser)
    add_caption_guard_args(parser)
    add_caption_override_args(parser)
    add_tta_disable_caption_args(parser)
    add_feature_frame_guard_args(parser)
    add_online_eval_args(parser)
    add_clip_gate_args(parser)

    args = parser.parse_args()

    # Default tta_total_frames to gen_start_frame (use all pre-anchor frames)
    if args.tta_total_frames is None:
        args.tta_total_frames = args.gen_start_frame
    # Default tta_context_frames to match generation conditioning
    if args.tta_context_frames is None or args.tta_context_frames > args.tta_total_frames:
        args.tta_context_frames = args.num_cond_frames
    # Safety: never let TTA include anchor/future GT frames.
    if args.tta_total_frames > args.gen_start_frame:
        print(
            f"[WARN] tta_total_frames ({args.tta_total_frames}) exceeds "
            f"gen_start_frame ({args.gen_start_frame}); clamping to avoid GT leakage."
        )
        args.tta_total_frames = args.gen_start_frame
    if args.tta_context_frames > args.tta_total_frames:
        args.tta_context_frames = args.tta_total_frames
    validate_tta_feature_budget(args, context="lora_tta")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # Resume support
    ckpt_path = os.path.join(args.output_dir, "checkpoint.json")
    ckpt = load_checkpoint(ckpt_path)
    start_idx = 0
    all_results = []
    if ckpt and not args.restart:
        start_idx = ckpt.get("next_idx", 0)
        all_results = ckpt.get("results", [])

    target_modules = [m.strip() for m in args.target_modules.split(",")]

    method_label = "LoRA TTA"
    if args.anchor_x0_weight > 0.0:
        method_label = f"LoRA TTA + x0 (λ={args.anchor_x0_weight:g})"

    print("=" * 70)
    print(f"{method_label} (Test-Time Adaptation) for LongCat-Video")
    print("=" * 70)
    print(f"Checkpoint dir : {args.checkpoint_dir}")
    print(f"Data dir       : {args.data_dir}")
    print(f"Output dir     : {args.output_dir}")
    use_builtin = getattr(args, "use_builtin_lora", False)
    print(f"LoRA impl      : {'builtin (LongCat native)' if use_builtin else 'custom (LoRALinear)'}")
    print(f"LoRA rank      : {args.lora_rank}")
    print(f"LoRA alpha     : {args.lora_alpha}")
    print(f"Target modules : {target_modules}")
    print(f"Target blocks  : {args.lora_target_blocks}")
    print(f"Target FFN     : {args.target_ffn}")
    print(f"Learning rate  : {args.learning_rate}")
    print(f"Num steps      : {args.num_steps}")
    print(f"Anchor-x0 wt   : {args.anchor_x0_weight}")
    print(f"TTA no-caption : {args.tta_disable_caption}")
    print(f"Resume from idx: {start_idx}")
    print("=" * 70)

    # Load model components
    print("\nLoading LongCat-Video model components...")
    components = load_longcat_components(
        args.checkpoint_dir, device=args.device, dtype=torch.bfloat16
    )
    dit = components["dit"]
    vae = components["vae"]
    pipe = components["pipe"]
    tokenizer = components["tokenizer"]
    text_encoder = components["text_encoder"]

    # Enable gradient checkpointing to reduce activation memory during backprop
    import functools
    from torch.utils.checkpoint import checkpoint as _ckpt_fn
    dit.gradient_checkpointing = True
    dit._gradient_checkpointing_func = functools.partial(_ckpt_fn, use_reentrant=False)
    print("Gradient checkpointing: ENABLED (use_reentrant=False)")

    mem_after_load = log_gpu_mem("After model load")

    # Freeze all DiT parameters
    for p in dit.parameters():
        p.requires_grad = False

    # Inject LoRA -- builtin or custom path
    lora_impl = "builtin" if use_builtin else "custom"
    print(f"\nInjecting LoRA adapters [{lora_impl}] (rank={args.lora_rank}, "
          f"alpha={args.lora_alpha}, blocks={args.lora_target_blocks})...")

    if use_builtin:
        lora_modules = inject_builtin_lora_into_dit(
            dit,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            target_modules=target_modules,
            target_ffn=args.target_ffn,
            target_blocks=args.lora_target_blocks,
        )
        param_counts = count_builtin_lora_parameters(lora_modules)
        _get_lora_params = lambda: get_builtin_lora_parameters(lora_modules)
        _reset_lora = lambda: reset_builtin_lora_weights(lora_modules)
    else:
        lora_modules = inject_lora_into_dit(
            dit,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            target_modules=target_modules,
            target_ffn=args.target_ffn,
            target_blocks=args.lora_target_blocks,
        )
        param_counts = count_lora_parameters(lora_modules)
        _get_lora_params = lambda: get_lora_parameters(lora_modules)
        _reset_lora = lambda: reset_lora_weights(lora_modules)

    total_dit_params = sum(p.numel() for p in dit.parameters())
    print(f"LoRA modules created : {len(lora_modules)}")
    print(f"LoRA trainable params: {param_counts['trainable']:,}")
    print(f"Total DiT params     : {total_dit_params:,}")
    print(f"Trainable %          : {100 * param_counts['trainable'] / total_dit_params:.4f}%")

    mem_after_lora = log_gpu_mem("After LoRA injection")

    # Save experiment config
    exp_config = {
        "method": f"lora_tta_{lora_impl}",
        "lora": {
            "implementation": lora_impl,
            "rank": args.lora_rank,
            "alpha": args.lora_alpha,
            "dropout": getattr(args, "lora_dropout", 0.0),
            "target_modules": target_modules,
            "target_blocks": args.lora_target_blocks,
            "target_ffn": args.target_ffn,
            "num_modules": len(lora_modules),
            "trainable_params": param_counts["trainable"],
        },
        "training": {
            "learning_rate": args.learning_rate,
            "num_steps": args.num_steps,
            "warmup_steps": args.warmup_steps,
            "weight_decay": args.weight_decay,
            "max_grad_norm": args.max_grad_norm,
        },
        "generation": {
            "num_cond_frames": args.num_cond_frames,
            "num_frames": args.num_frames,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "resolution": args.resolution,
        },
        "seed": args.seed,
        "max_videos": args.max_videos,
        "clip_gate_enabled": args.clip_gate_enabled,
        "clip_gate_threshold": args.clip_gate_threshold,
        "clip_gate_backend": args.clip_gate_backend,
        "clip_gate_model": args.clip_gate_model,
        "clip_gate_sample_frames": args.clip_gate_sample_frames,
        "clip_gate_aggregation": args.clip_gate_aggregation,
        "clip_gate_sampling_mode": "late_only" if args.clip_gate_late_only else args.clip_gate_sampling_mode,
        "clip_gate_late_fraction": args.clip_gate_late_fraction,
        "clip_gate_log_only": args.clip_gate_log_only,
        "clip_gate_fail_open": args.clip_gate_fail_open,
        "clip_gate": {
            "enabled": args.clip_gate_enabled,
            "threshold": args.clip_gate_threshold,
            "backend": args.clip_gate_backend,
            "model": args.clip_gate_model,
            "sample_frames": args.clip_gate_sample_frames,
            "aggregation": args.clip_gate_aggregation,
            "sampling_mode": "late_only" if args.clip_gate_late_only else args.clip_gate_sampling_mode,
            "late_fraction": args.clip_gate_late_fraction,
            "log_only": args.clip_gate_log_only,
            "fail_open": args.clip_gate_fail_open,
        },
    }
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(exp_config, f, indent=2)

    # Load evaluation videos (always from --data-dir)
    eval_videos = load_ucf101_video_list(
        args.data_dir, max_videos=args.max_videos, seed=args.seed, validate_decodable=True
    )
    eval_videos = apply_fixed_caption(eval_videos, args.fixed_caption, context="eval")
    validate_caption_quality(
        eval_videos,
        mode=args.caption_guard_mode,
        min_nonempty_ratio=args.caption_guard_min_nonempty_ratio,
        min_unique_ratio=args.caption_guard_min_unique_ratio,
        max_top1_ratio=args.caption_guard_max_top1_ratio,
        max_generic_top1_ratio=args.caption_guard_max_generic_top1_ratio,
        top_k=args.caption_guard_topk,
        context="eval",
    )
    if args.start_video_idx > 0 or args.chunk_size > 0:
        end = len(eval_videos)
        if args.chunk_size > 0:
            end = min(args.start_video_idx + args.chunk_size, end)
        eval_videos = eval_videos[args.start_video_idx:end]
        print(f"Chunk: videos [{args.start_video_idx}:{end}] → {len(eval_videos)} videos")

    print(f"\nEvaluation videos: {len(eval_videos)}")

    # Build retrieval pool for batch-level TTA
    batch_level = args.batch_videos > 1
    pool_entries = None
    pool_embeddings = None
    st_model = None

    if batch_level:
        pool_dir = args.retrieval_pool_dir or args.data_dir
        if pool_dir == args.data_dir:
            print(f"\nWARNING: --retrieval-pool-dir not set; using --data-dir as pool.",
                  file=sys.stderr)
        pool_entries = load_ucf101_video_list(pool_dir, max_videos=999999, seed=args.seed)
        pool_entries = apply_fixed_caption(pool_entries, args.fixed_caption, context="retrieval_pool")
        validate_caption_quality(
            pool_entries,
            mode=args.caption_guard_mode,
            min_nonempty_ratio=args.caption_guard_min_nonempty_ratio,
            min_unique_ratio=args.caption_guard_min_unique_ratio,
            max_top1_ratio=args.caption_guard_max_top1_ratio,
            max_generic_top1_ratio=args.caption_guard_max_generic_top1_ratio,
            top_k=args.caption_guard_topk,
            context="retrieval_pool",
        )
        print(f"Retrieval pool: {len(pool_entries)} videos from {pool_dir}")
        pool_embeddings, st_model = build_retrieval_pool(pool_entries)

    # Build early stopper
    early_stopper = build_early_stopper_from_args(args)
    if early_stopper is not None:
        print(f"[EarlyStopper] Enabled – check_every={early_stopper.check_every}, "
              f"patience={early_stopper.patience}")
    else:
        print("[EarlyStopper] Disabled")

    # Process videos
    print(f"\nProcessing {len(eval_videos) - start_idx} videos...\n")
    videos_dir = os.path.join(args.output_dir, "videos")
    fvd_accumulator = OnlineFrechetAccumulator(
        device=args.device, compute_fid=args.compute_fid,
        min_videos=args.min_fvd_videos,
        gt_cache_path=getattr(args, "gt_features_cache", None),
    ) if args.compute_fvd else None
    fvd_ckpt_path = os.path.join(args.output_dir, "fvd_checkpoint.npz")
    if fvd_accumulator is not None and start_idx > 0:
        fvd_accumulator.load_stats(fvd_ckpt_path)
    lora_dir = os.path.join(args.output_dir, "lora_weights")
    retain_set = set()
    if args.save_only_list:
        import json as _json
        with open(args.save_only_list) as _f:
            _retain = _json.load(_f)
        retain_set = set(_retain.get("all", []))
        print("[Retain] Will save %d videos from %s" % (len(retain_set), args.save_only_list))

    if not args.no_save_videos or retain_set:
        os.makedirs(videos_dir, exist_ok=True)
    if args.save_lora_weights:
        os.makedirs(lora_dir, exist_ok=True)

    for idx, eval_entry in enumerate(tqdm(eval_videos, desc="LoRA TTA")):
        if idx < start_idx:
            continue

        video_path = eval_entry["video_path"]
        caption = eval_entry["caption"]
        video_name = Path(video_path).stem

        print(f"\n[{idx + 1}/{len(eval_videos)}] {video_name}: {caption}")

        clip_gate_info = evaluate_clip_gate(
            video_path=video_path,
            caption=caption,
            gen_start_frame=args.gen_start_frame,
            tta_total_frames=args.tta_total_frames,
            device=args.device,
            enabled=args.clip_gate_enabled,
            threshold=args.clip_gate_threshold,
            backend=args.clip_gate_backend,
            model_name=args.clip_gate_model,
            sample_frames=args.clip_gate_sample_frames,
            aggregation=args.clip_gate_aggregation,
            sampling_mode=args.clip_gate_sampling_mode,
            late_fraction=args.clip_gate_late_fraction,
            late_only=args.clip_gate_late_only,
            fail_open=args.clip_gate_fail_open,
            log_only=args.clip_gate_log_only,
        )
        if clip_gate_info.get("clip_alignment_score") is not None:
            print(
                "  CLIP gate: "
                f"score={clip_gate_info['clip_alignment_score']:.4f}, "
                f"decision={clip_gate_info['clip_gate_decision']}, "
                f"mode={clip_gate_info['clip_gate_sampling_mode']}"
            )
        elif clip_gate_info.get("clip_gate_enabled"):
            print(f"  CLIP gate: decision={clip_gate_info['clip_gate_decision']} "
                  f"({clip_gate_info.get('clip_gate_reason', 'n/a')})")

        # Build training batch: eval video + K-1 nearest neighbours
        if batch_level and not clip_gate_info.get("tta_skipped", False):
            neighbors = retrieve_neighbors(
                eval_entry, pool_entries, pool_embeddings, st_model,
                k=args.batch_videos,
            )
            training_entries = [eval_entry] + neighbors
            print(f"  Batch: 1 eval + {len(neighbors)} retrieved neighbours")
        else:
            training_entries = [eval_entry]
            if batch_level and clip_gate_info.get("tta_skipped", False):
                print("  CLIP gate triggered: skip TTA for this sample, neighbors ignored.")

        try:
            if clip_gate_info.get("tta_skipped", False):
                _reset_lora()
                train_result = {
                    "losses": [],
                    "train_time": 0.0,
                    "es_check_time": 0.0,
                    "early_stopping_info": None,
                }
            elif batch_level:
                # ── Batch-level: pre-encode all videos, train jointly ──
                batch_data = []
                for te in training_entries:
                    tta_start = args.gen_start_frame - args.tta_total_frames
                    pf = load_video_frames(
                        te["video_path"], args.tta_total_frames, height=480, width=832,
                        start_frame=max(0, tta_start),
                    ).to(args.device, torch.bfloat16)

                    al = encode_video(vae, pf, normalize=True)
                    vae_t_scale = 4
                    num_ctx_lat = 1 + (args.tta_context_frames - 1) // vae_t_scale
                    cl, tl, _ = split_tta_latents(
                        al, num_ctx_lat,
                        holdout_fraction=0.0,  # batch path discards val; use all frames
                    )
                    pe, pm = encode_prompt(
                        tokenizer, text_encoder,
                        tta_caption_for(args, te["caption"]),
                        device=args.device, dtype=torch.bfloat16,
                    )
                    batch_data.append({
                        "cond_latents": cl.cpu(), "train_latents": tl.cpu(),
                        "prompt_embeds": pe.cpu(),
                        "prompt_mask": pm.cpu() if pm is not None else None,
                    })
                    del al, pf
                    torch_gc()

                # Reset LoRA and train on batch
                _reset_lora()
                train_result = finetune_lora_batch(
                    dit=dit, lora_modules=lora_modules,
                    batch_data=batch_data,
                    num_steps=args.num_steps, lr=args.learning_rate,
                    warmup_steps=args.warmup_steps,
                    weight_decay=args.weight_decay,
                    max_grad_norm=args.max_grad_norm,
                    device=args.device, dtype=torch.bfloat16,
                    lora_param_fn=_get_lora_params,
                    anchor_x0_weight=args.anchor_x0_weight,
                )
                del batch_data

            else:
                # ── Instance-level: original single-video path ──
                tta_start = args.gen_start_frame - args.tta_total_frames
                pixel_frames = load_video_frames(
                    video_path, args.tta_total_frames, height=480, width=832,
                    start_frame=max(0, tta_start),
                ).to(args.device, torch.bfloat16)

                all_latents = encode_video(vae, pixel_frames, normalize=True)

                vae_t_scale = 4
                num_ctx_lat = 1 + (args.tta_context_frames - 1) // vae_t_scale
                # Only hold out a val split if it will be consumed (early
                # stopping); otherwise adapt on all observed frames rather than
                # discarding ~25% of the TTA signal to an unused holdout.
                _use_val = early_stopper is not None or float(
                    getattr(args, "anchor_reg_weight", 0.0) or 0.0
                ) > 0.0
                cond_latents, train_latents, val_latents = split_tta_latents(
                    all_latents, num_ctx_lat,
                    holdout_fraction=(
                        getattr(args, "es_holdout_fraction", 0.25) if _use_val else 0.0
                    ),
                )

                prompt_embeds, prompt_mask = encode_prompt(
                    tokenizer, text_encoder,
                    tta_caption_for(args, caption),
                    device=args.device, dtype=torch.bfloat16,
                )

                train_latents_variants = None
                if args.aug_enabled:
                    from common import build_augmented_pixel_variants
                    pix_variants = build_augmented_pixel_variants(
                        pixel_frames,
                        enable_flip=args.aug_flip,
                        rotate_deg=args.aug_rotate_deg,
                        rotate_random_min=args.aug_rotate_random_min,
                        rotate_random_max=args.aug_rotate_random_max,
                        rotate_random_count=args.aug_rotate_random_count,
                        rotate_random_step=args.aug_rotate_random_step,
                        rotate_zoom=args.aug_rotate_zoom,
                        speed_factors=parse_speed_factors(args.aug_speed_factors),
                    )
                    train_latents_variants = []
                    for pv in pix_variants:
                        if pv["name"] == "orig":
                            train_latents_variants.append({"latents": train_latents, "name": "orig"})
                        else:
                            aug_lat = encode_video(vae, pv["pixel_frames"], normalize=True)
                            t_start = cond_latents.shape[2]
                            t_end = t_start + train_latents.shape[2]
                            train_latents_variants.append({
                                "latents": aug_lat[:, :, t_start:t_end],
                                "name": pv["name"],
                            })

                _reset_lora()

                if early_stopper is not None and val_latents is not None:
                    def _es_forward_fn(hs, ts, ncl):
                        return dit(
                            hidden_states=hs, timestep=ts,
                            encoder_hidden_states=prompt_embeds,
                            encoder_attention_mask=prompt_mask,
                            num_cond_latents=ncl,
                        )

                    def _save_fn():
                        # Keep snapshot format consistent with finetune_lora_on_conditioning()
                        # so restore_fn can always replay tensors in parameter order.
                        return [p.data.clone() for p in _get_lora_params()]

                    early_stopper.setup(
                        model=dit,
                        cond_latents=cond_latents,
                        val_latents=val_latents,
                        prompt_embeds=prompt_embeds,
                        prompt_mask=prompt_mask,
                        device=args.device,
                        dtype=torch.bfloat16,
                        forward_fn=_es_forward_fn,
                        video_id=video_name,
                        save_fn=_save_fn,
                    )

                train_result = finetune_lora_on_conditioning(
                    dit=dit, lora_modules=lora_modules,
                    cond_latents=cond_latents, train_latents=train_latents,
                    prompt_embeds=prompt_embeds, prompt_mask=prompt_mask,
                    num_steps=args.num_steps, lr=args.learning_rate,
                    warmup_steps=args.warmup_steps,
                    weight_decay=args.weight_decay,
                    max_grad_norm=args.max_grad_norm,
                    device=args.device, dtype=torch.bfloat16,
                    early_stopper=early_stopper if val_latents is not None else None,
                    train_latents_variants=train_latents_variants,
                    lora_param_fn=_get_lora_params,
                    anchor_x0_weight=args.anchor_x0_weight,
                )

                if args.tta_total_frames >= args.num_cond_frames:
                    _cached_gen_cond_frames = pixel_frames[:, :, -args.num_cond_frames:].clone()
                else:
                    _cached_gen_cond_frames = None
                del all_latents, cond_latents, train_latents, val_latents
                del pixel_frames, prompt_embeds, prompt_mask
                torch_gc()

            result = {
                "idx": idx,
                "video_name": video_name,
                "video_path": video_path,
                "caption": caption,
                "train_time": train_result["train_time"],
                "es_check_time": train_result.get("es_check_time", 0.0),
                "final_loss": train_result["losses"][-1] if train_result["losses"] else None,
                "num_train_steps": len(train_result["losses"]),
                "batch_size": len(training_entries),
                "num_neighbors": len(training_entries) - 1,
                "early_stopping_info": train_result.get("early_stopping_info"),
                "lora_delta_l2": train_result.get("lora_delta_l2"),
                "mean_grad_norm": (sum(train_result.get("grad_norms", [])) / len(train_result["grad_norms"])) if train_result.get("grad_norms") else None,
                "gpu_mem_train": train_result.get("gpu_mem_train"),
                "success": True,
            }
            result.update(clip_gate_info)

            # Generate video continuation (eval video only)
            gen_time = 0.0
            if not args.skip_generation:
                from PIL import Image

                num_gen = args.num_frames - args.num_cond_frames
                rollout_steps = args.rollout_steps

                if _cached_gen_cond_frames is not None:
                    gen_pixel_frames = _cached_gen_cond_frames
                    _cached_gen_cond_frames = None
                else:
                    gen_cond_start = args.gen_start_frame - args.num_cond_frames
                    gen_pixel_frames = load_video_frames(
                        video_path, args.num_cond_frames, height=480, width=832,
                        start_frame=max(0, gen_cond_start),
                    ).to(args.device, torch.bfloat16)

                pf = gen_pixel_frames.squeeze(0)
                pf = ((pf + 1.0) / 2.0).clamp(0, 1)
                cond_images = []
                for t_idx in range(pf.shape[1]):
                    frame_np = (pf[:, t_idx].permute(1, 2, 0).float().cpu().numpy() * 255).astype(np.uint8)
                    cond_images.append(Image.fromarray(frame_np))

                all_step_metrics = []
                prev_gen_frames = None

                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats(args.device)

                for step_i in range(rollout_steps):
                    step_gen_start_frame = args.gen_start_frame + step_i * num_gen

                    if step_i > 0:
                        tail = prev_gen_frames[num_gen:]
                        cond_images = []
                        for t_idx in range(tail.shape[0]):
                            frame_np = (np.clip(tail[t_idx], 0, 1) * 255).astype(np.uint8)
                            cond_images.append(Image.fromarray(frame_np))

                    gen_start = time.time()
                    gen_frames = generate_video_continuation(
                        pipe=pipe, video_frames=cond_images, prompt=caption,
                        num_cond_frames=args.num_cond_frames,
                        num_frames=args.num_frames,
                        num_inference_steps=args.num_inference_steps,
                        guidance_scale=args.guidance_scale,
                        seed=args.seed + idx + step_i,
                        resolution=args.resolution, device=args.device,
                    )
                    step_gen_time = time.time() - gen_start
                    gen_time += step_gen_time

                    step_metrics = evaluate_generation_metrics(
                        gen_output=gen_frames, video_path=video_path,
                        num_cond_frames=args.num_cond_frames,
                        num_gen_frames=num_gen,
                        gen_start_frame=step_gen_start_frame, device=args.device,
                        return_gt_frames=(step_i == 0 and fvd_accumulator is not None),
                    )
                    _gt_for_fvd = step_metrics.pop("gt_frames_hwc", None)
                    all_step_metrics.append(step_metrics)

                    if step_i == 0 and fvd_accumulator is not None:
                        fvd_accumulator.update(gen_frames, video_path,
                                               args.num_cond_frames, num_gen, args.gen_start_frame,
                                               gt_frames_hwc=_gt_for_fvd)

                    if step_i == 0:
                        output_path = os.path.join(videos_dir, f"{video_name}_lora.mp4")
                        should_save = (not args.no_save_videos) or (video_name in retain_set)
                        if should_save:
                            save_video_from_numpy(gen_frames, output_path, fps=args.fps)
                            result["output_path"] = output_path

                    prev_gen_frames = gen_frames

                gen_mem = gpu_mem_stats(args.device) if torch.cuda.is_available() else {}
                result["gen_time"] = gen_time
                result["rollout_steps"] = rollout_steps
                result["gpu_mem_gen"] = gen_mem

                for si, sm in enumerate(all_step_metrics):
                    for mk in ("psnr", "ssim", "lpips"):
                        result["step_%d_%s" % (si + 1, mk)] = sm.get(mk)

                avg_metrics = {}
                for mk in ("psnr", "ssim", "lpips"):
                    vals = [sm[mk] for sm in all_step_metrics if sm.get(mk) is not None and sm[mk] == sm[mk]]
                    avg_metrics[mk] = float(np.mean(vals)) if vals else float("nan")
                result.update(avg_metrics)

                print("    Metrics: PSNR=%.2f, SSIM=%.4f, LPIPS=%.4f" % (
                    avg_metrics["psnr"], avg_metrics["ssim"], avg_metrics["lpips"]))
                if rollout_steps > 1:
                    print("    Rollout: " + ", ".join(
                        "step%d PSNR=%.2f" % (si + 1, sm.get("psnr", float("nan")))
                        for si, sm in enumerate(all_step_metrics)))

                del gen_pixel_frames
                torch_gc()

            result["total_time"] = (
                float(clip_gate_info.get("clip_gate_eval_time", 0.0))
                + train_result["train_time"]
                + gen_time
            )

            if args.save_lora_weights:
                lora_path = os.path.join(lora_dir, f"{video_name}_lora.pt")
                save_lora_weights(lora_modules, lora_path)

            train_peak = train_result.get("gpu_mem_train", {}).get("peak_gib", 0)
            gen_peak = result.get("gpu_mem_gen", {}).get("peak_gib", 0)
            print(
                  f"  CLIP: {clip_gate_info.get('clip_gate_eval_time', 0.0):.2f}s, "
                  f"ES: {train_result.get('es_check_time', 0.0):.2f}s, "
                  f"Train: {train_result['train_time']:.1f}s, "
                  + (f"Loss: {result['final_loss']:.4f}"
                     if result.get("final_loss") is not None else "Loss: N/A (skipped)")
                  + (f", Gen: {gen_time:.1f}s" if not args.skip_generation else "")
                  + f" | GPU peak: train={train_peak:.1f}G gen={gen_peak:.1f}G")

            all_results.append(result)

        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()
            all_results.append({
                "idx": idx, "video_name": video_name,
                "video_path": video_path, "error": str(e), "success": False,
            })

        save_checkpoint({"next_idx": idx + 1, "results": all_results}, ckpt_path)
        if fvd_accumulator is not None:
            fvd_accumulator.save_stats(fvd_ckpt_path)

    # Save final results
    successful = [r for r in all_results if r.get("success", False)]
    # Aggregate GPU memory stats
    train_peaks = [r.get("gpu_mem_train", {}).get("peak_gib", 0)
                   for r in successful if r.get("gpu_mem_train")]
    gen_peaks = [r.get("gpu_mem_gen", {}).get("peak_gib", 0)
                 for r in successful if r.get("gpu_mem_gen")]
    gpu_total = successful[0].get("gpu_mem_train", {}).get("total_gib", 0) if successful else 0

    summary = {
        "method": "lora_tta",
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "learning_rate": args.learning_rate,
        "num_steps": args.num_steps,
        "num_cond_frames": args.num_cond_frames,
        "num_frames": args.num_frames,
        "gen_start_frame": args.gen_start_frame,
        "batch_videos": args.batch_videos,
        "retrieval_pool_dir": args.retrieval_pool_dir,
        "num_videos": len(all_results),
        "num_successful": len(successful),
        "num_failed": len(all_results) - len(successful),
        "avg_train_time": (
            np.mean([r["train_time"] for r in successful]) if successful else 0
        ),
        "avg_clip_gate_eval_time": (
            np.mean([r.get("clip_gate_eval_time", 0.0) for r in successful]) if successful else 0
        ),
        "avg_es_check_time": (
            np.mean([r.get("es_check_time", 0.0) for r in successful]) if successful else 0
        ),
        "avg_gen_time": (
            np.mean([r.get("gen_time", 0.0) for r in successful]) if successful else 0
        ),
        "avg_total_time": (
            np.mean([r.get("total_time", 0.0) for r in successful]) if successful else 0
        ),
        "avg_final_loss": (
            np.mean([r["final_loss"] for r in successful if r.get("final_loss") is not None])
            if successful else 0
        ),
        "clip_gate_enabled": args.clip_gate_enabled,
        "clip_gate_threshold": args.clip_gate_threshold,
        "clip_gate_model": args.clip_gate_model,
        "clip_gate_sample_frames": args.clip_gate_sample_frames,
        "clip_gate_aggregation": args.clip_gate_aggregation,
        "clip_gate_sampling_mode": "late_only" if args.clip_gate_late_only else args.clip_gate_sampling_mode,
        "clip_gate_late_fraction": args.clip_gate_late_fraction,
        "clip_gate_log_only": args.clip_gate_log_only,
        "clip_gate_fail_open": args.clip_gate_fail_open,
        "clip_gate_stats": summarize_clip_gate_stats(successful),
        "gpu_memory": {
            "total_gib": gpu_total,
            "model_load_gib": mem_after_load.get("allocated_gib", 0),
            "after_lora_gib": mem_after_lora.get("allocated_gib", 0),
            "train_peak_mean_gib": round(float(np.mean(train_peaks)), 2) if train_peaks else 0,
            "train_peak_max_gib": round(float(np.max(train_peaks)), 2) if train_peaks else 0,
            "gen_peak_mean_gib": round(float(np.mean(gen_peaks)), 2) if gen_peaks else 0,
            "gen_peak_max_gib": round(float(np.max(gen_peaks)), 2) if gen_peaks else 0,
        },
        "results": all_results,
    }
    aggregate_quality_metrics(summary)
    finalize_online_eval(fvd_accumulator, summary, videos_dir, args)
    save_results(summary, os.path.join(args.output_dir, "summary.json"))

    n_ok = len(successful)
    n_all = len(all_results)
    print("\n" + "=" * 70)
    print("LoRA TTA Complete!")
    print("=" * 70)
    print("Successful: %d/%d" % (n_ok, n_all))
    if successful:
        clip_t = summary.get("avg_clip_gate_eval_time", 0)
        es_t = summary.get("avg_es_check_time", 0)
        train_t = summary.get("avg_train_time", 0)
        gen_t = summary.get("avg_gen_time", 0)
        total_t = summary.get("avg_total_time", 0)
        avg_loss = summary.get("avg_final_loss")
        print("Avg CLIP gate time: %.2fs" % clip_t)
        print("Avg ES check time : %.2fs" % es_t)
        print("Avg train time: %.1fs" % train_t)
        print("Avg gen time: %.1fs" % gen_t)
        print("Avg total time: %.1fs" % total_t)
        if avg_loss is not None:
            print("Avg final loss: %.4f" % avg_loss)
        else:
            print("Avg final loss: N/A")
    gpu_info = summary.get("gpu_memory", {})
    if gpu_info:
        print("--- GPU Memory ---")
        print("  GPU total:       %.1f GiB" % gpu_info.get("total_gib", 0))
        print("  Model load:      %.2f GiB" % gpu_info.get("model_load_gib", 0))
        print("  After LoRA:      %.2f GiB" % gpu_info.get("after_lora_gib", 0))
        print("  Train peak (avg): %.2f GiB" % gpu_info.get("train_peak_mean_gib", 0))
        print("  Train peak (max): %.2f GiB" % gpu_info.get("train_peak_max_gib", 0))
        print("  Gen peak (avg):  %.2f GiB" % gpu_info.get("gen_peak_mean_gib", 0))
        print("  Gen peak (max):  %.2f GiB" % gpu_info.get("gen_peak_max_gib", 0))
        if gpu_info.get("total_gib"):
            headroom = gpu_info["total_gib"] - gpu_info.get("train_peak_max_gib", 0)
            print("  Headroom:        %.1f GiB (%.0f%%)" % (
                headroom, 100 * headroom / gpu_info["total_gib"]))
    print("Results saved to: %s" % args.output_dir)
    print("=" * 70)

if __name__ == "__main__":
    main()
