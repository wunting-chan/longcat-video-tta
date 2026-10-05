#!/usr/bin/env python3
"""Benchmark batched independent test-time adaptation throughput.

This script is intentionally separate from ``batch_videos`` in the sweep
runner. ``batch_videos`` means retrieval-augmented shared adaptation; this
script measures whether independent per-video adapters can be processed in
parallel on one GPU.

Implemented paths:
  * adasteer_batched: true batched independent AdaSteer deltas [B, 512].
  * lora_serial: serial LoRA baseline using the existing runner.
  * tinylora_serial: serial TinyLoRA baseline using the existing runner.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW

REPO_ROOT = Path(__file__).resolve().parents[2]
DELTA_SCRIPTS = REPO_ROOT / "delta_experiment" / "scripts"
sys.path.insert(0, str(DELTA_SCRIPTS))
sys.path.insert(0, str(REPO_ROOT))


def parse_batch_sizes(raw: str) -> List[int]:
    sizes = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not sizes or any(x < 1 for x in sizes):
        raise argparse.ArgumentTypeError("batch sizes must be positive integers")
    return sizes


def gpu_mem_stats(device: str = "cuda") -> Dict[str, float]:
    if not torch.cuda.is_available():
        return {
            "allocated_gib": 0.0,
            "reserved_gib": 0.0,
            "peak_gib": 0.0,
            "total_gib": 0.0,
        }
    total = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
    return {
        "allocated_gib": torch.cuda.memory_allocated(device) / (1024 ** 3),
        "reserved_gib": torch.cuda.memory_reserved(device) / (1024 ** 3),
        "peak_gib": torch.cuda.max_memory_allocated(device) / (1024 ** 3),
        "total_gib": total,
    }


def load_video_entries(args) -> List[Dict]:
    from common import (
        apply_fixed_caption,
        load_panda70m_video_list,
        load_ucf101_video_list,
        validate_caption_quality,
    )

    data_dir_lower = args.data_dir.lower()
    if "ucf" in data_dir_lower:
        videos = load_ucf101_video_list(
            args.data_dir,
            max_videos=args.max_videos,
            seed=args.seed,
            validate_decodable=args.validate_decodable,
        )
    else:
        videos = load_panda70m_video_list(
            args.data_dir,
            max_videos=args.max_videos,
            seed=args.seed,
            validate_decodable=args.validate_decodable,
        )
    videos = apply_fixed_caption(videos, args.fixed_caption, context="eval")
    validate_caption_quality(
        videos,
        mode=args.caption_guard_mode,
        min_nonempty_ratio=args.caption_guard_min_nonempty_ratio,
        min_unique_ratio=args.caption_guard_min_unique_ratio,
        max_top1_ratio=args.caption_guard_max_top1_ratio,
        max_generic_top1_ratio=args.caption_guard_max_generic_top1_ratio,
        top_k=args.caption_guard_topk,
        context="eval",
    )
    return videos


class BatchedDeltaAWrapper(nn.Module):
    """AdaSteer wrapper with one independent delta per batch row."""

    def __init__(self, dit: nn.Module, batch_size: int, adaln_tembed_dim: int = 512):
        super().__init__()
        self.dit = dit
        for p in self.dit.parameters():
            p.requires_grad = False
        self.delta = nn.Parameter(torch.zeros(batch_size, adaln_tembed_dim))

    @property
    def config(self):
        return self.dit.config

    def forward(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        encoder_attention_mask=None,
        num_cond_latents=0,
        **kwargs,
    ):
        dit = self.dit
        B, _, T, H, W = hidden_states.shape
        if B != self.delta.shape[0]:
            raise ValueError(f"batch mismatch: hidden B={B}, delta B={self.delta.shape[0]}")

        N_t = T // dit.patch_size[0]
        N_h = H // dit.patch_size[1]
        N_w = W // dit.patch_size[2]

        if len(timestep.shape) == 1:
            timestep = timestep.unsqueeze(1).expand(-1, N_t)

        dtype = dit.x_embedder.proj.weight.dtype
        hidden_states = hidden_states.to(dtype)
        timestep = timestep.to(dtype)
        encoder_hidden_states = encoder_hidden_states.to(dtype)

        hidden_states = dit.x_embedder(hidden_states)

        import torch.amp as amp

        with amp.autocast(device_type="cuda", dtype=torch.float32):
            t = dit.t_embedder(
                timestep.float().flatten(), dtype=torch.float32
            ).reshape(B, N_t, -1)

        t = t + self.delta[:, None, :].to(dtype=t.dtype)
        encoder_hidden_states = dit.y_embedder(encoder_hidden_states)

        if dit.text_tokens_zero_pad and encoder_attention_mask is not None:
            encoder_hidden_states = (
                encoder_hidden_states * encoder_attention_mask[:, None, :, None]
            )
            encoder_attention_mask = (
                encoder_attention_mask * 0 + 1
            ).to(encoder_attention_mask.dtype)

        if encoder_attention_mask is not None:
            encoder_attention_mask = encoder_attention_mask.squeeze(1).squeeze(1)
            encoder_hidden_states = (
                encoder_hidden_states.squeeze(1)
                .masked_select(encoder_attention_mask.unsqueeze(-1) != 0)
                .view(1, -1, hidden_states.shape[-1])
            )
            y_seqlens = encoder_attention_mask.sum(dim=1).tolist()
        else:
            y_seqlens = [encoder_hidden_states.shape[2]] * encoder_hidden_states.shape[0]
            encoder_hidden_states = encoder_hidden_states.squeeze(1).view(
                1, -1, hidden_states.shape[-1]
            )

        import functools as _ft
        from torch.utils.checkpoint import checkpoint as _ckpt_fn

        _ckpt = _ft.partial(_ckpt_fn, use_reentrant=False)
        for block in dit.blocks:
            if torch.is_grad_enabled():
                hidden_states = _ckpt(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    t,
                    y_seqlens,
                    (N_t, N_h, N_w),
                    num_cond_latents=num_cond_latents,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    encoder_hidden_states,
                    t,
                    y_seqlens,
                    (N_t, N_h, N_w),
                    num_cond_latents=num_cond_latents,
                )

        hidden_states = dit.final_layer(hidden_states, t, (N_t, N_h, N_w))
        hidden_states = dit.unpatchify(hidden_states, N_t, N_h, N_w)
        return hidden_states.to(torch.float32)


def prepare_batch(
    entries: List[Dict],
    vae,
    tokenizer,
    text_encoder,
    args,
) -> Dict[str, torch.Tensor]:
    from common import (
        encode_prompt,
        encode_video,
        load_video_frames,
        split_tta_latents,
        torch_gc,
    )

    conds = []
    trains = []
    prompts = []
    masks = []

    tta_start = args.gen_start_frame - args.tta_total_frames
    vae_t_scale = 4
    num_ctx_lat = 1 + (args.tta_context_frames - 1) // vae_t_scale

    for entry in entries:
        pixel_frames = load_video_frames(
            entry["video_path"],
            args.tta_total_frames,
            height=480,
            width=832,
            start_frame=max(0, tta_start),
        ).to(args.device, torch.bfloat16)
        all_latents = encode_video(vae, pixel_frames, normalize=True)
        cond_latents, train_latents, _val_latents = split_tta_latents(
            all_latents,
            num_ctx_lat,
            holdout_fraction=0.25,
        )
        prompt_embeds, prompt_mask = encode_prompt(
            tokenizer,
            text_encoder,
            entry["caption"],
            device=args.device,
            dtype=torch.bfloat16,
        )
        conds.append(cond_latents.cpu())
        trains.append(train_latents.cpu())
        prompts.append(prompt_embeds.cpu())
        masks.append(prompt_mask.cpu() if prompt_mask is not None else None)
        del pixel_frames, all_latents, cond_latents, train_latents, prompt_embeds, prompt_mask
        torch_gc()

    prompt_mask = None if any(m is None for m in masks) else torch.cat(masks, dim=0)
    return {
        "cond_latents": torch.cat(conds, dim=0),
        "train_latents": torch.cat(trains, dim=0),
        "prompt_embeds": torch.cat(prompts, dim=0),
        "prompt_mask": prompt_mask,
    }


def iter_groups(videos: List[Dict], batch_size: int, max_groups: int) -> Iterable[List[Dict]]:
    total = len(videos) // batch_size
    if max_groups > 0:
        total = min(total, max_groups)
    for i in range(total):
        yield videos[i * batch_size : (i + 1) * batch_size]


def benchmark_adasteer_batched(args, videos: List[Dict]) -> List[Dict]:
    from common import compute_flow_matching_loss_conditioned, load_longcat_components, torch_gc

    components = load_longcat_components(args.checkpoint_dir, device=args.device)
    tokenizer = components["tokenizer"]
    text_encoder = components["text_encoder"]
    vae = components["vae"]
    dit = components["dit"]

    adaln_dim = getattr(dit.config, "adaln_tembed_dim", 512)
    records = []

    for batch_size in args.batch_sizes:
        if len(videos) < batch_size:
            records.append({
                "method": "adasteer_batched",
                "requested_batch_size": batch_size,
                "status": "skipped",
                "reason": "not enough videos",
            })
            continue

        print(f"\n=== AdaSteer batched B={batch_size} ===", flush=True)
        batch_records = []
        oom = False
        for group_idx, group in enumerate(iter_groups(videos, batch_size, args.max_groups)):
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(args.device)

                encode_t0 = time.time()
                batch = prepare_batch(group, vae, tokenizer, text_encoder, args)
                encode_time = time.time() - encode_t0

                vae.to("cpu")
                text_encoder.to("cpu")
                torch.cuda.empty_cache()

                wrapper = BatchedDeltaAWrapper(
                    dit,
                    batch_size=len(group),
                    adaln_tembed_dim=adaln_dim,
                ).to(args.device)
                optimizer = AdamW([wrapper.delta], lr=args.delta_lr, betas=(0.9, 0.999), eps=1e-15)

                cond = batch["cond_latents"].to(args.device)
                train = batch["train_latents"].to(args.device)
                prompts = batch["prompt_embeds"].to(args.device)
                mask = batch["prompt_mask"].to(args.device) if batch["prompt_mask"] is not None else None

                losses = []
                train_t0 = time.time()
                wrapper.train()
                for _step in range(args.delta_steps):
                    optimizer.zero_grad()
                    loss = compute_flow_matching_loss_conditioned(
                        dit=wrapper,
                        cond_latents=cond,
                        target_latents=train,
                        prompt_embeds=prompts,
                        prompt_mask=mask,
                        device=args.device,
                        dtype=torch.bfloat16,
                    )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_([wrapper.delta], 1.0)
                    optimizer.step()
                    losses.append(float(loss.item()))

                train_time = time.time() - train_t0
                mem = gpu_mem_stats(args.device)
                batch_records.append({
                    "group_idx": group_idx,
                    "videos": [Path(e["video_path"]).stem for e in group],
                    "encode_seconds": encode_time,
                    "train_seconds": train_time,
                    "train_seconds_per_video": train_time / len(group),
                    "total_seconds": encode_time + train_time,
                    "total_seconds_per_video": (encode_time + train_time) / len(group),
                    "final_loss": losses[-1] if losses else None,
                    "losses": losses,
                    "gpu_memory": mem,
                })

                del batch, cond, train, prompts, mask, wrapper, optimizer
                vae.to(args.device)
                text_encoder.to(args.device)
                torch_gc()
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                oom = True
                print(f"OOM at B={batch_size}: {exc}", file=sys.stderr, flush=True)
                torch.cuda.empty_cache()
                torch_gc()
                break

        if oom:
            records.append({
                "method": "adasteer_batched",
                "requested_batch_size": batch_size,
                "status": "oom",
                "groups": batch_records,
            })
            break

        if batch_records:
            train_times = [r["train_seconds_per_video"] for r in batch_records]
            total_times = [r["total_seconds_per_video"] for r in batch_records]
            peak = max(r["gpu_memory"]["peak_gib"] for r in batch_records)
            records.append({
                "method": "adasteer_batched",
                "requested_batch_size": batch_size,
                "actual_batch_size": batch_size,
                "status": "ok",
                "groups": batch_records,
                "mean_train_seconds_per_video": sum(train_times) / len(train_times),
                "mean_total_seconds_per_video": sum(total_times) / len(total_times),
                "peak_memory_gib": peak,
            })
        else:
            records.append({
                "method": "adasteer_batched",
                "requested_batch_size": batch_size,
                "status": "skipped",
                "reason": "no full groups",
            })

    return records


def run_serial_baseline(args, method: str) -> Dict:
    out_dir = Path(args.output_dir) / method
    out_dir.mkdir(parents=True, exist_ok=True)

    if method == "lora_serial":
        cmd = [
            sys.executable,
            str(REPO_ROOT / "lora_experiment" / "scripts" / "run_lora_tta.py"),
            "--checkpoint-dir", args.checkpoint_dir,
            "--data-dir", args.data_dir,
            "--output-dir", str(out_dir),
            "--max-videos", str(args.max_videos),
            "--lora-rank", "8",
            "--lora-alpha", "16",
            "--lora-target-blocks", "all",
            "--learning-rate", "5e-5",
            "--num-steps", "10",
            "--warmup-steps", "3",
            "--weight-decay", "0.01",
            "--max-grad-norm", "10.0",
            "--num-cond-frames", str(args.num_cond_frames),
            "--num-frames", str(args.num_frames),
            "--gen-start-frame", str(args.gen_start_frame),
            "--tta-total-frames", str(args.tta_total_frames),
            "--tta-context-frames", str(args.tta_context_frames),
            "--num-inference-steps", str(args.num_inference_steps),
            "--guidance-scale", str(args.guidance_scale),
            "--resolution", args.resolution,
            "--seed", str(args.seed),
            "--skip-generation",
            "--no-save-videos",
            "--caption-guard-mode", args.caption_guard_mode,
            "--feature-frame-guard-mode", args.feature_frame_guard_mode,
            "--es-disable",
        ]
    elif method == "tinylora_serial":
        cmd = [
            sys.executable,
            str(REPO_ROOT / "delta_experiment" / "scripts" / "run_tinylora.py"),
            "--checkpoint-dir", args.checkpoint_dir,
            "--data-dir", args.data_dir,
            "--output-dir", str(out_dir),
            "--max-videos", str(args.max_videos),
            "--svd-rank", "2",
            "--n-tie", "1",
            "--alpha", "1.0",
            "--target-preset", "qkv_proj",
            "--target-blocks", "last_24",
            "--tta-steps", "20",
            "--tta-lr", "1e-3",
            "--num-cond-frames", str(args.num_cond_frames),
            "--num-frames", str(args.num_frames),
            "--gen-start-frame", str(args.gen_start_frame),
            "--tta-total-frames", str(args.tta_total_frames),
            "--tta-context-frames", str(args.tta_context_frames),
            "--num-inference-steps", str(args.num_inference_steps),
            "--guidance-scale", str(args.guidance_scale),
            "--resolution", args.resolution,
            "--seed", str(args.seed),
            "--skip-generation",
            "--no-save-videos",
            "--caption-guard-mode", args.caption_guard_mode,
            "--feature-frame-guard-mode", args.feature_frame_guard_mode,
            "--es-disable",
        ]
    else:
        raise ValueError(f"unknown serial baseline: {method}")

    printable = " ".join(shlex.quote(x) for x in cmd)
    if args.dry_run:
        return {
            "method": method,
            "status": "dry_run",
            "command": printable,
            "effective_parallel_batch_size": 1,
        }

    t0 = time.time()
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    elapsed = time.time() - t0
    summary_path = out_dir / "summary.json"
    summary = {}
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
    videos_done = summary.get("videos_done") or args.max_videos
    return {
        "method": method,
        "status": "ok",
        "command": printable,
        "elapsed_seconds": elapsed,
        "seconds_per_video": elapsed / max(float(videos_done), 1.0),
        "effective_parallel_batch_size": 1,
        "summary_path": str(summary_path),
        "runner_summary": summary,
    }


def write_summary(args, records: List[Dict]) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "benchmark": "batched_independent_tta",
        "data_dir": args.data_dir,
        "checkpoint_dir": args.checkpoint_dir,
        "frame_config": {
            "num_cond_frames": args.num_cond_frames,
            "num_frames": args.num_frames,
            "gen_start_frame": args.gen_start_frame,
            "tta_total_frames": args.tta_total_frames,
            "tta_context_frames": args.tta_context_frames,
        },
        "delta_config": {
            "delta_steps": args.delta_steps,
            "delta_lr": args.delta_lr,
        },
        "records": records,
    }
    path = out_dir / "batched_tta_benchmark.json"
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote benchmark summary: {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--methods",
        default="adasteer_batched",
        help="Comma-separated: adasteer_batched,lora_serial,tinylora_serial,all",
    )
    parser.add_argument("--batch-sizes", type=parse_batch_sizes, default=parse_batch_sizes("1,2,4,8,16"))
    parser.add_argument("--max-videos", type=int, default=16)
    parser.add_argument("--max-groups", type=int, default=1)
    parser.add_argument("--delta-steps", type=int, default=10)
    parser.add_argument("--delta-lr", type=float, default=5e-3)
    parser.add_argument("--num-cond-frames", type=int, default=14)
    parser.add_argument("--num-frames", type=int, default=28)
    parser.add_argument("--gen-start-frame", type=int, default=48)
    parser.add_argument("--tta-total-frames", type=int, default=48)
    parser.add_argument("--tta-context-frames", type=int, default=14)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--resolution", default="480p")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fixed-caption", default=None)
    parser.add_argument("--caption-guard-mode", default="warn", choices=["fail", "warn", "off"])
    parser.add_argument("--caption-guard-min-nonempty-ratio", type=float, default=0.95)
    parser.add_argument("--caption-guard-min-unique-ratio", type=float, default=0.10)
    parser.add_argument("--caption-guard-max-top1-ratio", type=float, default=0.50)
    parser.add_argument("--caption-guard-max-generic-top1-ratio", type=float, default=0.20)
    parser.add_argument("--caption-guard-topk", type=int, default=5)
    parser.add_argument("--feature-frame-guard-mode", default="warn")
    parser.add_argument("--validate-decodable", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    if "all" in methods:
        methods = ["adasteer_batched", "lora_serial", "tinylora_serial"]

    if args.dry_run:
        print("Dry run: benchmark configuration")
        print(json.dumps({
            "methods": methods,
            "batch_sizes": args.batch_sizes,
            "max_videos": args.max_videos,
            "max_groups": args.max_groups,
            "data_dir": args.data_dir,
            "output_dir": args.output_dir,
        }, indent=2))

    records: List[Dict] = []
    videos: Optional[List[Dict]] = None
    if "adasteer_batched" in methods:
        if args.dry_run:
            records.append({
                "method": "adasteer_batched",
                "status": "dry_run",
                "requested_batch_sizes": args.batch_sizes,
            })
        else:
            videos = load_video_entries(args)
            records.extend(benchmark_adasteer_batched(args, videos))

    for method in methods:
        if method in {"lora_serial", "tinylora_serial"}:
            records.append(run_serial_baseline(args, method))
        elif method != "adasteer_batched":
            raise ValueError(f"unknown method: {method}")

    write_summary(args, records)


if __name__ == "__main__":
    main()
