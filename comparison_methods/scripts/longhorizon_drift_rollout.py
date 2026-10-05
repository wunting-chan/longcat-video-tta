#!/usr/bin/env python3
"""
Long-horizon DRIFT diagnostic: TRUE autoregressive rollout on LongCat.

WHY THIS EXISTS
---------------
Our whole sweep (AdaSteer / TANGO / placement) has produced null/tiny effects.
The 2026-08-06 problem-difficulty audit
(`sweep_experiment/reports/paper_tables/2026-08-06_problem_difficulty_field_geometry.md`)
concluded we may have set LongCat too easy a task: a single 14->14 in-domain
continuation on a 13.6B RLHF model whose headline capability is drift-free
continuation. The whole long-video literature (Rolling Forcing / BAgger /
Self-Forcing / Pathwise TTC / Meta-ARVDM) gets its headroom from ERROR
ACCUMULATION over LONG AUTOREGRESSIVE rollouts -- a regime our pipeline never
enters, because even `ttc_longcat.py` (93 frames) generates all gen frames in a
SINGLE diffusion call (no chaining, no exposure bias, no drift).

This script closes that gap. It performs a *true* autoregressive rollout:

    chunk 0 : condition on GT[0:num_cond]                 -> generate chunk_gen
    chunk c : condition on the model's OWN last num_cond  -> generate chunk_gen
              frames from chunk c-1

so errors compound across chunks exactly as in the streaming literature. We then
measure per-chunk quality to produce a DRIFT CURVE (quality vs chunk index).

WHAT IT MEASURES (per chunk, then aggregated across videos by the plotter)
--------------------------------------------------------------------------
GT-free drift signatures (the honest degradation signals; no future GT needed,
so they are valid even though an AR trajectory legitimately diverges from GT):
  * sharpness       : variance of the Laplacian (blur / over-smoothing -> down)
  * motion          : mean |frame_t - frame_{t-1}| (freezing -> down; chaos -> up)
  * colorfulness    : Hasler-Susstrunk colorfulness (fading -> down)
  * saturation      : mean (max_c - min_c) per pixel (over-saturation drift)
  * brightness      : mean pixel value (exposure drift)
  * seam            : |first frame of chunk c - last frame of chunk c-1| vs the
                      within-chunk adjacent-frame diff (cross-chunk discontinuity,
                      cf. STAS). Larger seam = worse chunk stitching.

Optional GT-anchored divergence (best-effort; NaN once the source clip runs out):
  * psnr_vs_gt per chunk -- "how fast does the trajectory leave the real future".
    This conflates plausible-but-different with degraded, so it is reported as a
    secondary signal, not the headline.

DETERMINISM
-----------
Per (video, chunk) the sampling noise generator is seeded deterministically as
    seed + video_idx*1000 + chunk_idx
so a rerun (or a TTA variant sharing this convention) draws identical noise for
the same (video, chunk) -- apples-to-apples across policies.

This is a NOTTA baseline (plain Euler, CFG on). A TTA/steering variant can reuse
the exact rollout by swapping the per-chunk sampler; kept out of scope here so
the diagnostic first answers the single question: DOES QUALITY DRIFT AT ALL under
long AR rollout on LongCat? If yes -> headroom exists. If no -> LongCat is too
strong for this framing and we should switch base model.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[1]
_LONGCAT_DIR = _REPO_ROOT / "LongCat-Video"
_DELTA_SCRIPTS = _REPO_ROOT / "delta_experiment" / "scripts"
sys.path.insert(0, str(_LONGCAT_DIR))
sys.path.insert(0, str(_DELTA_SCRIPTS))
sys.path.insert(0, str(_REPO_ROOT))

from common import (  # noqa: E402
    load_longcat_components,
    load_video_frames,
    encode_video,
    decode_latents,
    encode_prompt,
    load_ucf101_video_list,
    apply_fixed_caption,
    save_results,
    load_checkpoint,
    save_checkpoint,
    torch_gc,
)
from comparison_methods.scripts.savi_dno_longcat import SAViDNO_LongCat  # noqa: E402


# ----------------------------------------------------------------------------
# GT-free per-frame drift signals (operate on [T,H,W,C] float in [0,1])
# ----------------------------------------------------------------------------
def _to_gray(frames: np.ndarray) -> np.ndarray:
    """[T,H,W,C] -> [T,H,W] luminance."""
    w = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    return frames[..., :3] @ w


def _laplacian_var(gray: np.ndarray) -> np.ndarray:
    """Per-frame variance of a 4-neighbour Laplacian. Blur -> lower."""
    g = gray
    lap = (
        -4.0 * g
        + np.roll(g, 1, axis=1) + np.roll(g, -1, axis=1)
        + np.roll(g, 1, axis=2) + np.roll(g, -1, axis=2)
    )
    # drop the 1px border (roll wrap-around) before taking variance
    lap = lap[:, 1:-1, 1:-1]
    return lap.reshape(lap.shape[0], -1).var(axis=1)


def _colorfulness(frames: np.ndarray) -> np.ndarray:
    """Hasler-Susstrunk colorfulness per frame."""
    r, g, b = frames[..., 0], frames[..., 1], frames[..., 2]
    rg = r - g
    yb = 0.5 * (r + g) - b
    rg2 = rg.reshape(rg.shape[0], -1)
    yb2 = yb.reshape(yb.shape[0], -1)
    std = np.sqrt(rg2.var(axis=1) + yb2.var(axis=1))
    mean = np.sqrt(rg2.mean(axis=1) ** 2 + yb2.mean(axis=1) ** 2)
    return std + 0.3 * mean


def _saturation(frames: np.ndarray) -> np.ndarray:
    mx = frames[..., :3].max(axis=-1)
    mn = frames[..., :3].min(axis=-1)
    s = (mx - mn)
    return s.reshape(s.shape[0], -1).mean(axis=1)


def _brightness(frames: np.ndarray) -> np.ndarray:
    return frames[..., :3].reshape(frames.shape[0], -1).mean(axis=1)


def _motion(frames: np.ndarray) -> np.ndarray:
    """Per-frame mean |f_t - f_{t-1}| (first frame = NaN)."""
    d = np.abs(np.diff(frames[..., :3], axis=0)).reshape(frames.shape[0] - 1, -1).mean(axis=1)
    return np.concatenate([[np.nan], d])


def per_frame_signals(frames: np.ndarray) -> dict:
    """frames [T,H,W,C] in [0,1] -> dict of per-frame 1-D arrays (as lists)."""
    gray = _to_gray(frames)
    return {
        "sharpness": _laplacian_var(gray).tolist(),
        "colorfulness": _colorfulness(frames).tolist(),
        "saturation": _saturation(frames).tolist(),
        "brightness": _brightness(frames).tolist(),
        "motion": _motion(frames).tolist(),
    }


def _nanmean(a):
    a = np.asarray(a, dtype=np.float64)
    return float(np.nanmean(a)) if a.size else float("nan")


# ----------------------------------------------------------------------------
# Plain Euler chunk sampler (faithful NOTTA; == TTC with correction weight 0)
# ----------------------------------------------------------------------------
@torch.no_grad()
def sample_chunk(engine: SAViDNO_LongCat, cond_latents, target_shape,
                 prompt_embeds, prompt_mask, use_cfg=True, generator=None):
    sigmas = engine._build_sigmas()
    x_t = torch.randn(target_shape, device=engine.device, dtype=torch.float32,
                      generator=generator)
    step_fn = engine._dit_forward_step_cfg if use_cfg else engine._dit_forward_step
    for i in range(len(sigmas) - 1):
        t_curr = sigmas[i].item()
        dt = sigmas[i + 1].item() - t_curr
        v = step_fn(x_t, cond_latents, t_curr, prompt_embeds, prompt_mask).to(x_t.dtype)
        x_t = x_t + dt * v
    return decode_latents(engine.vae, x_t, denorm=True)  # [1,C,T,H,W] in [0,1]


def _pixels_to_hwc(pred_pixels: torch.Tensor) -> np.ndarray:
    """[1,C,T,H,W] in [0,1] -> [T,H,W,C]."""
    x = pred_pixels.squeeze(0).float().cpu().numpy()      # [C,T,H,W]
    return np.clip(x.transpose(1, 2, 3, 0), 0, 1)         # [T,H,W,C]


def _load_gt_hwc(video_path, start_frame, num_frames, height, width):
    """Best-effort GT frames [T,H,W,C] in [0,1] from absolute start_frame."""
    from PIL import Image
    import av
    container = av.open(video_path)
    out = []
    decoded = 0
    for frame in container.decode(video=0):
        if decoded < start_frame:
            decoded += 1
            continue
        if len(out) >= num_frames:
            break
        img = frame.to_image().resize((width, height), Image.LANCZOS)
        out.append(np.asarray(img, dtype=np.float32) / 255.0)
        decoded += 1
    container.close()
    if not out:
        return np.zeros((0, height, width, 3), dtype=np.float32)
    return np.stack(out, axis=0)


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a - b) ** 2))
    return 50.0 if mse < 1e-10 else float(10.0 * np.log10(1.0 / mse))


def main():
    p = argparse.ArgumentParser(description="True autoregressive long-horizon drift diagnostic on LongCat")
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-videos", type=int, default=8)
    p.add_argument("--start-video-idx", type=int, default=0)
    p.add_argument("--chunk-size", type=int, default=0, help="video-list chunking for parallel jobs (0=all)")
    p.add_argument("--seed", type=int, default=42)

    # Rollout geometry. chunk_gen chosen so (chunk_gen-1) %% 4 == 0 => VAE
    # temporal packing decodes EXACTLY chunk_gen frames (no length drift).
    p.add_argument("--num-cond-frames", type=int, default=13,
                   help="overlap frames fed back as conditioning (LongCat native=13)")
    p.add_argument("--chunk-gen", type=int, default=17,
                   help="new frames generated per AR chunk ((chunk_gen-1) multiple of 4)")
    p.add_argument("--num-chunks", type=int, default=8,
                   help="number of autoregressive chunks (horizon = num_chunks*chunk_gen frames)")
    p.add_argument("--num-inference-steps", type=int, default=50)
    p.add_argument("--guidance-scale", type=float, default=4.0)
    p.add_argument("--resolution", type=str, default="480p")
    p.add_argument("--no-cfg", action="store_true")

    p.add_argument("--eval-gt", action="store_true",
                   help="also compute best-effort per-chunk PSNR vs the real future")
    p.add_argument("--save-videos", action="store_true",
                   help="save the full stitched rollout mp4 per video")
    p.add_argument("--fixed-caption", type=str, default=None)
    args = p.parse_args()

    if (args.chunk_gen - 1) % 4 != 0:
        print(f"WARNING: (chunk_gen-1)={args.chunk_gen-1} not a multiple of 4; "
              f"decoded chunk length may differ from --chunk-gen (handled dynamically).")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    height = 480 if args.resolution == "480p" else 720
    width = 832 if args.resolution == "480p" else 1280
    vae_t = 4

    os.makedirs(args.output_dir, exist_ok=True)
    videos_dir = os.path.join(args.output_dir, "videos")
    if args.save_videos:
        os.makedirs(videos_dir, exist_ok=True)

    print("=" * 70)
    print("Long-horizon DRIFT rollout on LongCat (TRUE autoregressive)")
    print("=" * 70)
    print("  cond=%d  chunk_gen=%d  num_chunks=%d  horizon=%d gen frames"
          % (args.num_cond_frames, args.chunk_gen, args.num_chunks,
             args.chunk_gen * args.num_chunks))
    print("  steps=%d  cfg=%s  guidance=%.1f  res=%s"
          % (args.num_inference_steps, not args.no_cfg, args.guidance_scale, args.resolution))
    print("=" * 70)

    components = load_longcat_components(args.checkpoint_dir, device=device, dtype=torch.bfloat16)
    dit, vae = components["dit"], components["vae"]
    scheduler = components["scheduler"]
    tokenizer, text_encoder = components["tokenizer"], components["text_encoder"]
    for m in (dit, vae, text_encoder):
        for prm in m.parameters():
            prm.requires_grad = False
    dit.eval(); vae.eval(); text_encoder.eval()

    engine = SAViDNO_LongCat(
        dit=dit, vae=vae, scheduler=scheduler,
        tokenizer=tokenizer, text_encoder=text_encoder,
        device=device, dtype=torch.bfloat16,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        gradient_checkpointing=False,
    )

    eval_videos = load_ucf101_video_list(
        args.data_dir, max_videos=args.max_videos, seed=args.seed, validate_decodable=True)
    eval_videos = apply_fixed_caption(eval_videos, args.fixed_caption, context="drift")
    if args.start_video_idx > 0 or args.chunk_size > 0:
        end = len(eval_videos)
        if args.chunk_size > 0:
            end = min(args.start_video_idx + args.chunk_size, end)
        eval_videos = eval_videos[args.start_video_idx:end]

    ckpt_path = os.path.join(args.output_dir, "checkpoint.json")
    ckpt = load_checkpoint(ckpt_path)
    start_idx = ckpt.get("next_idx", 0) if ckpt else 0
    per_video = ckpt.get("per_video", []) if ckpt else []

    save_video_from_numpy = None
    if args.save_videos:
        from lora_experiment.scripts.run_lora_tta import save_video_from_numpy

    for idx, entry in enumerate(tqdm(eval_videos, desc="drift")):
        if idx < start_idx:
            continue
        video_path = entry["video_path"]
        caption = entry["caption"]
        video_name = Path(video_path).stem
        try:
            prompt_embeds, prompt_mask = encode_prompt(
                tokenizer, text_encoder, caption, device=device, dtype=torch.bfloat16)

            # Chunk 0 conditions on the real first num_cond frames.
            pixel_cond = load_video_frames(
                video_path, args.num_cond_frames, height=height, width=width, start_frame=0
            ).to(device, torch.bfloat16)                       # [1,3,cond,H,W] in [-1,1]

            # Running trajectory in pixel space [0,1] as a torch tensor [3,T,H,W]
            # on device; start with the conditioning frames themselves so seams
            # can be measured from the first generated chunk.
            running = ((pixel_cond.squeeze(0).float() + 1.0) / 2.0).clamp(0, 1)  # [3,cond,H,W]

            chunk_frames_hwc = []   # list of [t,H,W,C] per chunk (gen only)
            abs_start = args.num_cond_frames   # absolute source index of chunk 0's first gen frame
            gt_psnr_per_chunk = []
            gt_n_per_chunk = []
            t0 = time.time()

            for c in range(args.num_chunks):
                # conditioning = last num_cond frames of the running trajectory
                cond_pix = running[:, -args.num_cond_frames:].unsqueeze(0)   # [1,3,cond,H,W] in [0,1]
                cond_pix = (cond_pix * 2.0 - 1.0).to(device, torch.bfloat16)
                cond_latents = encode_video(vae, cond_pix, normalize=True)

                T_gen_latent = 1 + (args.chunk_gen - 1) // vae_t
                target_shape = (1, cond_latents.shape[1], T_gen_latent,
                                cond_latents.shape[3], cond_latents.shape[4])

                gen = torch.Generator(device=device).manual_seed(
                    args.seed + idx * 1000 + c)
                pred_pixels = sample_chunk(
                    engine, cond_latents, target_shape, prompt_embeds, prompt_mask,
                    use_cfg=not args.no_cfg, generator=gen)               # [1,3,t,H,W] in [0,1]

                gen_hwc = _pixels_to_hwc(pred_pixels)                     # [t,H,W,C]
                chunk_frames_hwc.append(gen_hwc)

                # append to running trajectory (as [3,t,H,W])
                gen_t = torch.from_numpy(gen_hwc.transpose(3, 0, 1, 2)).to(
                    running.device, running.dtype)
                running = torch.cat([running, gen_t], dim=1)

                # best-effort GT PSNR for this chunk
                if args.eval_gt:
                    gt = _load_gt_hwc(video_path, abs_start, gen_hwc.shape[0], height, width)
                    n = min(len(gt), gen_hwc.shape[0])
                    if n > 0:
                        gt_psnr_per_chunk.append(_psnr(gen_hwc[:n], gt[:n]))
                        gt_n_per_chunk.append(int(n))
                    else:
                        gt_psnr_per_chunk.append(float("nan"))
                        gt_n_per_chunk.append(0)
                abs_start += gen_hwc.shape[0]

                del pred_pixels, cond_latents, cond_pix
                torch_gc()

            elapsed = time.time() - t0

            # ---- per-chunk GT-free signals + seams --------------------------
            per_chunk = {k: [] for k in
                         ("sharpness", "colorfulness", "saturation", "brightness", "motion")}
            per_frame_all = {k: [] for k in per_chunk}
            seams = []
            prev_last = None
            for ci, gen_hwc in enumerate(chunk_frames_hwc):
                sig = per_frame_signals(gen_hwc)
                for k in per_chunk:
                    vals = sig[k]
                    per_frame_all[k].extend(vals)
                    per_chunk[k].append(_nanmean(vals))
                # cross-chunk seam vs within-chunk adjacency
                first = gen_hwc[0, ..., :3]
                if prev_last is not None:
                    seam = float(np.mean(np.abs(first - prev_last)))
                    within = _nanmean(sig["motion"])   # mean adjacent diff within chunk
                    seams.append({"chunk": ci, "seam": seam, "within": within,
                                  "ratio": (seam / within) if within and not np.isnan(within) else float("nan")})
                prev_last = gen_hwc[-1, ..., :3]

            rec = {
                "idx": idx, "video_name": video_name, "video_path": video_path,
                "caption": caption, "success": True,
                "num_chunks": len(chunk_frames_hwc),
                "chunk_lengths": [int(f.shape[0]) for f in chunk_frames_hwc],
                "gen_time": elapsed,
                "per_chunk": per_chunk,
                "seams": seams,
            }
            if args.eval_gt:
                rec["gt_psnr_per_chunk"] = gt_psnr_per_chunk
                rec["gt_n_per_chunk"] = gt_n_per_chunk

            if args.save_videos and save_video_from_numpy is not None:
                stitched = np.concatenate(chunk_frames_hwc, axis=0)
                out_path = os.path.join(videos_dir, f"{video_name}_rollout.mp4")
                save_video_from_numpy(stitched, out_path, fps=15)
                rec["output_path"] = out_path

            per_video.append(rec)
            # concise console read: sharpness + motion first vs last chunk
            def _fl(key):
                v = per_chunk[key]
                return (v[0], v[-1]) if v else (float("nan"), float("nan"))
            sh0, shN = _fl("sharpness"); mo0, moN = _fl("motion")
            print("    %s: sharp %.4f->%.4f  motion %.4f->%.4f  (%d chunks, %.1fs)"
                  % (video_name, sh0, shN, mo0, moN, len(chunk_frames_hwc), elapsed))

            del running
            torch_gc()
        except Exception as ex:
            import traceback
            traceback.print_exc()
            per_video.append({"idx": idx, "video_name": video_name,
                              "video_path": video_path, "error": str(ex), "success": False})

        save_checkpoint({"next_idx": idx + 1, "per_video": per_video}, ckpt_path)

    successful = [r for r in per_video if r.get("success")]
    summary = {
        "diagnostic": "longhorizon_drift_rollout",
        "backbone": "longcat",
        "mode": "true_autoregressive",
        "num_cond_frames": args.num_cond_frames,
        "chunk_gen": args.chunk_gen,
        "num_chunks": args.num_chunks,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "use_cfg": not args.no_cfg,
        "seed": args.seed,
        "num_videos": len(per_video),
        "num_successful": len(successful),
        "per_video": per_video,
    }
    save_results(summary, os.path.join(args.output_dir, "drift_stats.json"))
    print("\n" + "=" * 70)
    print("Drift rollout complete: %d/%d ok -> %s/drift_stats.json"
          % (len(successful), len(per_video), args.output_dir))
    print("Plot with: python comparison_methods/scripts/plot_drift_curves.py "
          "--stats %s/drift_stats.json --out <png>" % args.output_dir)
    print("=" * 70)


if __name__ == "__main__":
    main()
