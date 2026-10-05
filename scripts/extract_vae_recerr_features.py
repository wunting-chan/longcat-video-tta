#!/usr/bin/env python3
"""Per-video LongCat-VAE round-trip reconstruction error for H-T1-1 gating.

Encodes the TTA-visible window [0, 48) through the base VAE and decodes
back to pixels. Reports:

    rec_err_l1    — mean |recon − input| in [0, 1] pixel space
    rec_err_lpips — mean LPIPS(recon, input) when ``lpips`` is installed;
                    falls back to MSE when not available

Output CSV:
    video_id, n_visible_frames, tta_visible_range,
    rec_err_l1, rec_err_lpips, lpips_available

Run:
    python3 scripts/extract_vae_recerr_features.py \\
        --checkpoint-dir /scratch/$USER/longcat-video-checkpoints \\
        --videos-dir datasets/panda_1000_480p \\
        --output sweep_experiment/reports/per_video_analysis/2026-06-09/vae_recerr_features.csv
"""
from __future__ import annotations

import argparse
import csv
import math
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "delta_experiment" / "scripts"))
sys.path.insert(0, str(_REPO_ROOT))

TTA_TOTAL_FRAMES: int = 48
GEN_START_FRAME: int = 48
AUTO_TTA_VISIBLE_RANGE: Tuple[int, int] = (
    max(0, GEN_START_FRAME - TTA_TOTAL_FRAMES),
    GEN_START_FRAME,
)

_CANONICAL_PREFIX_RE = re.compile(r"^([A-Za-z][A-Za-z0-9]*_\d+)")


def _canonical_video_id(s: Optional[str]) -> str:
    if not s:
        return ""
    stem = Path(str(s)).stem
    m = _CANONICAL_PREFIX_RE.match(stem)
    return m.group(1) if m else stem


def _parse_frame_range_arg(arg: str, default: Tuple[int, int]) -> Tuple[int, int]:
    if not arg or arg.lower() == "auto":
        return default
    a, b = arg.split(":", 1)
    return int(a), int(b)


def list_video_paths(videos_dir: Path) -> List[Path]:
    candidates: List[Path] = []
    subdir = videos_dir / "videos"
    if subdir.is_dir():
        for ext in ("*.mp4", "*.avi"):
            candidates.extend(subdir.glob(ext))
    if not candidates:
        for ext in ("*.mp4", "*.avi"):
            candidates.extend(videos_dir.rglob(ext))
    return sorted(candidates, key=lambda p: _canonical_video_id(p.name))


class _LPIPSHelper:
    def __init__(self, device: str):
        import lpips
        import torch

        self.torch = torch
        self.net = lpips.LPIPS(net="alex").to(device).eval()

    def mean_distance(self, orig_01: "torch.Tensor", recon_01: "torch.Tensor") -> float:
        """orig/recon: [1, 3, T, H, W] in [0, 1]."""
        torch = self.torch
        t = min(orig_01.shape[2], recon_01.shape[2])
        vals: List[float] = []
        with torch.inference_mode():
            for i in range(t):
                d = self.net(
                    orig_01[:, :, i] * 2 - 1,
                    recon_01[:, :, i] * 2 - 1,
                )
                vals.append(float(d.mean().item()))
        return float(np.mean(vals)) if vals else float("nan")


def compute_rec_errors(
    vae,
    pixel_frames: "torch.Tensor",
    device: str,
    lpips_helper: Optional[_LPIPSHelper],
) -> Tuple[float, float, bool]:
    """pixel_frames: [1, 3, T, H, W] in [-1, 1]."""
    from common import decode_latents, encode_video

    import torch

    latents = encode_video(vae, pixel_frames, normalize=True)
    recon = decode_latents(vae, latents, denorm=True)
    orig_01 = (pixel_frames + 1.0) / 2.0
    # Wan VAE round-trip shortens T (48 px -> 12 lat -> 45 px at scale 4).
    t_cmp = min(orig_01.shape[2], recon.shape[2])
    orig_01 = orig_01[:, :, :t_cmp]
    recon = recon[:, :, :t_cmp]
    rec_err_l1 = float((recon - orig_01).abs().mean().item())

    if lpips_helper is not None:
        rec_err_lpips = lpips_helper.mean_distance(orig_01, recon)
        return rec_err_l1, rec_err_lpips, True

    rec_err_mse = float(((recon - orig_01) ** 2).mean().item())
    return rec_err_l1, rec_err_mse, False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint-dir", type=str, required=True)
    ap.add_argument("--videos-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--tta-visible-frames", type=str, default="auto")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--max-videos", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    visible = _parse_frame_range_arg(args.tta_visible_frames, AUTO_TTA_VISIBLE_RANGE)
    n_visible = visible[1] - visible[0]
    fieldnames = [
        "video_id", "n_visible_frames", "tta_visible_range",
        "rec_err_l1", "rec_err_lpips", "lpips_available",
    ]

    existing: Dict[str, dict] = {}
    if args.resume and args.output.exists():
        with args.output.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                vid = (r.get("video_id") or "").strip()
                if vid:
                    existing[vid] = r

    paths = list_video_paths(args.videos_dir)
    if args.max_videos > 0:
        paths = paths[: args.max_videos]
    todo = [
        vp for vp in paths
        if _canonical_video_id(vp.name) not in existing or not args.resume
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print("Loading LongCat VAE...")
    try:
        from common import load_longcat_components, load_video_frames
    except ImportError as exc:
        print(f"[error] common import failed: {exc}", file=sys.stderr)
        return 2
    import torch

    components = load_longcat_components(
        args.checkpoint_dir, device=args.device, dtype=torch.bfloat16,
    )
    vae = components["vae"]
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    lpips_helper: Optional[_LPIPSHelper] = None
    try:
        lpips_helper = _LPIPSHelper(args.device)
        print("LPIPS available (alex net)")
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] LPIPS unavailable ({exc}); rec_err_lpips uses MSE fallback",
              file=sys.stderr)

    rows = dict(existing)
    n_done, n_err = 0, 0
    t0 = time.time()
    tta_start = max(0, visible[0])

    for i, vp in enumerate(todo):
        vid = _canonical_video_id(vp.name)
        try:
            pixel = load_video_frames(
                str(vp),
                num_frames=n_visible,
                height=480,
                width=832,
                start_frame=tta_start,
            ).to(args.device, torch.bfloat16)
            l1, lp, has_lpips = compute_rec_errors(
                vae, pixel, args.device, lpips_helper,
            )
            rows[vid] = {
                "video_id": vid,
                "n_visible_frames": n_visible,
                "tta_visible_range": f"{visible[0]}:{visible[1]}",
                "rec_err_l1": l1,
                "rec_err_lpips": lp,
                "lpips_available": int(has_lpips),
            }
            n_done += 1
            torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001
            print(f"[error] {vp.name}: {exc}", file=sys.stderr)
            traceback.print_exc()
            n_err += 1

        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(todo)}] done={n_done} err={n_err}", flush=True)

    with args.output.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for vid in sorted(rows):
            row = rows[vid]
            out = {}
            for k in fieldnames:
                v = row.get(k)
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    out[k] = ""
                elif isinstance(v, float):
                    out[k] = f"{v:.6f}"
                else:
                    out[k] = "" if v is None else str(v)
            w.writerow(out)

    print(f"Wrote {args.output}  new={n_done} err={n_err} elapsed={time.time()-t0:.1f}s")
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
