#!/usr/bin/env python3
"""Per-video spatiotemporal FFT high-frequency energy for H-T1-3 gating.

CPU-only script. Computes on the TTA-visible window [0, 48) luma channel:

    hf_energy_ratio_3d            — HF / total energy in 3D rFFT magnitude
    hf_energy_ratio_spatial_only  — mean per-frame spatial HF / total energy

Output CSV:
    video_id, n_frames_used, tta_visible_range,
    hf_energy_ratio_3d, hf_energy_ratio_spatial_only

Run:
    python3 scripts/extract_fft_features.py \\
        --videos-dir datasets/panda_1000_480p \\
        --output sweep_experiment/reports/per_video_analysis/2026-06-09/fft_features.csv
"""
from __future__ import annotations

import argparse
import csv
import math
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

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


def decode_luma_stack(path: Path, start: int, n_frames: int) -> np.ndarray:
    import av

    container = av.open(str(path))
    frames: List[np.ndarray] = []
    decoded = 0
    try:
        for frame in container.decode(video=0):
            if decoded < start:
                decoded += 1
                continue
            if len(frames) >= n_frames:
                break
            rgb = frame.to_ndarray(format="rgb24")
            gray = (
                0.299 * rgb[:, :, 0].astype(np.float32)
                + 0.587 * rgb[:, :, 1]
                + 0.114 * rgb[:, :, 2]
            )
            frames.append(gray)
            decoded += 1
    finally:
        container.close()
    if not frames:
        raise ValueError(f"No frames from {path}")
    while len(frames) < n_frames:
        frames.append(frames[-1].copy())
    return np.stack(frames[:n_frames], axis=0)


def _hf_mask(shape: Tuple[int, ...], frac: float = 0.5) -> np.ndarray:
    """True for frequency bins in the upper half along each axis."""
    masks = []
    for dim, n in enumerate(shape):
        idx = np.arange(n)
        if dim == len(shape) - 1 and len(shape) == 3:
            # rFFT last axis is half-sized
            cutoff = int(n * frac)
            masks.append(idx >= cutoff)
        else:
            cutoff = int(n * frac)
            masks.append(idx >= cutoff)
    mg = np.meshgrid(*masks, indexing="ij")
    return mg[0] & mg[1] & mg[2]


def hf_ratio_3d(luma: np.ndarray, frac: float = 0.5) -> float:
    spec = np.fft.rfftn(luma.astype(np.float64))
    mag2 = np.abs(spec) ** 2
    total = float(mag2.sum())
    if total <= 0:
        return float("nan")
    hf = _hf_mask(mag2.shape, frac=frac)
    return float(mag2[hf].sum() / total)


def hf_ratio_spatial_mean(luma: np.ndarray, frac: float = 0.5) -> float:
    ratios: List[float] = []
    for t in range(luma.shape[0]):
        spec = np.fft.rfft2(luma[t].astype(np.float64))
        mag2 = np.abs(spec) ** 2
        total = float(mag2.sum())
        if total <= 0:
            continue
        h, w = mag2.shape
        hf = (np.arange(h)[:, None] >= int(h * frac)) & (
            np.arange(w)[None, :] >= int(w * frac)
        )
        ratios.append(float(mag2[hf].sum() / total))
    return float(np.mean(ratios)) if ratios else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--videos-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--tta-visible-frames", type=str, default="auto")
    ap.add_argument("--hf-frac", type=float, default=0.5,
                    help="Fraction of spectrum treated as high-frequency.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    visible = _parse_frame_range_arg(args.tta_visible_frames, AUTO_TTA_VISIBLE_RANGE)
    n_visible = visible[1] - visible[0]
    fieldnames = [
        "video_id", "n_frames_used", "tta_visible_range",
        "hf_energy_ratio_3d", "hf_energy_ratio_spatial_only",
    ]

    existing: Dict[str, dict] = {}
    if args.resume and args.output.exists():
        with args.output.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                vid = (r.get("video_id") or "").strip()
                if vid:
                    existing[vid] = r

    paths = list_video_paths(args.videos_dir)
    if args.limit:
        paths = paths[: args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    rows = dict(existing)
    n_done, n_err = 0, 0
    t0 = time.time()

    for i, vp in enumerate(paths):
        vid = _canonical_video_id(vp.name)
        if vid in rows and args.resume:
            continue
        try:
            luma = decode_luma_stack(vp, visible[0], n_visible)
            rows[vid] = {
                "video_id": vid,
                "n_frames_used": n_visible,
                "tta_visible_range": f"{visible[0]}:{visible[1]}",
                "hf_energy_ratio_3d": hf_ratio_3d(luma, frac=args.hf_frac),
                "hf_energy_ratio_spatial_only": hf_ratio_spatial_mean(
                    luma, frac=args.hf_frac
                ),
            }
            n_done += 1
        except Exception as exc:  # noqa: BLE001
            print(f"[error] {vp.name}: {exc}", file=sys.stderr)
            n_err += 1
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(paths)}] done={n_done} err={n_err}", flush=True)

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
