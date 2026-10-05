#!/usr/bin/env python3
"""Per-video bits-per-pixel (bpp) features for H-T1-2 gating.

Computes lossless-compression bpp proxies on the TTA-visible window [0, 48):

    bpp_h264   — container bits-per-pixel from ffprobe (file_size / frame count / pixels)
    bpp_png_avg — mean per-frame PNG encode size / (H×W) over visible frames

Output CSV:
    video_id, n_frames_used, tta_visible_range, bpp_h264, bpp_png_avg,
    file_size_bytes, container_frame_count, frame_h, frame_w

Run:
    python3 scripts/extract_bpp_features.py \\
        --videos-dir datasets/panda_1000_480p \\
        --output sweep_experiment/reports/per_video_analysis/2026-06-09/bpp_features.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
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


def ffprobe_info(path: Path) -> Tuple[int, int, int, int]:
    """Return (file_size_bytes, nb_frames, width, height)."""
    size = path.stat().st_size
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_streams", "-show_format", str(path),
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True)
        data = json.loads(out)
    except (subprocess.CalledProcessError, json.JSONDecodeError, FileNotFoundError):
        return size, 0, 0, 0

    nb_frames = 0
    width = height = 0
    for stream in data.get("streams", []):
        if stream.get("codec_type") != "video":
            continue
        width = int(stream.get("width") or 0)
        height = int(stream.get("height") or 0)
        if stream.get("nb_frames"):
            nb_frames = int(stream["nb_frames"])
        elif stream.get("duration") and stream.get("avg_frame_rate"):
            num, den = stream["avg_frame_rate"].split("/")
            fps = float(num) / max(float(den), 1e-6)
            nb_frames = int(float(stream["duration"]) * fps)
        break
    if nb_frames <= 0:
        nb_frames = max(1, size // max(width * height, 1))
    return size, nb_frames, width, height


def _ensure_uint8_hwc_stack(arr: np.ndarray) -> np.ndarray:
    """Coerce decoded frames to contiguous uint8 (T, H, W, 3) for PNG encode."""
    if hasattr(arr, "detach") and hasattr(arr, "cpu") and hasattr(arr, "numpy"):
        arr = arr.detach().cpu().numpy()
    if isinstance(arr, (list, tuple)):
        arr = np.stack([np.asarray(x) for x in arr], axis=0)
    arr = np.asarray(arr)
    if arr.ndim == 4 and arr.shape[1] == 3 and arr.shape[-1] != 3:
        arr = np.transpose(arr, (0, 2, 3, 1))
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            finite_max = float(np.nanmax(arr)) if arr.size else 0.0
            if finite_max <= 1.5:
                arr = arr * 255.0
            arr = np.clip(arr, 0.0, 255.0)
        arr = arr.astype(np.uint8, copy=False)
    return np.ascontiguousarray(arr)


def decode_window_rgb(path: Path, start: int, n_frames: int) -> np.ndarray:
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
            img = np.asarray(frame.to_ndarray(format="rgb24"))
            frames.append(np.ascontiguousarray(img.astype(np.uint8, copy=False)))
            decoded += 1
    finally:
        container.close()
    if not frames:
        raise ValueError(f"No frames decoded from {path}")
    while len(frames) < n_frames:
        frames.append(frames[-1].copy())
    return _ensure_uint8_hwc_stack(np.stack(frames[:n_frames], axis=0))


def png_bpp_mean(frames_rgb: np.ndarray) -> float:
    from io import BytesIO

    from PIL import Image

    frames_rgb = _ensure_uint8_hwc_stack(frames_rgb)
    h, w = frames_rgb.shape[1:3]
    bpps: List[float] = []
    for t in range(frames_rgb.shape[0]):
        buf = BytesIO()
        Image.fromarray(frames_rgb[t]).save(buf, format="PNG")
        bpps.append(8.0 * len(buf.getvalue()) / max(h * w, 1))
    return float(np.mean(bpps)) if bpps else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--videos-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--tta-visible-frames", type=str, default="auto")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    visible = _parse_frame_range_arg(args.tta_visible_frames, AUTO_TTA_VISIBLE_RANGE)
    n_visible = visible[1] - visible[0]
    fieldnames = [
        "video_id", "n_frames_used", "tta_visible_range",
        "bpp_h264", "bpp_png_avg",
        "file_size_bytes", "container_frame_count", "frame_h", "frame_w",
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
            fsize, nb_frames, w, h = ffprobe_info(vp)
            pixels = max(w * h, 1)
            bpp_h264 = (8.0 * fsize) / (max(nb_frames, 1) * pixels)
            frames = decode_window_rgb(vp, visible[0], n_visible)
            bpp_png = png_bpp_mean(frames)
            rows[vid] = {
                "video_id": vid,
                "n_frames_used": n_visible,
                "tta_visible_range": f"{visible[0]}:{visible[1]}",
                "bpp_h264": bpp_h264,
                "bpp_png_avg": bpp_png,
                "file_size_bytes": fsize,
                "container_frame_count": nb_frames,
                "frame_h": h,
                "frame_w": w,
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
