#!/usr/bin/env python3
"""Per-video RAFT flow *distribution-shape* features for H-T1-4 gating.

Computes flow statistics on the TTA-visible window [0, 48) at 256×320
(RAFT-compatible resolution), matching ``extract_video_features_for_tta.py``
frame geometry for ``panda_1000v_standard``.

Unlike ``compute_dynamic_degree.py`` (28 frames, p99-as-max), this script
emits the shape features the gating plan needs:

    mean_flow          — mean ||flow|| over all pixels and frame pairs
    flow_max           — global max ||flow|| (not p99)
    flow_entropy       — Shannon entropy (bits) of the flow-magnitude histogram
    flow_max_over_mean — flow_max / mean_flow (concentration proxy)

Output CSV schema:
    video_id, n_frames_used, tta_visible_range,
    mean_flow, flow_max, flow_entropy, flow_max_over_mean,
    n_frame_pairs, input_size_h, input_size_w, flow_model

Run:
    python3 scripts/extract_flow_shape_features.py \\
        --videos-dir datasets/panda_1000_480p \\
        --output sweep_experiment/reports/per_video_analysis/2026-06-09/flow_shape_features.csv
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

# Frame geometry — matches extract_video_features_for_tta.py / TTA runners.
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
    if ":" in arg:
        a, b = arg.split(":", 1)
        return int(a), int(b)
    raise argparse.ArgumentTypeError(
        f"frame-range arg must be 'auto' or 'A:B', got {arg!r}"
    )


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


def decode_window(
    video_path: str,
    start_frame: int,
    num_frames: int,
    resize_hw: Tuple[int, int],
) -> np.ndarray:
    """Decode [start, start+num_frames) as uint8 (T, H, W, 3) RGB."""
    import av
    import torch
    import torch.nn.functional as F

    container = av.open(video_path)
    frames: List[np.ndarray] = []
    decoded = 0
    try:
        for frame in container.decode(video=0):
            if decoded < start_frame:
                decoded += 1
                continue
            if len(frames) >= num_frames:
                break
            frames.append(frame.to_ndarray(format="rgb24"))
            decoded += 1
    finally:
        container.close()

    if not frames:
        raise ValueError(f"No frames decoded from {video_path} at start={start_frame}")
    while len(frames) < num_frames:
        frames.append(frames[-1].copy())
    stacked = np.stack(frames[:num_frames], axis=0)
    h_t, w_t = resize_hw
    if stacked.shape[1] != h_t or stacked.shape[2] != w_t:
        t = torch.from_numpy(stacked).permute(0, 3, 1, 2).float()
        t = F.interpolate(t, size=(h_t, w_t), mode="bilinear", align_corners=False)
        stacked = t.permute(0, 2, 3, 1).clamp(0, 255).numpy().astype(np.uint8)
    return np.ascontiguousarray(stacked)


class RaftFlowShapeEstimator:
    """RAFT-small with flow distribution-shape statistics."""

    def __init__(self, device: str = "cuda", hist_bins: int = 64):
        import torch
        from torchvision.models.optical_flow import raft_small, Raft_Small_Weights

        self.torch = torch
        self.device = torch.device(device)
        self.hist_bins = hist_bins
        weights = Raft_Small_Weights.DEFAULT
        self.transforms = weights.transforms()
        self.model = raft_small(weights=weights, progress=False).to(self.device)
        self.model.eval()
        self.name = "raft_small"

    def flow_shape_stats(self, frames_uint8: np.ndarray) -> Dict[str, float]:
        torch = self.torch
        if frames_uint8.shape[0] < 2:
            return {
                "mean_flow": float("nan"),
                "flow_max": float("nan"),
                "flow_entropy": float("nan"),
                "flow_max_over_mean": float("nan"),
                "n_frame_pairs": 0,
            }

        t = torch.from_numpy(frames_uint8).permute(0, 3, 1, 2).float() / 255.0
        img1 = t[:-1]
        img2 = t[1:]
        img1, img2 = self.transforms(img1, img2)
        img1 = img1.to(self.device)
        img2 = img2.to(self.device)

        with torch.inference_mode():
            flow = self.model(img1, img2)[-1]
        mag = torch.sqrt(flow[:, 0] ** 2 + flow[:, 1] ** 2)

        mean_flow = float(mag.mean().item())
        flow_max = float(mag.max().item())
        flat = mag.detach().cpu().numpy().ravel()
        if flat.size == 0 or mean_flow <= 0:
            flow_entropy = float("nan")
            flow_max_over_mean = float("nan")
        else:
            hist, _ = np.histogram(flat, bins=self.hist_bins, range=(0.0, float(flat.max()) + 1e-6))
            total = float(hist.sum())
            if total <= 0:
                flow_entropy = float("nan")
            else:
                p = hist[hist > 0].astype(np.float64) / total
                flow_entropy = float(-(p * np.log2(p)).sum())
            flow_max_over_mean = float(flow_max / mean_flow)

        return {
            "mean_flow": mean_flow,
            "flow_max": flow_max,
            "flow_entropy": flow_entropy,
            "flow_max_over_mean": flow_max_over_mean,
            "n_frame_pairs": int(flow.shape[0]),
        }


def _fieldnames() -> List[str]:
    return [
        "video_id", "n_frames_used", "tta_visible_range",
        "mean_flow", "flow_max", "flow_entropy", "flow_max_over_mean",
        "n_frame_pairs", "input_size_h", "input_size_w", "flow_model",
    ]


def _format_row(row: dict, fieldnames: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k in fieldnames:
        v = row.get(k)
        if v is None:
            out[k] = ""
        elif isinstance(v, float):
            out[k] = "" if (math.isnan(v) or math.isinf(v)) else f"{v:.6f}"
        else:
            out[k] = str(v)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--tta-visible-frames", type=str, default="auto")
    ap.add_argument("--target-size", type=str, default="256x320",
                    help="HxW before RAFT (divisible by 8).")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    H, W = (int(x) for x in args.target_size.lower().split("x"))
    if H % 8 != 0 or W % 8 != 0:
        print("[error] target-size must be divisible by 8 for RAFT", file=sys.stderr)
        return 2

    visible_range = _parse_frame_range_arg(args.tta_visible_frames, AUTO_TTA_VISIBLE_RANGE)
    n_visible = visible_range[1] - visible_range[0]

    existing: Dict[str, dict] = {}
    if args.resume and args.output.exists():
        with args.output.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                vid = (r.get("video_id") or "").strip()
                if vid:
                    existing[vid] = r

    video_paths = list_video_paths(args.videos_dir)
    if args.limit:
        video_paths = video_paths[: args.limit]
    if not video_paths:
        print(f"[error] no videos under {args.videos_dir}", file=sys.stderr)
        return 2

    print(f"Videos: {len(video_paths)}  window={visible_range}  resize={H}x{W}")
    estimator = RaftFlowShapeEstimator(device=args.device)
    fieldnames = _fieldnames()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    rows_by_id = dict(existing)
    n_done, n_skip, n_err = 0, 0, 0
    t0 = time.time()

    for i, vp in enumerate(video_paths):
        vid = _canonical_video_id(vp.name)
        if vid in rows_by_id and args.resume:
            n_skip += 1
            continue
        try:
            frames = decode_window(str(vp), visible_range[0], n_visible, (H, W))
            stats = estimator.flow_shape_stats(frames)
        except Exception as exc:  # noqa: BLE001
            print(f"[error] {vp.name}: {exc}", file=sys.stderr)
            n_err += 1
            continue
        rows_by_id[vid] = {
            "video_id": vid,
            "n_frames_used": n_visible,
            "tta_visible_range": f"{visible_range[0]}:{visible_range[1]}",
            "input_size_h": H,
            "input_size_w": W,
            "flow_model": estimator.name,
            **stats,
        }
        n_done += 1
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(video_paths)}] new={n_done} skip={n_skip} err={n_err}",
                  flush=True)

    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for vid in sorted(rows_by_id):
            writer.writerow(_format_row(rows_by_id[vid], fieldnames))

    print(f"Wrote {args.output}  new={n_done} skip={n_skip} err={n_err} "
          f"elapsed={time.time()-t0:.1f}s")
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
