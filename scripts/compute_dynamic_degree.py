#!/usr/bin/env python3
"""Compute per-video dynamicness scores using RAFT optical flow.

This implements the underlying continuous score used by VBench's
"Dynamic Degree" metric (Huang et al., VBench, CVPR 2024) and adopted as
the standard "motion" measure across recent video-generation papers
(Stable Video Diffusion, AnimateDiff, ModelScopeT2V, CogVideoX, ...).

For each video we compute pairwise dense optical flow between consecutive
frames using RAFT-small (torchvision), then report:

    mean_flow_mag : mean over all pixels and all frame pairs of
                    || (u(x,y), v(x,y)) ||_2  (pixels-per-frame)
    max_flow_mag  : mean over frame pairs of the per-frame 99th-percentile
                    flow magnitude (robust max, suppresses single-pixel outliers)

The continuous mean_flow_mag is what we use for the dynamicness-vs-metric
correlation plot. The max_flow_mag is reported for VBench-style binary
thresholding compatibility (VBench thresholds raw max flow at ~6 pixels).

Output JSON:
    {
      "model": "raft_small",
      "max_frames_used": 28,
      "input_size": [256, 320],
      "videos": {
        "00001.mp4": {"mean_flow": 5.32, "max_flow": 12.5,
                      "n_frame_pairs": 27, "h": 480, "w": 852},
        ...
      }
    }

Falls back to OpenCV Farnebäck flow (CPU) if RAFT is unavailable.

Run:
    python scripts/compute_dynamic_degree.py \
        --videos-dir /scratch/$USER/longcat-video-tta/datasets/panda_1000_480p \
        --output-json datasets/panda_1000_480p/dynamic_degree.json \
        --max-frames 28
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Frame loading (cv2 is fast + standard; ffmpeg fallback could be added later)
# ---------------------------------------------------------------------------
def _load_video_frames(
    path: Path, max_frames: int, target_hw: Tuple[int, int]
) -> Optional[np.ndarray]:
    """Return [T, H, W, 3] uint8 RGB array, or None on failure."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None

    H_tgt, W_tgt = target_hw
    frames: List[np.ndarray] = []
    while len(frames) < max_frames:
        ret, bgr = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (W_tgt, H_tgt), interpolation=cv2.INTER_AREA)
        frames.append(rgb)
    cap.release()

    if len(frames) < 2:
        return None
    return np.stack(frames, axis=0)


# ---------------------------------------------------------------------------
# RAFT-small flow (preferred)
# ---------------------------------------------------------------------------
class RaftFlowEstimator:
    """Wraps torchvision's raft_small for batched pairwise flow."""

    def __init__(self, device: str = "cuda"):
        import torch
        from torchvision.models.optical_flow import raft_small, Raft_Small_Weights

        self.torch = torch
        self.device = torch.device(device)
        weights = Raft_Small_Weights.DEFAULT
        self.transforms = weights.transforms()
        self.model = raft_small(weights=weights, progress=False).to(self.device)
        self.model.eval()
        self.name = "raft_small"

    def flow_magnitude(self, frames_uint8: np.ndarray) -> Tuple[float, float, int]:
        """Compute (mean_flow, max_flow, n_pairs) for a [T, H, W, 3] uint8 video."""
        torch = self.torch
        if frames_uint8.shape[0] < 2:
            return float("nan"), float("nan"), 0

        # [T, 3, H, W] in [0, 1]
        t = torch.from_numpy(frames_uint8).permute(0, 3, 1, 2).float() / 255.0
        img1 = t[:-1]
        img2 = t[1:]
        img1, img2 = self.transforms(img1, img2)
        img1 = img1.to(self.device)
        img2 = img2.to(self.device)

        with torch.inference_mode():
            flow = self.model(img1, img2)[-1]  # [N, 2, H, W]
        mag = torch.sqrt(flow[:, 0] ** 2 + flow[:, 1] ** 2)  # [N, H, W]

        mean_flow = mag.mean().item()
        # Per-frame 99th-percentile, then mean over frames (robust "max" score)
        per_frame_p99 = torch.quantile(
            mag.flatten(1), q=0.99, dim=1
        )  # [N]
        max_flow = per_frame_p99.mean().item()
        n_pairs = flow.shape[0]
        return mean_flow, max_flow, n_pairs


# ---------------------------------------------------------------------------
# OpenCV Farnebäck fallback
# ---------------------------------------------------------------------------
class FarnebackFlowEstimator:
    def __init__(self):
        self.name = "farneback"

    def flow_magnitude(self, frames_uint8: np.ndarray) -> Tuple[float, float, int]:
        import cv2

        if frames_uint8.shape[0] < 2:
            return float("nan"), float("nan"), 0

        gray = np.stack(
            [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames_uint8], axis=0
        )
        mean_acc, max_acc, n = 0.0, 0.0, 0
        for i in range(gray.shape[0] - 1):
            flow = cv2.calcOpticalFlowFarneback(
                gray[i], gray[i + 1],
                flow=None,
                pyr_scale=0.5, levels=3, winsize=15, iterations=3,
                poly_n=5, poly_sigma=1.2, flags=0,
            )
            mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            mean_acc += float(mag.mean())
            max_acc += float(np.quantile(mag, 0.99))
            n += 1
        return mean_acc / n, max_acc / n, n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _build_estimator(prefer: str = "raft", device: str = "cuda"):
    if prefer == "farneback":
        return FarnebackFlowEstimator()
    try:
        return RaftFlowEstimator(device=device)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] RAFT init failed ({e!s}); falling back to Farnebäck (CPU).",
              file=sys.stderr, flush=True)
        return FarnebackFlowEstimator()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos-dir", required=True, type=Path,
                    help="Directory containing *.mp4 evaluation videos")
    ap.add_argument("--output-json", required=True, type=Path,
                    help="Destination JSON path")
    ap.add_argument("--max-frames", type=int, default=28,
                    help="Max frames per video to process. For standard "
                         "horizon, 28 = full clip. For long horizon use 76.")
    ap.add_argument("--target-size", type=str, default="256x320",
                    help="Resize HxW before flow (divisible by 8). Default 256x320.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--method", default="raft", choices=["raft", "farneback"])
    ap.add_argument("--glob", default="*.mp4")
    ap.add_argument("--limit", type=int, default=None,
                    help="Optional: cap number of videos (for smoke testing).")
    args = ap.parse_args()

    H, W = (int(x) for x in args.target_size.lower().split("x"))
    assert H % 8 == 0 and W % 8 == 0, "target-size must be divisible by 8 for RAFT"

    videos = sorted(args.videos_dir.glob(args.glob))
    if args.limit:
        videos = videos[: args.limit]
    if not videos:
        print(f"[error] no videos matched {args.videos_dir}/{args.glob}",
              file=sys.stderr)
        return 2

    print(f"Found {len(videos)} videos in {args.videos_dir}")
    estimator = _build_estimator(prefer=args.method, device=args.device)
    print(f"Using flow estimator: {estimator.name}")

    out = {
        "model": estimator.name,
        "max_frames_used": args.max_frames,
        "input_size": [H, W],
        "videos_dir": str(args.videos_dir),
        "videos": {},
    }

    t_start = time.time()
    n_ok, n_fail = 0, 0
    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    for i, vp in enumerate(videos):
        frames = _load_video_frames(vp, args.max_frames, (H, W))
        if frames is None:
            n_fail += 1
            out["videos"][vp.name] = {"error": "decode_failed"}
            continue
        try:
            mean_f, max_f, n_pairs = estimator.flow_magnitude(frames)
        except Exception as e:  # noqa: BLE001
            n_fail += 1
            out["videos"][vp.name] = {"error": f"flow_failed:{e!s}"}
            continue

        out["videos"][vp.name] = {
            "mean_flow": mean_f,
            "max_flow": max_f,
            "n_frame_pairs": n_pairs,
            "h_orig": int(frames.shape[1]),
            "w_orig": int(frames.shape[2]),
        }
        n_ok += 1

        if (i + 1) % 50 == 0 or (i + 1) == len(videos):
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta = (len(videos) - i - 1) / max(rate, 1e-6)
            print(f"  [{i+1:>4}/{len(videos)}]  ok={n_ok}  fail={n_fail}  "
                  f"rate={rate:.2f} vid/s  eta={eta:.0f}s", flush=True)

    with open(args.output_json, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nWrote {args.output_json}")
    print(f"  ok={n_ok}  fail={n_fail}  total={len(videos)}  "
          f"elapsed={time.time()-t_start:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
