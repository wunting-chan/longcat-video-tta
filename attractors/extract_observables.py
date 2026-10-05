#!/usr/bin/env python3
"""Phase -1 of the attractor study: per-frame observables for existing 30 s rollouts.

For every mp4 in each method directory this writes one npz with, per frame:
  cls   [T, 768] fp16  frozen DINOv2 ViT-B/14 [CLS] (224 center crop, ImageNet norm)
  low   [T, K]   fp32  low-level observables (names in LOW_NAMES)
plus the sidecar's per-chunk chosen candidate (search arms) and prefix length.
Read-only on the inputs; writes only under --out.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch

REAL = False
LOW_NAMES = ["brightness", "contrast", "saturation", "colorfulness", "sharpness",
             "spectral_slope", "diff_energy", "flow_mag"]


def spectral_slope(g):
    f = np.abs(np.fft.fftshift(np.fft.fft2(g - g.mean()))) ** 2
    h, w = g.shape
    y, x = np.indices((h, w))
    r = np.hypot(y - h / 2, x - w / 2).astype(int)
    rad = np.bincount(r.ravel(), f.ravel()) / np.maximum(np.bincount(r.ravel()), 1)
    k = np.arange(len(rad))
    sel = (k >= 2) & (k <= min(h, w) // 2 - 1)
    return float(np.polyfit(np.log(k[sel]), np.log(rad[sel] + 1e-12), 1)[0])


def low_level(frames):
    """frames: uint8 [T, H, W, 3] RGB."""
    out = np.zeros((len(frames), len(LOW_NAMES)), np.float32)
    prev_small = None
    for t, fr in enumerate(frames):
        small = cv2.resize(fr, (208, 120), interpolation=cv2.INTER_AREA)
        mid = cv2.resize(fr, (416, 240), interpolation=cv2.INTER_AREA)
        g = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        gm = cv2.cvtColor(mid, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
        r, gg, b = [small[..., i].astype(np.float32) for i in range(3)]
        rg, yb = r - gg, 0.5 * (r + gg) - b
        colorful = np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
        out[t, 0] = g.mean()
        out[t, 1] = g.std()
        out[t, 2] = hsv[..., 1].mean() / 255.0
        out[t, 3] = colorful / 255.0
        out[t, 4] = cv2.Laplacian(gm, cv2.CV_32F).var()
        out[t, 5] = spectral_slope(g)
        if prev_small is not None:
            out[t, 6] = np.abs(g - prev_small).mean()
            flow = cv2.calcOpticalFlowFarneback((prev_small * 255).astype(np.uint8), (g * 255).astype(np.uint8),
                                                None, 0.5, 3, 15, 3, 5, 1.2, 0)
            out[t, 7] = np.linalg.norm(flow, axis=-1).mean()
        else:
            out[t, 6] = np.nan
            out[t, 7] = np.nan
        prev_small = g
    return out


def load_frames(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(frames)


def real_frames(gen_mp4):
    """Real control: the generated clip's own 33-frame real opening, then the true source future
    after the 33 encoded source frames, time-resampled to 504 frames (as in score_v2v_pixel_metrics)."""
    side = json.load(open(Path(gen_mp4).with_suffix(".json")))
    gen = load_frames(gen_mp4)
    pre = int(side.get("prefix_pix", 33))
    cap = cv2.VideoCapture(side["video_path"])
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_src = int(round((len(gen) - pre) / 16.0 * fps))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    n_src = max(2, min(n_src, total - 33)) if total > 0 else n_src
    want = np.linspace(0, n_src - 1, len(gen) - pre).round().astype(int) + 33
    need = {}
    for k, fi in enumerate(want):
        need.setdefault(int(fi), []).append(k)
    h, w = gen.shape[1:3]
    tail = np.zeros((len(gen) - pre, h, w, 3), np.uint8)
    i, last = 0, None
    while i <= want[-1]:
        ok, fr = cap.read()
        if not ok:
            break
        if i in need:
            last = cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB), (w, h), interpolation=cv2.INTER_AREA)
            for k in need[i]:
                tail[k] = last
        i += 1
    cap.release()
    if i <= want[-1] and last is not None:          # short source: hold the last frame
        filled = max(k for fi, ks in need.items() if fi < i for k in ks)
        tail[filled + 1:] = last
    return np.concatenate([gen[:pre], tail])


def work(mp4):
    """CPU worker: decode, low-level observables, and 224 center crops for DINOv2."""
    frames = real_frames(mp4) if REAL else load_frames(mp4)
    low = low_level(frames)
    h, w = frames.shape[1:3]
    s = 224 / min(h, w)
    nh, nw = round(h * s), round(w * s)
    crops = np.stack([cv2.resize(f, (nw, nh), interpolation=cv2.INTER_CUBIC) for f in frames])
    top, left = (nh - 224) // 2, (nw - 224) // 2
    return mp4, low, np.ascontiguousarray(crops[:, top:top + 224, left:left + 224])


@torch.no_grad()
def dino_features(model, crops, device, bs=256):
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    feats = []
    for i in range(0, len(crops), bs):
        x = torch.from_numpy(crops[i:i + bs]).to(device).permute(0, 3, 1, 2).float() / 255.0
        x = (x - mean) / std
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
            feats.append(model(x).float().cpu())
    return torch.cat(feats).numpy().astype(np.float16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series-dir", required=True)
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--real", action="store_true", help="build the real-future control from the first method's clips")
    a = ap.parse_args()
    global REAL
    REAL = a.real
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", source="github", trust_repo=True).to(device).eval()
    t0 = time.time()
    for m in a.methods:
        src = Path(a.series_dir) / f"{m}_h30s_shard0"
        dst = Path(a.out) / ("real" if a.real else m)
        dst.mkdir(parents=True, exist_ok=True)
        mp4s = sorted(src.glob("*.mp4"))[: a.limit or None]
        todo = [str(p) for p in mp4s if not (dst / f"{p.stem}.npz").exists()]
        with ProcessPoolExecutor(a.workers) as ex:
            for mp4, low, crops in ex.map(work, todo):
                p = Path(mp4)
                cls = dino_features(model, crops, device)
                side = json.load(open(p.with_suffix(".json")))
                chosen = [c.get("chosen_cand") for c in side.get("chunks", [])]
                np.savez_compressed(dst / f"{p.stem}.npz", cls=cls, low=low, low_names=np.array(LOW_NAMES),
                                    chosen=np.array([-1 if c is None else c for c in chosen]),
                                    prefix_pix=side.get("prefix_pix", 0), fps=16, stem=side.get("stem", p.stem))
        print(m, len(mp4s), f"{time.time() - t0:.0f}s", flush=True)
    print("ALL_DONE")


if __name__ == "__main__":
    main()
